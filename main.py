import os
import json
import wandb
import torch
import random
import numpy as np
from datetime import datetime
from ppo.ppo import PPO
from ppo.ppo_utils import Memory, WelfordNormalizer
from config import get_config, classify_and_return_args
import torch.multiprocessing as mp
from sweep import HyperParameterTuner
from torch.utils.tensorboard import SummaryWriter
from utils import *
from plots import *
from simulation.control_env import ControlEnv
from simulation.design_env import DesignEnv
from simulation.worker import parallel_eval_worker

def train(train_config, is_sweep=False, sweep_config=None):
    """
    The lower level (control) agent is present inside the design environment as lower_ppo.
    lower level actor parallelization occurs inside step function of design environment. 
    Policy updates in both agents are serial/ centralized.
    """
    SEED = train_config['seed'] if train_config['seed'] else random.randint(0, 1000000)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda") if train_config['gpu'] and torch.cuda.is_available() else torch.device("cpu") 
    print(f"\nRandom seed: {SEED} \nUsing device: {device}")

    # Set and save hyperparameters 
    if is_sweep:
        for key, value in sweep_config.items():
            train_config[key] = value

    os.makedirs('runs', exist_ok=True)
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    run_dir = os.path.join('runs', current_time)
    os.makedirs(run_dir, exist_ok=True)
    design_args, control_args, higher_ppo_args, lower_ppo_args, eval_args = classify_and_return_args(train_config, device)

    # Print stats from dummy environment
    dummy_envs = {
        'higher' : DesignEnv(design_args, control_args, lower_ppo_args, run_dir), 
        'lower' : ControlEnv(control_args, run_dir, worker_id=None)
    }

    print(f"\nEnvironments")
    for env_type, env in dummy_envs.items():
        print(f"\n{env_type}-level:")
        print(f"\tObservation space: {env.observation_space}")
        print(f"\tObservation space shape: {env.observation_space.shape}")
        print(f"\tAction space: {str(env.action_space).replace('\n', '\n\t')}")
        if env_type == 'lower':
            print(f"\tAction dimension: {train_config['lower_action_dim']}")
        else: 
            print(f"\tIn channels: {train_config['higher_in_channels']}, Action dimension: {train_config['max_proposals']}\n")

    # use the dummy env shapes to init normalizers
    lower_obs_shape = dummy_envs['lower'].observation_space.shape

    eval_args['lower_state_dim'] = lower_obs_shape
    dummy_envs['lower'].close() 
    dummy_envs['higher'].close()

    # Model saving and tensorboard 
    writer = SummaryWriter(log_dir=run_dir)
    eval_args['eval_save_dir'] = os.path.join(run_dir, f'results/train_{current_time}')
    os.makedirs(eval_args['eval_save_dir'], exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'saved_policies'), exist_ok=True)
    design_args.update({'save_dir': run_dir})
    higher_ppo_args['model_kwargs'].update({'run_dir': run_dir})
    control_args.update({'global_seed': SEED})
    control_args.update({'total_action_timesteps_per_episode': train_config['lower_max_timesteps'] // train_config['lower_action_duration']})

    # Instead of using total_episodes, we will use total_iterations. 
    # Every iteration, num_process control agents interact with the environment for total_action_timesteps_per_episode steps (which further internally contains action_duration steps)
    total_iterations = train_config['total_timesteps'] // (train_config['lower_max_timesteps'] * train_config['lower_num_processes'])
    total_updates_higher = total_iterations // train_config['higher_update_freq']
    total_updates_lower = (train_config['total_timesteps'] / train_config['lower_action_duration']) // train_config['lower_update_freq']

    higher_ppo = PPO(**higher_ppo_args)
    higher_env = DesignEnv(design_args, control_args, lower_ppo_args, run_dir)
    higher_env.total_updates_lower = total_updates_lower
    higher_env.lower_state_normalizer = WelfordNormalizer(lower_obs_shape)
    higher_ppo.policy_old = higher_ppo.policy_old.to(device)
    higher_ppo.policy = higher_ppo.policy.to(device)

    higher_state = higher_env.reset()
    print(f"\nHigher state at reset: {higher_state}")
    higher_memories = Memory()

    higher_update_count = 0
    best_higher_reward = float('-inf') 
    best_higher_loss = float('inf')
    best_higher_eval = float('inf')
    eval_ped_avg_arrival = float('inf')
    higher_loss = {
        'policy_loss': float('inf'),
        'value_loss': float('inf'),
        'entropy_loss': float('inf'),
        'total_loss': float('inf'),
        'approx_kl': float('inf')
    }

    lower_avg_eval = 200.0 # arbitrary large numbers
    eval_veh_avg_wait = 200.0
    eval_ped_avg_wait = 200.0
    current_lr_higher = higher_ppo_args['lr']
    save_config(higher_ppo_args, lower_ppo_args, control_args, design_args, run_dir)

    for iteration in range(1, total_iterations + 1): #start from 1, else policy gets updated at step 0.
        print(f"\nStarting iteration: {iteration}/{total_iterations} with {higher_env.global_step} total steps so far\n")

        # Higher level agent takes node features, edge index, edge attributes and batch (to make single large graph) as input 
        # To produce padded fixed-sized actions num_actual_proposals is also returned.
        higher_ppo.policy_old.eval()
        original_proposals, merged_proposals, num_proposals, higher_logprob = higher_ppo.policy_old.act(higher_state, 
                                                                                        iteration, 
                                                                                        design_args['clamp_min'], 
                                                                                        design_args['clamp_max'], 
                                                                                        device,
                                                                                        training=True,
                                                                                        visualize=True) 
        
        # Since the higher agent internally takes a step where a number of parallel lower agents take their own steps, 
        # We return things relevant to both the higher and lower agents. First, for higher.
        higher_next_state, higher_reward_norm, higher_reward_unnorm, higher_done, info = higher_env.step(merged_proposals, # Act on the environment with merged proposals
                                                                                     num_proposals, 
                                                                                     iteration)
        
        # Get value from critic network
        with torch.no_grad():
            critic_output = higher_ppo.policy_old.critic(higher_state, device=device)
            # print(f"Critic output shape: {critic_output.shape}")
            # print(f"Critic output: {critic_output}")
            # The critic returns a tensor of shape [1], so we can directly call .item()
            higher_value = critic_output.item()
        # print(f"\n\n\nNum proposals: {num_proposals}\n\n\n")
        # Append to memory, the original proposals. Get reward based on merged proposals. Merging operation = environment physics.
        # Add num_proposals for code re-use and consistency but dont make use of it in the higher agent.
        higher_memories.append(higher_state, original_proposals, num_proposals, higher_value, higher_logprob, higher_reward_norm, higher_done) 

        if iteration % design_args['higher_update_freq'] == 0:
            print(f"Updating Higher PPO with {len(higher_memories.actions)} memories") 
            higher_update_count += 1

            if design_args['higher_anneal_lr']:
                current_lr_higher = higher_ppo.update_learning_rate(higher_update_count, total_updates_higher)

            avg_higher_reward = sum(higher_memories.rewards) / len(higher_memories.rewards)
            with torch.no_grad():
                bootstrap_value = higher_ppo.policy_old.critic(higher_next_state, device=device).item()
            higher_loss = higher_ppo.update(higher_memories, bootstrap_value=bootstrap_value)
            del higher_memories # Reset memory
            higher_memories = Memory()

            # eval the policies every now and then. 
            if higher_update_count % design_args['eval_freq'] == 0:
                policy_path = os.path.join(design_args['save_dir'], f'saved_policies/policy_at_{higher_env.global_step}.pth')
                save_policy(higher_ppo.policy, 
                            higher_env.lower_ppo.policy, 
                            higher_env.lower_state_normalizer, 
                            higher_env.normalizer_x, 
                            higher_env.normalizer_y, 
                            policy_path)
                
                # Evaluate the latest policies
                # At the time when higher agent is saved, lower agent is also exposed to designs from the same distribution.
                print(f"Evaluating policies: {policy_path} at step {higher_env.global_step}")
                eval_json = eval(design_args, 
                                 control_args, 
                                 higher_ppo_args, 
                                 lower_ppo_args, 
                                 eval_args, 
                                 global_step=higher_env.global_step,
                                 policy_path=policy_path)
                
                # calculate metrics for both policies
                _, lower_avg_veh_wait, lower_avg_ped_wait, higher_avg_ped_arrival, _, _, _ = get_averages(eval_json)

                # Get a single evaluation metric for both agents.
                eval_veh_avg_wait = np.mean(lower_avg_veh_wait)
                eval_ped_avg_wait = np.mean(lower_avg_ped_wait)
                eval_ped_avg_arrival = np.mean(higher_avg_ped_arrival)
                lower_avg_eval = ((eval_veh_avg_wait + eval_ped_avg_wait) / 2)
                print(f"Evaluation results: \n\tHigher: {eval_ped_avg_arrival} \n\tLower: {lower_avg_eval}")

            # save the policies at every update
            if avg_higher_reward > best_higher_reward: 
                save_policy(higher_ppo.policy, 
                            higher_env.lower_ppo.policy, 
                            higher_env.lower_state_normalizer, 
                            higher_env.normalizer_x, 
                            higher_env.normalizer_y, 
                            os.path.join(design_args['save_dir'], 
                            'saved_policies/best_reward_policy.pth'))
                best_higher_reward = avg_higher_reward

            if higher_loss['total_loss'] < best_higher_loss:
                save_policy(higher_ppo.policy, 
                            higher_env.lower_ppo.policy, 
                            higher_env.lower_state_normalizer, 
                            higher_env.normalizer_x, 
                            higher_env.normalizer_y, 
                            os.path.join(design_args['save_dir'], 
                            'saved_policies/best_loss_policy.pth'))
                best_higher_loss = higher_loss['total_loss']

            if eval_ped_avg_arrival < best_higher_eval:
                save_policy(higher_ppo.policy, 
                            higher_env.lower_ppo.policy, 
                            higher_env.lower_state_normalizer, 
                            higher_env.normalizer_x, 
                            higher_env.normalizer_y, 
                            os.path.join(design_args['save_dir'], 
                            'saved_policies/best_eval_policy.pth'))
                best_higher_eval = eval_ped_avg_arrival

        # logging at every iteration (every time sample is drawn)
        if is_sweep:
            wandb.log(data = {
                "iteration": iteration,
                "global_step": higher_env.global_step,
                "higher/avg_reward (normalized)": higher_reward_norm,
                "higher/avg_reward (unnormalized)": higher_reward_unnorm,
                "higher/update_count": higher_update_count,
                "higher/current_lr": current_lr_higher if train_config['higher_anneal_lr'] else higher_ppo_args['lr'],
                "higher/losses/policy_loss": higher_loss['policy_loss'],
                "higher/losses/value_loss": higher_loss['value_loss'],
                "higher/losses/entropy_loss": higher_loss['entropy_loss'],
                "higher/losses/total_loss": higher_loss['total_loss'],
                "higher/approx_kl": higher_loss['approx_kl'],
                "evals/avg_ped_arrival": eval_ped_avg_arrival,
                "lower/avg_reward (normalized)": info['lower_avg_reward_norm'],
                "lower/avg_reward (unnormalized)": info['lower_avg_reward_unnorm'],
                "lower/update_count": info['lower_update_count'],
                "lower/current_lr": info['lower_current_lr'],
                "lower/losses/policy_loss": info['lower_policy_loss'],
                "lower/losses/value_loss": info['lower_value_loss'],
                "lower/losses/entropy_loss": info['lower_entropy_loss'],
                "lower/losses/total_loss": info['lower_total_loss'],
                "lower/approx_kl": info['lower_approx_kl'],
                "evals/lower_ped_avg_wait": eval_ped_avg_wait,
                "evals/lower_veh_avg_wait": eval_veh_avg_wait,
                "evals/lower_avg_eval": lower_avg_eval }, 
                step=higher_env.global_step)
        else:
            writer.add_scalar('Iteration', iteration, higher_env.global_step)
            writer.add_scalar('Higher/Average_Reward (normalized)', higher_reward_norm, higher_env.global_step)
            writer.add_scalar('Higher/Average_Reward (unnormalized)', higher_reward_unnorm, higher_env.global_step)
            writer.add_scalar('Higher/Update_Count', higher_update_count, higher_env.global_step)
            writer.add_scalar('Higher/Current_LR', current_lr_higher if train_config['higher_anneal_lr'] else higher_ppo_args['lr'], higher_env.global_step)
            writer.add_scalar('Higher/Losses/Policy_Loss', higher_loss['policy_loss'], higher_env.global_step)
            writer.add_scalar('Higher/Losses/Value_Loss', higher_loss['value_loss'], higher_env.global_step)
            writer.add_scalar('Higher/Losses/Entropy_Loss', higher_loss['entropy_loss'], higher_env.global_step)
            writer.add_scalar('Higher/Losses/Total_Loss', higher_loss['total_loss'], higher_env.global_step)
            writer.add_scalar('Higher/Approx_KL', higher_loss['approx_kl'], higher_env.global_step)
            writer.add_scalar('Evaluation/Avg_Ped_Arrival', eval_ped_avg_arrival, higher_env.global_step)
            writer.add_scalar('Lower/Average_Reward (normalized)', info['lower_avg_reward_norm'], higher_env.global_step)
            writer.add_scalar('Lower/Average_Reward (unnormalized)', info['lower_avg_reward_unnorm'], higher_env.global_step)
            writer.add_scalar('Lower/Update_Count', info['lower_update_count'], higher_env.global_step)
            writer.add_scalar('Lower/Current_LR', info['lower_current_lr'], higher_env.global_step)
            writer.add_scalar('Lower/Losses/Policy_Loss', info['lower_policy_loss'], higher_env.global_step)
            writer.add_scalar('Lower/Losses/Value_Loss', info['lower_value_loss'], higher_env.global_step)
            writer.add_scalar('Lower/Losses/Entropy_Loss', info['lower_entropy_loss'], higher_env.global_step)
            writer.add_scalar('Lower/Losses/Total_Loss', info['lower_total_loss'], higher_env.global_step)
            writer.add_scalar('Lower/Approx_KL', info['lower_approx_kl'], higher_env.global_step)
            writer.add_scalar('Evaluation/Avg_Veh_Wait', eval_veh_avg_wait, higher_env.global_step)
            writer.add_scalar('Evaluation/Avg_Ped_Wait', eval_ped_avg_wait, higher_env.global_step)
            writer.add_scalar('Evaluation/Avg_Eval', lower_avg_eval, higher_env.global_step)

        higher_state = higher_next_state

    if is_sweep:
        wandb.finish()
    else:
        writer.close() # TODO: close writer for both agents?

def eval(design_args, 
         control_args, 
         higher_ppo_args, 
         lower_ppo_args, 
         eval_args, 
         policy_path=None, 
         global_step=None,
         tl=False, 
         unsignalized=False,
         real_world=False):
    """
    Evaluate both higher and lower level policies (during training or stand-alone evaluation)
    - For higher agent, its just a single step (single greedy design) 
    - For lower agent, it includes benchmarks like tl and unsignalized
    - Each demand is run on a different worker
    - Results returned as json  
    """
    
    if tl:
        if unsignalized:
            tl_state = "unsignalized"
        else:
            tl_state = "tl"
    else:
        tl_state = "ppo"
    
    n_workers = eval_args['eval_lower_workers'] 
    n_iterations = eval_args['eval_n_iterations']
    eval_device = torch.device("cuda") if eval_args['eval_worker_device']=='gpu' and torch.cuda.is_available() else torch.device("cpu")
    eval_demand_scales = eval_args['in_range_demand_scales'] + eval_args['out_of_range_demand_scales']
    all_results = {}

    eval_ppo_lower = PPO(**lower_ppo_args)
    eval_ppo_higher = PPO(**higher_ppo_args)
    lower_state_normalizer = WelfordNormalizer(eval_args['lower_state_dim'])

    higher_env = DesignEnv(design_args, control_args, lower_ppo_args, higher_ppo_args['model_kwargs']['run_dir']) # Pass the correct run_dir
    if policy_path:
        norm_x, norm_y = load_policy(eval_ppo_higher.policy, eval_ppo_lower.policy, lower_state_normalizer, policy_path)
        higher_env.normalizer_x = norm_x # Then replace them
        higher_env.normalizer_y = norm_y
        result_json_path = os.path.join(eval_args['eval_save_dir'], f'{policy_path.split("/")[-1].split(".")[0]}_{tl_state}.json')
    else: 
        result_json_path = os.path.join(eval_args['eval_save_dir'], f'realworld_{tl_state}.json')

    higher_policy = eval_ppo_higher.policy.to(eval_device)
    shared_lower_policy = eval_ppo_lower.policy.to(eval_device)
    shared_lower_policy.share_memory()
    shared_lower_policy.eval()
    higher_policy.eval()
    lower_state_normalizer.eval()
    higher_state = higher_env.reset() # At reset, we get the original real-world configuration.

    if real_world:
        iteration = "0"
        sumo_net_file = f"{higher_env.network_dir}/network_iteration_0.net.xml"
        num_proposals = 7 # not used.
    else: 
        iteration = f"eval{global_step}"
        # Which is the input network to the act function
        _, merged_proposals, num_proposals, _ = higher_policy.act(higher_state,
                                                                iteration, 
                                                                design_args['clamp_min'], 
                                                                design_args['clamp_max'], 
                                                                eval_device,
                                                                training=False, # !important
                                                                visualize=True) 

        # Convert tensor action to proposals
        merged_proposals = merged_proposals.cpu().numpy()  # Convert to numpy array if it's not already
        proposals = merged_proposals[0][:num_proposals]  # Only consider the actual proposals
        # print(f"\nProposals: {proposals}")
        
        # Apply the action to output the latest SUMO network file
        higher_env._apply_action(proposals, iteration) # Pass the actual proposals derived above
        sumo_net_file = higher_env.current_net_file_path #f"{higher_env.network_dir}/network_iteration_{iteration}.net.xml"

    # number of times the n_workers have to be repeated to cover all eval demands
    num_times_workers_recycle = len(eval_demand_scales) if len(eval_demand_scales) < n_workers else (len(eval_demand_scales) // n_workers) + 1
    for i in range(num_times_workers_recycle):
        start = n_workers * i   
        end = n_workers * (i + 1)
        demand_scales_evaluated_current_cycle = eval_demand_scales[start: end]

        eval_queue = mp.Queue()
        eval_processes = []  
        active_eval_workers = []
        demand_scale_to_rank = {}
        for rank, demand_scale in enumerate(demand_scales_evaluated_current_cycle): 
            demand_scale_to_rank[demand_scale] = rank
            print(f"For demand: {demand_scale}")    
            worker_config = {
                'network_iteration': iteration,
                'n_iterations': n_iterations,
                'total_action_timesteps_per_episode': eval_args['eval_lower_timesteps'] // control_args['lower_action_duration'], # Each time
                'worker_demand_scale': demand_scale,
                'num_proposals': num_proposals,
                'lower_policy': shared_lower_policy,
                'control_args': control_args,
                'worker_device': eval_device,
                'lower_state_normalizer': lower_state_normalizer,
                'run_dir': higher_ppo_args['model_kwargs']['run_dir']
            }
            p = mp.Process(
                target=parallel_eval_worker,
                args=(rank, worker_config, eval_queue, sumo_net_file, higher_env.extreme_edge_dict, tl, unsignalized, real_world))
            
            p.start()
            eval_processes.append(p)
            active_eval_workers.append(rank)

        while active_eval_workers:
            worker_demand_scale, result = eval_queue.get() #timeout=60) # Result is obtained after all iterations are complete
            # print(f"\nResult from worker with demand scale: {worker_demand_scale}: {result}\n")
            all_results[worker_demand_scale] = result
            active_eval_workers.remove(demand_scale_to_rank[worker_demand_scale])

        for p in eval_processes:
            p.join()

    # if torch.cuda.is_available():
    #     torch.cuda.empty_cache()
    del eval_queue
    del higher_policy
    del shared_lower_policy
    # print(f"\nAll results: {all_results}\n")    
    with open(result_json_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    print(f"Results saved to JSON path: {result_json_path}")
    return result_json_path
    
def main(config):
    """
    Cannot create a bunch of connections in main and then pass them around. 
    Because each new worker needs a separate pedestrian and vehicle trips file.
    """
    # Set the start method for multiprocessing. It does not create a process itself but sets the method for creating a process.
    # Spawn means create a new process. There is a fork method as well which will create a copy of the current process.
    mp.set_start_method('spawn') 
    if config['evaluate']: 
        # TODO: Load from saved config

        # setup params
        device = torch.device("cuda") if config['eval_worker_device']=='gpu' and torch.cuda.is_available() else torch.device("cpu")
        design_args, control_args, higher_ppo_args, lower_ppo_args, eval_args = classify_and_return_args(config, device)
        dummy_env = ControlEnv(control_args, " ", worker_id=None)

        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        # Correctly determine the run directory from the model path
        eval_model_path = config['eval_model_path']
        run_dir_parent = os.path.dirname(os.path.dirname(eval_model_path)) # e.g., runs/Apr12_17-40-27
        higher_ppo_args['model_kwargs']['run_dir'] = run_dir_parent
        print(f"Run directory: {run_dir_parent}")
        
        eval_save_dir = os.path.join(run_dir_parent, 'results', f'eval_{current_time}')
        print(f"Eval save directory: {eval_save_dir}")
        os.makedirs(eval_save_dir, exist_ok=True)
        eval_args['eval_save_dir'] = eval_save_dir
        eval_args['lower_state_dim'] = dummy_env.observation_space.shape
        
        real_world_design_unsignalized_results_path = ""
        # Evaluate the real-world design in the unsignalized setting. A control policy was never trained on the real-world design. 
        # real_world_design_unsignalized_results_path = eval(design_args, 
        #                                                    control_args, 
        #                                                    higher_ppo_args, 
        #                                                    lower_ppo_args, 
        #                                                    eval_args, 
        #                                                    policy_path=None, 
        #                                                    global_step="_final", 
        #                                                    tl=True, 
        #                                                    unsignalized=True, 
        #                                                    real_world=True) 
        
        # Evaluate the ``new design`` in the all three settings. The new design network has to be same across all three settings.
        # Load the policy params (model sizes etc) from the config file. 
        config_file_path = os.path.join(run_dir_parent, 'config.json')
        print(f"Config file path: {config_file_path}")
        with open(config_file_path, 'r') as f:
            saved_config = json.load(f)['hyperparameters']
        higher_ppo_args = saved_config['higher_ppo_args']
        lower_ppo_args = saved_config['lower_ppo_args']
        higher_ppo_args['model_kwargs']['run_dir'] = run_dir_parent

        new_design_ppo_results_path = eval(design_args, 
                                           control_args, 
                                           higher_ppo_args, 
                                           lower_ppo_args, 
                                           eval_args, 
                                           policy_path=config['eval_model_path'], 
                                           global_step="_final")
        
        # new_design_tl_results_path = eval(design_args,
        #                                    control_args, 
        #                                    higher_ppo_args, 
        #                                    lower_ppo_args, 
        #                                    eval_args, 
        #                                    policy_path=config['eval_model_path'], # Although this wont use the policy, provide path.
        #                                    global_step="_final", 
        #                                    tl=True) 

        # new_design_unsignalized_results_path = eval(design_args, 
        #                                             control_args, 
        #                                             higher_ppo_args, 
        #                                             lower_ppo_args, 
        #                                             eval_args, 
        #                                             policy_path=config['eval_model_path'], # Although this wont use the policy, provide path.
        #                                             global_step="_final", 
        #                                             tl=True, 
        #                                             unsignalized=True)

        # plot_control_results(new_design_unsignalized_results_path,  
        #                   new_design_tl_results_path,
        #                   new_design_ppo_results_path,
        #                   in_range_demand_scales = eval_args['in_range_demand_scales'])
        
        # plot_design_results(new_design_unsignalized_results_path, 
        #                   real_world_design_unsignalized_results_path,
        #                   in_range_demand_scales = eval_args['in_range_demand_scales'])
        
        plot_design_and_control_results(
            design_unsig_path = new_design_unsignalized_results_path,
            realworld_unsig_path = real_world_design_unsignalized_results_path,
            control_tl_path = new_design_tl_results_path,
            control_ppo_path = new_design_ppo_results_path,
            in_range_demand_scales = eval_args['in_range_demand_scales']
        )

    elif config['sweep']:
        tuner = HyperParameterTuner(config, train)
        tuner.start()
    else:
        train(config)

if __name__ == "__main__":
    config = get_config()
    main(config)

    