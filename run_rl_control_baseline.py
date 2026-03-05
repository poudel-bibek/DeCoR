#!/usr/bin/env python3
"""
RL Control Baseline: evaluate the trained control policy on the FIXED
real-world 7-crosswalk layout across all demand scales.

Uses the existing parallel_eval_worker infrastructure with real_world=True.
"""
import os, sys, json, random
import numpy as np
import torch
import torch.multiprocessing as mp

os.environ['SUMO_HOME'] = '/usr/share/sumo'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_config, classify_and_return_args
from ppo.ppo import PPO
from ppo.ppo_utils import WelfordNormalizer
from simulation.design_env import DesignEnv
from simulation.worker import parallel_eval_worker


def main():
    mp.set_start_method('spawn', force=True)

    config = get_config()
    config['gui'] = False
    config['gpu'] = False
    config['evaluate'] = True
    device = torch.device('cpu')

    design_args, control_args, higher_ppo_args, lower_ppo_args, eval_args = \
        classify_and_return_args(config, device)

    run_dir = './runs/baselines_experiment/rl_control_baseline'
    for d in ['', '/components', '/network_iterations']:
        os.makedirs(run_dir + d, exist_ok=True)
    design_args['save_dir'] = run_dir
    design_args['save_graph_images'] = False
    design_args['save_gmm_plots'] = False
    higher_ppo_args['model_kwargs']['run_dir'] = run_dir

    # Load checkpoint
    ckpt_path = config['eval_model_path']
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')

    # Build control policy
    lower_ppo_args_eval = dict(lower_ppo_args)
    lower_ppo_args_eval['action_dim'] = config['max_proposals'] + 4
    lower_ppo_args_eval['device'] = 'cpu'
    control_ppo = PPO(**lower_ppo_args_eval)
    control_ppo.policy.load_state_dict(ckpt['lower']['state_dict'])
    control_ppo.policy_old.load_state_dict(ckpt['lower']['state_dict'])
    control_ppo.policy.eval()
    control_ppo.policy_old.eval()
    control_ppo.policy_old.share_memory()

    # Build state normalizer
    state_shape = (1, 1, config['lower_action_duration'],
                   config['lower_per_timestep_state_dim'])
    state_normalizer = WelfordNormalizer(shape=state_shape)
    state_normalizer.mean = ckpt['lower']['state_normalizer_mean']
    state_normalizer.M2 = ckpt['lower']['state_normalizer_M2']
    state_normalizer.count = ckpt['lower']['state_normalizer_count']

    # Init DesignEnv to get extreme_edge_dict
    print("Initializing DesignEnv...")
    env = DesignEnv(design_args, control_args, lower_ppo_args, run_dir)
    env.reset()
    extreme_edge_dict = env.extreme_edge_dict
    net_file = config['original_net_file']
    num_proposals_rw = 7

    total_action_steps = config['eval_lower_timesteps'] // config['lower_action_duration']
    control_args['total_action_timesteps_per_episode'] = total_action_steps

    all_demand_scales = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75]
    n_iterations = 5
    results = {}

    print("\n=== RL Control on Real-World Layout (7 CW) ===")
    for scale in all_demand_scales:
        eval_config = {
            'control_args': dict(control_args),
            'lower_policy': control_ppo.policy_old,
            'lower_state_normalizer': state_normalizer,
            'worker_demand_scale': scale,
            'n_iterations': n_iterations,
            'run_dir': run_dir,
            'network_iteration': 0,
            'total_action_timesteps_per_episode': total_action_steps,
            'num_proposals': num_proposals_rw,
            'worker_device': device,
        }

        eval_queue = mp.Queue()
        p = mp.Process(
            target=parallel_eval_worker,
            args=(0, eval_config, eval_queue, net_file, extreme_edge_dict),
            kwargs={'tl': False, 'unsignalized': False, 'real_world': True}
        )
        p.start()
        p.join(timeout=300)

        if not eval_queue.empty():
            _, worker_result = eval_queue.get()
            veh_avgs = [worker_result[i]['veh_avg_waiting_time'] for i in range(n_iterations)]
            ped_avgs = [worker_result[i]['ped_avg_waiting_time'] for i in range(n_iterations)]
            results[scale] = {
                'veh_avg_mean': round(float(np.mean(veh_avgs)), 2),
                'veh_avg_std': round(float(np.std(veh_avgs)), 2),
                'ped_avg_mean': round(float(np.mean(ped_avgs)), 2),
                'ped_avg_std': round(float(np.std(ped_avgs)), 2),
            }
        else:
            results[scale] = {'veh_avg_mean': 0, 'veh_avg_std': 0,
                              'ped_avg_mean': 0, 'ped_avg_std': 0}
            print(f"  {scale}x: TIMEOUT/ERROR")
            continue

        print(f"  {scale}x: veh={results[scale]['veh_avg_mean']:.2f}±{results[scale]['veh_avg_std']:.2f}  "
              f"ped={results[scale]['ped_avg_mean']:.2f}±{results[scale]['ped_avg_std']:.2f}")

    out_path = os.path.join(run_dir, 'rl_control_baseline_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Print summary comparison
    print("\n" + "="*70)
    print("RL Control on Real-World Layout — Wait Times (seconds)")
    print("="*70)
    print(f"{'Scale':<10} {'Veh Wait':>15} {'Ped Wait':>15}")
    print("-"*40)
    for s in all_demand_scales:
        r = results[s]
        print(f"{s}x{'':<7} {r['veh_avg_mean']:>7.2f}±{r['veh_avg_std']:<5.2f}  "
              f"{r['ped_avg_mean']:>7.2f}±{r['ped_avg_std']:<5.2f}")


if __name__ == '__main__':
    main()
