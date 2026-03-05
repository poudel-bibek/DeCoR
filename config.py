def get_config():
    config = {
        # Simulation
        "sweep": False,  # Use wandb sweeps for hyperparameter tuning
        "evaluate": True, 
        "gui": True,  # Use SUMO GUI (default: False)
         
        "vehicle_input_trips": "./simulation/original_vehtrips.xml",  # Original Input trips file
        "vehicle_output_trips": "./simulation/scaled_trips/scaled_vehtrips.xml",  # Output trips file
        "pedestrian_input_trips": "./simulation/original_pedtrips.xml",  # Original Input pedestrian trips file
        "pedestrian_output_trips": "./simulation/scaled_trips/scaled_pedtrips.xml",  # Output pedestrian trips file
        "original_net_file": "./simulation/Craver_traffic_lights_wide.net.xml",  # Original net file
        "total_sweep_trials": 128,  # Total number of trials for the wandb sweep

        # Demand scaling
        "manual_demand_veh": None,  # Manually scale vehicle demand before starting the simulation (veh/hr)
        "manual_demand_ped": None,  # Manually scale pedestrian demand before starting the simulation (ped/hr)
        "demand_scale_min": 1.0,  # Minimum demand scaling factor for automatic scaling
        "demand_scale_max": 2.25,  # Maximum demand scaling factor for automatic scaling

        # PPO (general params)
        "seed": None,  # Random seed (default: None)
        "gpu": True,  # Use GPU if available (default: use CPU)
        "total_timesteps": 15000000,  # Total number of timesteps the simulation will run
        "eval_freq": 1,  # Evaluate both higher and lower-level policies and their normalizers after every n updates of the higher policy (0 to disable). 
        # Also decides how often to evaluate

        # PPO Higher level agent params
        "higher_anneal_lr": True,  # Anneal learning rate
        "higher_gae_lambda": 0.97,  # GAE lambda for higher-level agent
        "higher_max_grad_norm": 0.75,  # Maximum gradient norm for gradient clipping
        "higher_vf_clip_param": 0.5,  # Value function clipping parameter
        "higher_update_freq": 16,  # Number of action timesteps between each policy update. A low value incurs high variance for design agent.
        "higher_lr": 0.0001,  # Learning rate for higher-level agent
        "higher_gamma": 0.99,  # Discount factor for higher-level agent
        "higher_K_epochs": 4,  # Number of epochs to update policy for higher-level agent
        "higher_eps_clip": 0.2,  # Clip parameter for PPO for higher-level agent
        "higher_batch_size": 2,  # Batch size for higher-level agent
        "higher_dropout_rate": 0.25,  # Dropout rate for GATv2
        "higher_model_size": "medium",  # Model size for GATv2: 'small' or 'medium'
        "higher_ent_coef": 0.001,  # Entropy coefficient for higher-level agent
        "higher_vf_coef": 0.5,  # Value function coefficient for higher-level agent
        "higher_in_channels": 2,  # Number of input features per node (x and y coordinates)
        "higher_hidden_channels": 64, # Number of hidden channels in between two GATv2 layers
        'higher_out_channels': 64, # Number of channels at the ouput of last GATv2 layer
        "higher_activation": "tanh",  # Policy activation function
        "higher_readout_k": 32, # Number of nodes to keep for each graph

        # Design specific parameters
        "min_thickness": 2.0,  # Minimum thickness for crosswalks
        "max_thickness": 15.0,  # Maximum thickness for crosswalks
        "clamp_min": 0.01,  # Minimum value for x location and thickness for crosswalks # Add small buffer to avoid exact 0.0 or 1.0
        "clamp_max": 0.99,  # Maximum value for x location and thickness for crosswalks
        "max_proposals": 10,  # Maximum number of proposals to consider for higher-level agent
        "save_graph_images": True, # Save graph image every iteration.
        "save_gmm_plots": True, # Save GMM visualization every iteration.
        "num_mixtures": 7,  # Number of mixture components in GMM. Having more is ok (means will collapse into each other) but makes it difficult to learn. Having less will make it less representative.
        'initial_heads': 8, # Number of attention heads in first GATv2 layer
        'second_heads': 1, # Number of attention heads in second GATv2 layer
        'edge_dim': 2, # Number of features per edge 

        # PPO Lower level agent params
        "lower_anneal_lr": False,  # Anneal learning rate
        "lower_gae_lambda": 0.95,  # GAE lambda
        "lower_max_grad_norm": 0.75,  # Maximum gradient norm for gradient clipping
        "lower_vf_clip_param": 0.5,  # Value function clipping parameter
        "lower_update_freq": 1024,  # Number of action timesteps between each policy update
        "lower_lr": 0.001,  # Learning rate
        "lower_gamma": 0.99,  # Discount factor
        "lower_K_epochs": 8,  # Number of epochs to update policy
        "lower_eps_clip": 0.2,  # Clip parameter for PPO
        "lower_ent_coef": 0.01,  # Entropy coefficient
        "lower_vf_coef": 0.5,  # Value function coefficient
        "lower_batch_size": 32,  # Batch size
        "lower_num_processes": 10,  # Number of parallel processes to use (agent has multiple workers)
        "lower_model_size": "medium",  # Model size for CNN: 'small' or 'medium'
        "lower_dropout_rate": 0.25,  # Dropout rate for CNN
        "lower_action_dim": None, # will be set later
        "lower_in_channels": 1, # in_channels for cnn
        "lower_activation": "tanh",  # Policy activation function
        "lower_max_timesteps": 360,  # Maximum number of steps in one episode (make this multiple of 16*10)
        "lower_memory_transfer_freq": 16,  # Frequency of memory transfer from worker to main process 
        "lower_per_timestep_state_dim": 11 + 32 + 8 * 10,  # Number of features per timestep (corresponding to max_proposals = 10), calculation in _get_observation function.
        "lower_step_length": 1.0,  # Real-world time in seconds per simulation timestep (default: 1.0). 
        "lower_action_duration": 10,  # Number of simulation timesteps for each action (default: 10)
        "lower_warmup_steps": [40, 140],  # Number of steps to run before collecting data
        "lower_auto_start": True,  # Automatically start the simulation

        # Evaluation
        "eval_model_path": "./runs/readout_32/May09_11-34-05/saved_policies/policy_at_7603200.pth",  # Path to the saved PPO model for evaluation. replace xyz.
        "eval_lower_timesteps": 450,  # Number of timesteps to each episode. Warmup not counted.
        "eval_lower_workers": 10,  # Parallelizes how many demands can be evaluated at the same time.
        "eval_worker_device": "gpu",  # Policy during eval can be run in GPU 
    }
    return config

def classify_and_return_args(train_config, device):
    """
    Classify and return. 
    Design = higher level agent.
    Control = lower level agent.
    """

    design_args = {
        'save_graph_images': train_config['save_graph_images'],
        'save_gmm_plots': train_config['save_gmm_plots'],
        'save_dir': None, # Will be set later.
        'original_net_file': train_config['original_net_file'],
        'eval_freq': train_config['eval_freq'],
        'max_proposals': train_config['max_proposals'],
        'min_thickness': train_config['min_thickness'],
        'max_thickness': train_config['max_thickness'],
        'clamp_min': train_config['clamp_min'],
        'clamp_max': train_config['clamp_max'],
        'higher_anneal_lr': train_config['higher_anneal_lr'],
        'higher_update_freq': train_config['higher_update_freq'],
    }

    control_args = {
        'vehicle_input_trips': train_config['vehicle_input_trips'],
        'vehicle_output_trips': train_config['vehicle_output_trips'],
        'pedestrian_input_trips': train_config['pedestrian_input_trips'],
        'pedestrian_output_trips': train_config['pedestrian_output_trips'],
        'manual_demand_veh': train_config['manual_demand_veh'],
        'manual_demand_ped': train_config['manual_demand_ped'],
        'step_length': train_config['lower_step_length'],
        'lower_action_duration': train_config['lower_action_duration'],
        'warmup_steps': train_config['lower_warmup_steps'],
        'per_timestep_state_dim': train_config['lower_per_timestep_state_dim'], 
        'gui': train_config['gui'],
        'auto_start': train_config['lower_auto_start'],
        'max_timesteps': train_config['lower_max_timesteps'],
        'demand_scale_min': train_config['demand_scale_min'],
        'demand_scale_max': train_config['demand_scale_max'],
        'memory_transfer_freq': train_config['lower_memory_transfer_freq'],
        'max_proposals': train_config['max_proposals'],
        'total_action_timesteps_per_episode': None,
        'lower_num_processes': train_config['lower_num_processes'],
        'lower_anneal_lr': train_config['lower_anneal_lr'],
        'lower_update_freq': train_config['lower_update_freq'],
    }

    higher_model_kwargs = { 
        'num_mixtures': train_config['num_mixtures'],
        'hidden_channels': train_config['higher_hidden_channels'],
        'out_channels': train_config['higher_out_channels'],
        'initial_heads': train_config['initial_heads'],
        'second_heads': train_config['second_heads'],
        'edge_dim': train_config['edge_dim'],
        'activation': train_config['higher_activation'],
        'model_size': train_config['higher_model_size'],
        'dropout_rate': train_config['higher_dropout_rate'],
        'run_dir': None, # Will be set later.
        'readout_k': train_config['higher_readout_k'],
    }

    lower_model_kwargs = { 
        'action_duration': train_config['lower_action_duration'],
        'model_size': train_config['lower_model_size'],
        'dropout_rate': train_config['lower_dropout_rate'],
        'per_timestep_state_dim': train_config['lower_per_timestep_state_dim'],
        'activation': train_config['lower_activation'],
    }

    higher_ppo_args = {
        'model_dim': train_config['higher_in_channels'],
        'action_dim': train_config['max_proposals'],  # Action dimension
        'device': str(device),
        'lr': train_config['higher_lr'],
        'gamma': train_config['higher_gamma'],
        'K_epochs': train_config['higher_K_epochs'],
        'eps_clip': train_config['higher_eps_clip'],
        'ent_coef': train_config['higher_ent_coef'],
        'vf_coef': train_config['higher_vf_coef'],
        'batch_size': train_config['higher_batch_size'],
        'gae_lambda': train_config['higher_gae_lambda'],
        'max_grad_norm': train_config['higher_max_grad_norm'],
        'vf_clip_param': train_config['higher_vf_clip_param'],
        'agent_type': "higher",
        'model_kwargs': higher_model_kwargs
    }

    lower_ppo_args = {
        'model_dim': train_config['lower_in_channels'], 
        'action_dim': train_config['max_proposals'] + 4, # 4 for intersection action and max_proposals for midblock actions.
        'device': str(device),
        'lr': train_config['lower_lr'],
        'gamma': train_config['lower_gamma'],
        'K_epochs': train_config['lower_K_epochs'],
        'eps_clip': train_config['lower_eps_clip'],
        'ent_coef': train_config['lower_ent_coef'],
        'vf_coef': train_config['lower_vf_coef'],
        'batch_size': train_config['lower_batch_size'],
        'gae_lambda': train_config['lower_gae_lambda'],
        'max_grad_norm': train_config['lower_max_grad_norm'],
        'vf_clip_param': train_config['lower_vf_clip_param'],
        'agent_type': "lower",
        'model_kwargs': lower_model_kwargs
    }

    if train_config['evaluate']:
        # during evaluation
        eval_n_iterations = 10 # For the sliced window demand approach, we need more iterations.
        in_range_demand_scales = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25] 
        out_of_range_demand_scales = [0.5, 0.75, 2.5, 2.75]
    else: 
        # during training
        eval_n_iterations = 10
        in_range_demand_scales = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25] # The demand scales that are used for training.
        out_of_range_demand_scales = [0.5, 0.75, 2.5, 2.75] # The demand scales that are used ONLY for evaluation.
    
    eval_args = {
        'lower_state_dim': None,
        'eval_model_path': train_config['eval_model_path'],
        'eval_save_dir': None, # Will be set later.
        'eval_lower_timesteps': train_config['eval_lower_timesteps'],
        'eval_lower_workers': train_config['eval_lower_workers'],
        'eval_worker_device': train_config['eval_worker_device'],
        'eval_n_iterations': eval_n_iterations,
        'in_range_demand_scales': in_range_demand_scales,
        'out_of_range_demand_scales': out_of_range_demand_scales,
    }

    return design_args, control_args, higher_ppo_args, lower_ppo_args, eval_args
