import wandb
import subprocess
import gymnasium as gym
import networkx as nx
import numpy as np
import torch
import torch.multiprocessing as mp
import xml.etree.ElementTree as ET
from gymnasium import spaces
from torch_geometric.data import Data
from ppo.ppo import PPO
from ppo.ppo_utils import Memory, WelfordNormalizer
from .env_utils import *
from .sim_setup import CONTROLLED_CROSSWALKS_DICT, return_horizontal_nodes
from .worker import parallel_train_worker


class DesignEnv(gym.Env):
    """
    Higher level agent.
    - Modifies the net file based on design action.
    - No need to connect to or close this environment. 
    - Internally calls the parallel workers of the control environment.

    Improvements:
    - During initialization: 
        - Extract the pedestrian walkway graph in networkx and other plain XML components (node, edge, connection, type, tllogic) from the original net file.
        - Extract the normalizers from the original net file.
        - Create a base canvas (networkx and plain XML) by removing fringe, isolated nodes, and existing crosswalks from the networkx graph.
        - After every update to the networkx graph, the plain XML components are updated.
        
    - During reset: 
        - Every iteration, crosswalks are added to the base canvas networkx graph.
        - During reset, add the initial crosswalks (present in the real-world network) to the base canvas.
        - Whenever there is a requirement to return the state (e.g., reset, step), the networkx graph is converted to torch geometric and then state returned.
        
    - During step:
        - Apply action:
            - Denormalize proposed locations (thickness does not need denormalization) and add to networkx graph.
            - Use the same mechanism as reset to update the plain XML components.
    - Note: 
        - XML component files contain things other than the pedestrian network as well (e.g., vehicle lanes, junctions, etc.)
        - Original means unmodified SUMO net file.
        - Base means the networkx graph and plain XML components at the start of the training (before any iterations).
        - Iterative means the networkx graph and plain XML components after every iteration.
    """

    def __init__(self, design_args, control_args, lower_ppo_args, run_dir=""):
        super().__init__()
        self.design_args = design_args
        self.control_args = control_args
        self.lower_ppo_args = lower_ppo_args
        self.max_proposals = self.design_args['max_proposals']
        self.run_dir = run_dir
        self.component_dir = f"{run_dir}/components"
        self.network_dir = f"{run_dir}/network_iterations"
        os.makedirs(self.component_dir, exist_ok=True)
        os.makedirs(self.network_dir, exist_ok=True)

        # Generate the 5 different component XML files (node, edge, connection, type, tllogic) from the net file.
        self._create_component_xml_files(self.design_args['original_net_file'])

        # Extract networkx graph from the component files. (Also update locations of nodes in existing_crosswalks)
        pedestrian_networkx_graph  = self._extract_networkx_graph() 
        self.horizontal_nodes_top_ped, self.horizontal_nodes_bottom_ped = return_horizontal_nodes()

        # Initialize normalizers
        self._initialize_normalizers(pedestrian_networkx_graph)
        self.min_thickness = self.design_args['min_thickness']
        self.max_thickness = self.design_args['max_thickness']

        self.existing_crosswalks = self._get_existing_crosswalks(pedestrian_networkx_graph) # Make use of the networkx graph to add locations
        
        # Cleanup the pedestrian walkway graph (i.e, remove isolated, fringe, existing crosswalks) to create a base canvas
        self.base_networkx_graph = self._cleanup_graph(pedestrian_networkx_graph, self.existing_crosswalks)
        self.horizontal_edges_veh_original_data = self._get_original_veh_edge_config()
        
        # Relevant to the intersection
        self.extreme_edge_dict = {'leftmost': {'old': "16666012#2", 'new': None},
                             'rightmost': {'old': "16666012#17", 'new': None}}
                             
        self._update_xml_files(self.base_networkx_graph, 'base') # Create base XML files from latest networkx graph

        if self.design_args['save_graph_images']:
            os.makedirs(os.path.join(self.run_dir, 'graph_iterations'), exist_ok=True)
            save_graph_visualization(pedestrian_networkx_graph, 'original', self.run_dir)
            save_graph_visualization(self.base_networkx_graph, 'base', self.run_dir)
            save_better_graph_visualization(pedestrian_networkx_graph, 'original', self.run_dir)
            save_better_graph_visualization(self.base_networkx_graph, 'base', self.run_dir)

        # Lower level agent
        self.lower_ppo = PPO(**self.lower_ppo_args)

        self.global_step = 0
        self.action_timesteps = 0 # keep track of how many times action has been taken by all lower level workers
        self.lower_update_count = 0
        self.control_head_decisions = np.zeros(self.max_proposals, dtype=np.int64)
        self.control_head_updates = np.zeros(self.max_proposals, dtype=np.int64)
        self.total_updates_lower = None
        self.model_init_params_worker = {'model_dim': self.lower_ppo_args['model_dim'],
                                     'action_dim': self.lower_ppo_args['action_dim'],
                                     'kwargs': self.lower_ppo_args['model_kwargs']}
        self.info = {
            'lower_avg_reward_norm': float('-inf'),
            'lower_avg_reward_unnorm': float('-inf'),
            'lower_update_count': 0,
            'lower_policy_loss': float('inf'),
            'lower_value_loss': float('inf'),
            'lower_entropy_loss': float('inf'),
            'lower_total_loss': float('inf'),
            'lower_current_lr': 0.0,
            'lower_approx_kl': 0.0,
            'lower_exact_kl': 0.0,
            'lower_clip_fraction': 0.0,
            'lower_max_abs_logratio': 0.0,
            'lower_preupdate_max_abs_logratio': 0.0,
        }
        self.current_net_file_path = None
        self.current_network_iteration = None
        self.lower_memories = Memory()

        self.lower_state_normalizer = None
        self.lower_reward_normalizer = WelfordNormalizer(1)
        self.higher_reward_normalizer = WelfordNormalizer(1) # No state normalizer for design agent.

    @property
    def action_space(self):
        """
        Both location and thickness are generated in the range [0, 1], then clamped to [0.01, 0.99], then denormalized.
        """
        return spaces.Dict({
            'num_proposals': spaces.Discrete(self.max_proposals),  # 0 to max_proposals
            'proposals': spaces.Box(
                low=np.array([[0, 0]] * self.max_proposals),
                high=np.array([[1, 1]] * self.max_proposals),
                dtype=np.float32
            )
        })

    @property
    def observation_space(self):
        """
        Returns an arbitrary high-dimensional shape for the observation space.
        Note: This is an arbitrary shape and doesn't reflect the actual dimensions of the graph.
        The GATv2 model can handle variable-sized inputs, so this is mainly for compatibility.
        """
        return spaces.Box(low=0, high=1, shape=(1000, 3), dtype=np.float32)

    def _get_existing_crosswalks(self, networkx_graph):
        """
        Extract the crosswalks present initially in the XML. (exclude 0, 1, 2, 3, 10)
        Add the node locations to the existing crosswalk from the networkx graph.
        """
        excluded_ids = [0, 1, 2, 10]
        existing_crosswalks = {}
        
        for key, data in CONTROLLED_CROSSWALKS_DICT.items():
            for crosswalk_id in data['ids']:
                if key not in excluded_ids:
                    existing_crosswalks[crosswalk_id] = {
                        'pos': [], # will be added later
                        'crossing_nodes': data['crossing_nodes']
                    }

        for crosswalk_id, crosswalk_data in existing_crosswalks.items():
            for node in crosswalk_data['crossing_nodes']:
                if node in networkx_graph.nodes():
                    crosswalk_data['pos'].append(networkx_graph.nodes[node]['pos'])
        # print(f"Existing crosswalks: {existing_crosswalks}")
        return existing_crosswalks

    def _extract_networkx_graph(self,):
        """
        Extract the pedestrian walkway graph in networkx from the component files.

        To create a pedestrian network: 
        - If a node has an edge with type attribute to 'highway.footway' or 'highway.steps' and allow attribute to include 'pedestrian', keep the node and the edge.
        """
        G = nx.Graph() # undirected graph

        # Parse node file
        node_tree = ET.parse(f'{self.component_dir}/original.nod.xml')
        node_root = node_tree.getroot()

        # Add all nodes first (we'll remove non-pedestrian nodes later)
        for node in node_root.findall('node'):
            node_id = node.get('id')
            x = float(node.get('x'))
            y = float(node.get('y'))
            G.add_node(node_id, pos=(x, y), type='regular')

        # Parse edge file
        edge_tree = ET.parse(f'{self.component_dir}/original.edg.xml')
        edge_root = edge_tree.getroot()
        
        # Keep track of nodes that are part of pedestrian paths
        pedestrian_nodes = set()
        
        # Add edges that are pedestrian walkways
        for edge in edge_root.findall('edge'):
            edge_type = edge.get('type')
            allow = edge.get('allow', '')
            
            # Check if edge is a pedestrian walkway
            if edge_type in ['highway.footway', 'highway.steps'] and 'pedestrian' in allow:
                from_node = edge.get('from')
                to_node = edge.get('to')
                
                if from_node is not None and to_node is not None:
                    # Add edge with its attributes
                    width = float(edge.get('width', 2.0)) # default width is 2.0
                    G.add_edge(from_node, to_node, id=edge.get('id'), width=width)
                    
                    # Mark these nodes as part of pedestrian network
                    pedestrian_nodes.add(from_node)
                    pedestrian_nodes.add(to_node)
        
        # Remove nodes that aren't part of any pedestrian path
        non_pedestrian_nodes = set(G.nodes()) - pedestrian_nodes
        G.remove_nodes_from(non_pedestrian_nodes)
        return G

    def _create_component_xml_files(self, sumo_net_file):
        """
        Creates the base SUMO files (5 files).
        """
        # Run netconvert with output files in the specified directory
        command = f"netconvert --sumo-net-file {sumo_net_file} --plain-output-prefix {self.component_dir}/original --plain-output.lanes true"

        try:
            result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
            if result.stderr:
                print("Warnings/Errors from netconvert:", result.stderr)
        except subprocess.CalledProcessError as e:
            print(f"Error running netconvert: {e}")
            print("Error output:", e.stderr)

    def step(self, 
             padded_proposals, 
             num_proposals, 
             iteration,
             fixed_control=False,
             update_layout=True):
        """
        Every step in the design environment involves:
        - Updating the network xml file based on the design action.
        - A number of parallel workers (that utilize the new network file) to each carry out one episode in the control environment.
        """

        # print(f"\nHigher level action received: {action}\n")
        # Convert tensor action to proposals
        padded_proposals = padded_proposals.cpu().numpy()  # Convert to numpy array if it's not already
        proposals = padded_proposals[0][:num_proposals]  # Only consider the actual proposals
        # print(f"\nProposals: {proposals}")

        # Apply the action to output the latest SUMO network file as well as modify the iterative_torch_graph.
        if update_layout:
            self._apply_action(proposals, iteration)

        done = False
        lower_old_policy = None
        if not fixed_control:
            lower_old_policy = self.lower_ppo.policy_old.to(self.lower_ppo_args['device'])
            lower_old_policy.share_memory() # The same policy is shared among all workers.
            lower_old_policy.eval() # So that dropout, batchnorm, layer norm etc. are not used during inference

        lower_queue = mp.Queue()
        lower_processes = []
        for rank in range(self.control_args['lower_num_processes']):
            worker_seed = self.control_args['global_seed'] + iteration * 1000 + rank
            p = mp.Process(
                target=parallel_train_worker,
                args=(
                    rank,
                    self.run_dir,
                    lower_old_policy,
                    self.control_args,
                    lower_queue,
                    worker_seed,
                    num_proposals,
                    self.max_proposals,
                    self.lower_state_normalizer,
                    self.extreme_edge_dict,
                    self.lower_ppo_args['device'],
                    self.current_network_iteration,
                    self.current_net_file_path,
                    fixed_control,
                    self.signal_slots)
                )
            p.start()
            lower_processes.append(p)
        
        # lower_memories = Memory()
        design_rewards_norm = []
        design_rewards_unnorm = [] # Unnormalized reward for logging.
        lower_rewards_norm = []
        lower_rewards_unnorm = []
        for _ in lower_processes:
            rank, memory, design_reward_unnorm, executed_decisions = lower_queue.get(timeout=120)
            design_rewards_norm.append(self.higher_reward_normalizer.normalize([design_reward_unnorm]).item())
            design_rewards_unnorm.append(design_reward_unnorm)
            self.global_step += executed_decisions * self.control_args['lower_action_duration']
            if memory is None:
                continue
            current_action_timesteps = len(memory.states)
            print(f"Memory from worker {rank} received. Memory size: {current_action_timesteps}\n")
            self.lower_memories.states.extend(torch.from_numpy(np.asarray(memory.states)))
            actions = np.asarray(memory.actions)
            self.lower_memories.actions.extend(torch.from_numpy(actions))
            # Measured decisions containing each head, excluding fixed-control collection.
            self.control_head_decisions += (actions[:, 1:] >= 0).sum(axis=0)
            self.lower_memories.num_proposals.extend(torch.from_numpy(np.asarray(memory.num_proposals)))
            self.lower_memories.values.extend(memory.values)
            self.lower_memories.logprobs.extend(memory.logprobs)
            self.lower_memories.is_terminals.extend(memory.is_terminals)

            lower_rewards_unnorm.extend(memory.rewards)
            current_memory_norm = [self.lower_reward_normalizer.normalize([r]).item() for r in memory.rewards]

            # supply normalized rewards for gradient updates
            lower_rewards_norm.extend(current_memory_norm)
            self.lower_memories.rewards.extend(current_memory_norm)

            self.action_timesteps += current_action_timesteps
            print(f"Action timesteps: {self.action_timesteps}, global step: {self.global_step}")
            del memory #https://pytorch.org/docs/stable/multiprocessing.html

        # TODO: Update the normalizer stats for all 3.


        # Clean up. The join() method ensures that the main program waits for all processes to complete before continuing.
        for p in lower_processes:
            p.join()
        # print(f"All processes joined\n\n")

        # Keep shared policy weights fixed until every worker has finished.
        if not fixed_control and self.action_timesteps >= self.control_args['lower_update_freq']:
            # print(f"Updating Lower PPO with {len(self.lower_memories.actions)} memories")
            self.lower_update_count += 1

            avg_lower_reward_unnorm = sum(lower_rewards_unnorm) / len(lower_rewards_unnorm)
            avg_lower_reward_norm = sum(lower_rewards_norm) / len(lower_rewards_norm)

            # Anneal after every update
            if self.control_args['lower_anneal_lr']:
                current_lr_lower = self.lower_ppo.update_learning_rate(iteration, self.total_updates_lower)

            lower_loss = self.lower_ppo.update(self.lower_memories, num_proposals)
            # Count update batches containing a head, not optimizer steps or nonzero gradients.
            self.control_head_updates += (torch.stack(self.lower_memories.actions)[:, 1:] >= 0).any(dim=0).cpu().numpy()

            # Reset all memories
            del self.lower_memories
            self.lower_memories = Memory()
            self.action_timesteps = 0

            # Record the latest lower-policy update.
            self.info = {
                'lower_avg_reward_norm': avg_lower_reward_norm,
                'lower_avg_reward_unnorm': avg_lower_reward_unnorm,
                'lower_update_count': self.lower_update_count,
                'lower_policy_loss': lower_loss['policy_loss'],
                'lower_value_loss': lower_loss['value_loss'],
                'lower_entropy_loss': lower_loss['entropy_loss'],
                'lower_total_loss': lower_loss['total_loss'],
                'lower_current_lr': current_lr_lower if self.control_args['lower_anneal_lr'] else self.lower_ppo_args['lr'],
                'lower_approx_kl': lower_loss['approx_kl'],
                'lower_exact_kl': lower_loss['exact_kl'],
                'lower_clip_fraction': lower_loss['clip_fraction'],
                'lower_max_abs_logratio': lower_loss['max_abs_logratio'],
                'lower_preupdate_max_abs_logratio': lower_loss['preupdate_max_abs_logratio'],
            }

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        del lower_queue

        # Higher level agent's reward can only be obtained after the lower level workers have finished
        # It is also averaged across the various lower level workers.
        average_design_reward_norm = np.mean(design_rewards_norm)
        average_design_reward_unnorm = np.mean(design_rewards_unnorm)
        # print(f"\nAverage design reward (Normalized): {average_design_reward_norm}\n")
        # print(f"\nAverage design reward (Unnormalized): {average_design_reward_unnorm}\n")
        iterative_torch_graph = self._convert_to_torch_geometric(self.iterative_networkx_graph)
        next_state = Data(x=iterative_torch_graph.x,
                           edge_index=iterative_torch_graph.edge_index,
                           edge_attr=iterative_torch_graph.edge_attr)
                      
        return next_state, average_design_reward_norm, average_design_reward_unnorm, done, self.info

    def _apply_action(self, proposals, iteration, crossing_ids=None, signal_slots=None):
        """
        Every iteration, new proposals are added to networkx graph. Then that is converted to torch geometric (for state) and XML (for SUMO)
        Updates all three graph representations (networkx, torch, XML) based on the action.
        The proposals are expected to be a list of tuples (location, thickness) for each proposed crosswalk.
        Explicit logical IDs and slot assignments are retained across paired layouts.
        Without metadata, this layout gets new local IDs and deterministic x-rank slots.
        
        Process:
        1. Denormalize proposed locations (thickness doesn't need denormalization)
        2. Add proposed crosswalks to networkx graph
        3. Update XML
        """
        crossing_ids = [f"iter{iteration}_{i}" for i in range(len(proposals))] if crossing_ids is None else list(crossing_ids)
        if len(crossing_ids) != len(proposals) or any(not isinstance(tid, str) or not tid for tid in crossing_ids) or len(set(crossing_ids)) != len(crossing_ids):
            raise ValueError("Crossing IDs must contain one unique nonempty string per proposal.")
        signal_ids = [f"{tid}_mid" for tid in crossing_ids]
        if signal_slots is None:
            order = sorted(range(len(proposals)), key=lambda i: (proposals[i][0], signal_ids[i]))
            signal_slots = {signal_ids[i]: slot for slot, i in enumerate(order)}
        if set(signal_slots) != set(signal_ids):
            raise ValueError("Signal slots must cover exactly the generated crossing IDs.")
        slots = list(signal_slots.values())
        if any(not isinstance(slot, (int, np.integer)) or slot < 0 or slot >= self.max_proposals for slot in slots):
            raise ValueError("Signal slots must be integer indices within max_proposals.")
        if len(set(slots)) != len(slots):
            raise ValueError("Signal slots must be unique.")
        self.crossing_ids = crossing_ids
        self.signal_slots = {tid: int(slot) for tid, slot in signal_slots.items()}

        # First make a copy
        self.iterative_networkx_graph = self.base_networkx_graph.copy()
        latest_horizontal_nodes_top_ped = self.horizontal_nodes_top_ped
        latest_horizontal_nodes_bottom_ped = self.horizontal_nodes_bottom_ped

        for i, (location, thickness) in enumerate(proposals):
            location = location.item() 
            thickness = thickness.item()

            # 1. Denormalize the location (x-coordinate) and thickness
            denorm_location = self.normalizer_x['min'] + location * (self.normalizer_x['max'] - self.normalizer_x['min'])
            denorm_thickness = self.min_thickness + thickness * (self.max_thickness - self.min_thickness)

            # 2. Add to base networkx graph
            # Add new nodes in both sides in this intersection of type 'regular'.
            # Connect the new nodes to the existing nodes via edges with the given thickness.

            latest_horizontal_segment = self._get_horizontal_segment_ped(latest_horizontal_nodes_top_ped, latest_horizontal_nodes_bottom_ped, self.iterative_networkx_graph) # Start with set of nodes in base graph

            # print(f"\nLatest horizontal segment (Top): {latest_horizontal_segment['top']}\n")
            # print(f"\nLatest horizontal segment (Bottom): {latest_horizontal_segment['bottom']}\n")
            new_intersects = self._find_intersects_ped(denorm_location, latest_horizontal_segment, self.iterative_networkx_graph)
            # print(f"\nNew intersects: {new_intersects}\n")

            mid_node_details = {'top': {'y_cord': None, 'node_id': None}, 'bottom': {'y_cord': None, 'node_id': None}}
            # Now add an edge from from_node to the pos of the new node. As well as from the new node to to_node.
           
            for side in ['top', 'bottom']:
                 # Remove the old edge first.
                from_node, to_node = new_intersects[side]['edge'][0], new_intersects[side]['edge'][1]
                self.iterative_networkx_graph.remove_edge(from_node, to_node)

                # Add the new edge  
                end_node_pos = new_intersects[side]['intersection_pos']
                end_node_id = f"{self.crossing_ids[i]}_{side}"
                self.iterative_networkx_graph.add_node(end_node_id, pos=end_node_pos, type='regular', width=-1) # type for this is regular (width specified for completeness as -1: Not used)
                self.iterative_networkx_graph.add_edge(from_node, end_node_id, width=2.0) # The width of these edges is default (Not from the proposal)
                self.iterative_networkx_graph.add_edge(end_node_id, to_node, width=2.0)

                # Modify the horizontal segment (add the new node)
                if side == 'top':
                    latest_horizontal_nodes_top_ped.append(end_node_id)
                else:
                    latest_horizontal_nodes_bottom_ped.append(end_node_id)

                mid_node_details[side]['y_cord'] = end_node_pos[1]
                mid_node_details[side]['node_id'] = end_node_id

            # Add the mid node and edges 
            mid_node_id = f"{self.crossing_ids[i]}_mid"
            
            # Obtain the y_coordinate of the middle node. Based on adjacent vehicle edges. Use interpolation to find the y coordinate.
            # To ensure that the y coordinates of the graph and the net file are the same. This has to be done here. 
            # previous method, midpoint
            # mid_node_pos = (denorm_location, (mid_node_details['top']['y_cord'] + mid_node_details['bottom']['y_cord']) / 2)

            # new method, interpolation. Always using the original vehicle edge list (not updated with split of split).
            mid_node_pos = (denorm_location, interpolate_y_coordinate(denorm_location, self.horizontal_edges_veh_original_data))

            self.iterative_networkx_graph.add_node(mid_node_id, pos=mid_node_pos, type='middle', width=denorm_thickness) 
            self.iterative_networkx_graph.add_edge(mid_node_details['top']['node_id'], mid_node_id, width=denorm_thickness) 
            self.iterative_networkx_graph.add_edge(mid_node_id, mid_node_details['bottom']['node_id'], width=denorm_thickness) 

        if self.design_args['save_graph_images']:
            save_graph_visualization(self.iterative_networkx_graph, iteration, self.run_dir)
            save_better_graph_visualization(self.iterative_networkx_graph, iteration, self.run_dir)

        # 3. Update XML
        self._update_xml_files(self.iterative_networkx_graph, iteration)
    
    def _find_segment_intersects_ped(self, segments, x_location):
        """
        Helper function to check intersection. 
        x_location is denormalized.
        """
        
        for start_x, (length, edge) in segments.items():
            end_x = start_x + length
            if start_x <= x_location <= end_x:
                return {
                    'edge': edge,
                    'start_x': start_x,
                    'length_x': length
                }
                    
    def _find_intersects_ped(self, x_location, latest_horizontal_segment, latest_graph):
        """
        Find where a given x-coordinate intersects with the horizontal pedestriansegments.
        Returns the edge IDs and positions where the intersection occurs.
        The graph is always changing as edges are added/removed.
        """
        intersections = {}

        for side in ['top', 'bottom']:
            intersections[side] = {}
            intersect = self._find_segment_intersects_ped(latest_horizontal_segment[side], x_location)

            from_node, to_node = intersect['edge'][0], intersect['edge'][1]
            
            # Extract node positions
            from_x, from_y = latest_graph.nodes[from_node]['pos']
            to_x, to_y = latest_graph.nodes[to_node]['pos']

            # Ensure from_x < to_x for consistency
            if from_x > to_x:
                from_x, to_x = to_x, from_x
                from_y, to_y = to_y, from_y

            # Compute how far along the segment x_location lies as a fraction
            x_diff = (x_location - from_x) / (to_x - from_x) + 1e-6 # Avoid division by zero 
            # Now simply interpolate y
            y_location = from_y + x_diff * (to_y - from_y)

            intersections[side]['edge'] = intersect['edge']
            intersections[side]['intersection_pos'] = (x_location, y_location)

        return intersections

    def _get_horizontal_segment_ped(self, horizontal_nodes_top, horizontal_nodes_bottom, latest_graph, validation=False):
        """
        Get the entire horizontal pedestrian segment of the corridor.
        """

        base_nodes_dict = {node[0]: node[1] for node in latest_graph.nodes(data=True)}
        edges = list(latest_graph.edges(data=True))

        horizontal_segment = {'top': {}, 'bottom': {}}

        # find edge ids horizontal_edges_top, horizontal_edges_bottom in edges
        for edge in edges:
            from_node, to_node = edge[0], edge[1]
            from_node_x, to_node_x = base_nodes_dict[from_node]['pos'][0], base_nodes_dict[to_node]['pos'][0]

            if from_node in horizontal_nodes_top and to_node in horizontal_nodes_top:
                smaller_x, larger_x = min(from_node_x, to_node_x), max(from_node_x, to_node_x)
                horizontal_segment['top'][smaller_x] = [larger_x - smaller_x, edge] #[2]['id']] # starting position, length, edge id

            elif from_node in horizontal_nodes_bottom and to_node in horizontal_nodes_bottom:
                smaller_x, larger_x = min(from_node_x, to_node_x), max(from_node_x, to_node_x)
                horizontal_segment['bottom'][smaller_x] = [larger_x - smaller_x, edge] #[2]['id']] # starting position, length, edge id

        # validation plot (to see if they make continuous horizontal segments)
        # Uncomment when necessary
        # if validation:
        #     _, ax = plt.subplots()
        #     horizontal_segment_top = sorted(list(horizontal_segment['top'].keys()))
        #     horizontal_segment_bottom = sorted(list(horizontal_segment['bottom'].keys()))
        #     for start_pos in horizontal_segment_top:
        #         x_min, x_max = horizontal_segment_top[0], horizontal_segment_top[-1]
        #         length = horizontal_segment['top'][start_pos][0]
        #         ax.plot([start_pos, start_pos + length], [2, 2], 'r-')
        #         ax.plot(start_pos, 2, 'x')

        #         # plot the min and max x-coordinate values
        #         ax.text(x_min, 2, f'{x_min:.2f}', fontsize=12, verticalalignment='bottom')
        #         ax.text(x_max, 2, f'{x_max:.2f}', fontsize=12, verticalalignment='bottom')

        #     for start_pos in horizontal_segment_bottom:
        #         x_min, x_max = horizontal_segment_bottom[0], horizontal_segment_bottom[-1]
        #         length = horizontal_segment['bottom'][start_pos][0]
        #         ax.plot([start_pos, start_pos + length], [8, 8], 'b-')
        #         ax.plot(start_pos, 8, 'x')

        #         # plot the min and max x-coordinate values
        #         ax.text(x_min, 8, f'{x_min:.2f}', fontsize=12, verticalalignment='bottom')
        #         ax.text(x_max, 8, f'{x_max:.2f}', fontsize=12, verticalalignment='bottom')

        #     ax.set_ylim(-1, 11)
        #     ax.set_xlabel('X-coordinate')
        #     plt.savefig('./simulation/horizontal_segments.png')
        #     plt.close()

        return horizontal_segment

    def reset(self, start_from_base=False):
        """
        Reset the environment to its initial state.
        Option to start with the initial original set of crosswalks or start from base (empty canvas).
        """
    
        self.iterative_networkx_graph = self.base_networkx_graph.copy()
        if start_from_base:
            pass # Do nothing
        else: 
            # Add middle nodes and edges in the networkx graph. 
            # This middle node configuration will be slightly different from the middle nodes present in the original. 
            for cid, crosswalk_data in self.existing_crosswalks.items():
                # End nodes are already present in the networkx graph, add a connecting edge between them.
                bottom_pos, top_pos = crosswalk_data['pos'][0], crosswalk_data['pos'][-1]
 
                # create a new middle pos
                middle_x = (bottom_pos[0] + top_pos[0]) / 2

                # hard code
                if cid == ':9727816850_c0':
                    middle_y = 2628.38
                elif cid == ':9740157209_c0':
                    middle_y = 2645.06
                elif cid == ':9740157194_c0':
                    middle_y = 2645.51
                elif cid == ':cluster_9740157181_9740483933_c0':
                    middle_y = 2647.24
                elif cid == ':9740157155_c0':
                    middle_x = 2565.50
                    middle_y = 2647.35
                else: 
                    # # For y, If there are 3 nodes,take the y value from the middle node. They are mostly exactly in the middle.
                    if len(crosswalk_data['pos']) == 3:
                        middle_y = crosswalk_data['pos'][1][1]
                    else: 
                        middle_y = (bottom_pos[1] + top_pos[1]) / 2

                middle_pos = (middle_x, middle_y)

                # sanitize the id 
                cid = cid.replace(":", "") # Do not use a space
                middle_node_id = f"{cid}_mid"
                # Add the new middle node to the networkx graph
                self.iterative_networkx_graph.add_node(middle_node_id, pos=middle_pos, type='middle', width=3.0) # At reset, the default width of 3.0 is used

                # Add the connecting edge between the end nodes
                crossing_nodes = crosswalk_data['crossing_nodes']
                bottom_node, top_node = crossing_nodes[0], crossing_nodes[-1]
                self.iterative_networkx_graph.add_edge(bottom_node, middle_node_id, width=3.0)
                self.iterative_networkx_graph.add_edge(middle_node_id, top_node, width=3.0)

        if self.design_args['save_graph_images']:
            save_graph_visualization(self.iterative_networkx_graph, '0', self.run_dir)
            save_better_graph_visualization(self.iterative_networkx_graph, '0', self.run_dir)

        # Everytime the networkx graph is updated, the XML graph needs to be updated.
        # Make the added nodes/edges a crossing with traffic light in XML.
        self._update_xml_files(self.iterative_networkx_graph, 0)
        
        # Return state
        iterative_torch_graph = self._convert_to_torch_geometric(self.iterative_networkx_graph)
        state = Data(x=iterative_torch_graph.x,
                    edge_index=iterative_torch_graph.edge_index,
                    edge_attr=iterative_torch_graph.edge_attr)
        
        return state
    
    def _cleanup_graph(self, graph, existing_crosswalks):
        """
        This step creates a base graph upon which additional edges are added/ removed during training.
        Multiple things happen (Requires some manual input):

        0. Primary cleanup:
            - In the original graph, at least 3 nodes exist to create a crosswalk.
            - To create a base graph, we need to remove these nodes in the middle. 
            - This clears existing crosswalks in the corridor. 

        1. Remove nodes and edges too far away from the corridor based on y values (Not used)
            - Clear the corridor of any nodes and edges that are irrelevant to the pedestrian walkway.
        
        2. Remove isolated and fringe nodes. (Not used: In the new approach, there are no fringe or isolated nodes.)
            - They could have existed because some vehicle roads allow pedestrians.
        """

        cleanup_graph = graph.copy()
        #print(f"\nBefore cleanup: {len(cleanup_graph.nodes())} nodes, {len(cleanup_graph.edges())} edges\n")

        # 0. primary cleanup 
        #print(f"Existing crosswalks: {existing_crosswalks}")
        middle_nodes = []
        for crosswalk_data in existing_crosswalks.values():
            nodes = crosswalk_data['crossing_nodes'] # There will always be more than 2 nodes.
            middle_nodes.extend(nodes[1:-1]) # Add all nodes except first and last
                
        # print(f"Removing middle nodes: {middle_nodes}")
        cleanup_graph.remove_nodes_from(middle_nodes)

        # 1.
        # remove everything with y-coordinates outside 10% and 90% range
        # y_coords = [data['pos'][1] for _, data in cleanup_graph.nodes(data=True)]
        # min_y, max_y = min(y_coords), max(y_coords)
        
        # y_range = max_y - min_y
        # y_lower = min_y + y_range * 0.1
        # y_upper = min_y + y_range * 0.9
        # nodes_to_remove_2 = [node for node, data in cleanup_graph.nodes(data=True)
        #                   if data['pos'][1] < y_lower or data['pos'][1] > y_upper]
                
        # cleanup_graph.remove_nodes_from(nodes_to_remove_2)
        # cleanup_graph.remove_edges_from(cleanup_graph.edges(nodes_to_remove_2))

        # 2.
        # fringe_nodes = ['9727779406','9740484031','cluster_9740411700_9740411702','9740155241','9740484518', '9740484521']
        # isolated_nodes = list(nx.isolates(cleanup_graph))
        # isolated_and_fringe_nodes = fringe_nodes + isolated_nodes
        # cleanup_graph.remove_nodes_from(isolated_and_fringe_nodes)
        #cleanup_graph.remove_edges_from(cleanup_graph.edges(isolated_and_fringe_nodes))  # TODO: When the isolated nodes are removed, are the edges automatically removed?
        #print(f"\nAfter cleanup: {len(cleanup_graph.nodes())} nodes, {len(cleanup_graph.edges())} edges\n")
        return cleanup_graph
    

    def _convert_to_torch_geometric(self, graph):
        """
        Converts the NetworkX graph to a PyTorch Geometric Data object.
        Normalizes the coordinates to lie between 0 and 1 and scales the width values proportionally.
        """
        # Create a mapping from node IDs to indices
        node_id_to_index = {node_id: idx for idx, node_id in enumerate(graph.nodes())}

        # Extract node features (x, y coordinates)
        node_features = []
        for node_id in graph.nodes():
            data = graph.nodes[node_id]
            node_features.append([data['pos'][0], data['pos'][1]])

        data, range_x, range_y = self._normalize_features(torch.tensor(node_features, dtype=torch.float))
        
        # Extract edge indices and attributes
        edge_index = []
        edge_attr = []
        for source_id, target_id, edge_data in graph.edges(data=True):
            source = node_id_to_index[source_id]
            target = node_id_to_index[target_id]

            # The pedestrian network is undirected.
            # To enable bidirectional message passing, we need to add both directions.
            edge_index.append([source, target])
            edge_index.append([target, source]) 

            # Get source node's x coordinate and add it to edge attribute.
            # # TODO: come back to this. DONE.
            # source_x = (graph.nodes[source_id]['pos'][0] - self.normalizer_x['min']) / (self.normalizer_x['max'] - self.normalizer_x['min'])
            # edge_attr.append([edge_data['width'], source_x])  # Add source x-coordinate alongside width
            # edge_attr.append([edge_data['width'], source_x])  # For the reverse edge as well.

            # Edge length (normalized)
            norm_source_x, norm_source_y = (graph.nodes[source_id]['pos'][0]- self.normalizer_x['min'])/ range_x, (graph.nodes[source_id]['pos'][1]- self.normalizer_y['min'])/ range_y
            norm_target_x, norm_target_y = (graph.nodes[target_id]['pos'][0]- self.normalizer_x['min'])/ range_x, (graph.nodes[target_id]['pos'][1]- self.normalizer_y['min'])/ range_y
            edge_length = np.linalg.norm(np.array([norm_source_x, norm_source_y]) - np.array([norm_target_x, norm_target_y]))
            edge_attr.append([edge_data['width'], edge_length])
            edge_attr.append([edge_data['width'], edge_length])
            
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        # Create PyTorch Geometric Data object
        return Data(x=data, edge_index=edge_index, edge_attr=edge_attr)

    def _normalize_features(self, features):
        """
        Normalizers are gotten once from the original graph.
        Normalize the features (coordinates) to lie between 0 and 1.
        Save the normalizer values for potential later use.
        """
        x_coords = features[:, 0]
        y_coords = features[:, 1]
        range_x = self.normalizer_x['max'] - self.normalizer_x['min']
        range_y = self.normalizer_y['max'] - self.normalizer_y['min']
        normalized_x = (x_coords - self.normalizer_x['min']) / range_x
        normalized_y = (y_coords - self.normalizer_y['min']) / range_y

        return torch.stack([normalized_x, normalized_y], dim=1), range_x, range_y
    
    def _get_original_veh_edge_config(self):
        """
        Get the original vehicle edge config from the original XML component files.
        """

        horizontal_edges_veh= {
        'top': ['-16666012#2', '-16666012#3', '-16666012#4', '-16666012#5', 
                                '-16666012#6', '-16666012#7', '-16666012#9', '-16666012#11', 
                                '-16666012#12', '-16666012#13', '-16666012#14', '-16666012#15', 
                                '-16666012#16', '-16666012#17'],
        'bottom': ['16666012#2', '16666012#3', '16666012#4', '16666012#5',
                                    '16666012#6', '16666012#7', '16666012#9', '16666012#11',
                                    '16666012#12', '16666012#13', '16666012#14', '16666012#15',
                                    '16666012#16', '16666012#17']
                                }
        
        node_file = f'{self.component_dir}/original.nod.xml'
        node_tree = ET.parse(node_file)
        node_root = node_tree.getroot()

        edge_file = f'{self.component_dir}/original.edg.xml'
        edge_tree = ET.parse(edge_file)
        edge_root = edge_tree.getroot()

        horizontal_edges_veh_original_data = {
            'top': {},
            'bottom': {}}

        for direction in ['top', 'bottom']:
            for edge in edge_root.findall('edge'):
                id = edge.get('id')
                if id in horizontal_edges_veh[direction]:
                    from_node = edge.get('from')
                    from_node_data = node_root.find(f'node[@id="{from_node}"]')
                    # Convert coordinates to float
                    from_x = float(from_node_data.get('x'))
                    from_y = float(from_node_data.get('y'))

                    to_node = edge.get('to')
                    to_node_data = node_root.find(f'node[@id="{to_node}"]')
                    # Convert coordinates to float
                    to_x = float(to_node_data.get('x'))
                    to_y = float(to_node_data.get('y'))

                    horizontal_edges_veh_original_data[direction][id] = {
                        'from_x': from_x,
                        'from_y': from_y, 
                        'to_x': to_x,
                        'to_y': to_y
                    }

        return horizontal_edges_veh_original_data
    

    def _update_xml_files(self, networkx_graph, iteration):
        """
        Update the XML component files to reflect the current state of the networkx graph. 
        For base, use the "original" XML component files. For other iterations, use the "base" XML component files as a foundation and add/ remove elements.
        Iterative component files are saved in components directory.
        Iterative net files are saved in network_iterations directory.

        Networkx graph will already have:
          - End nodes with position values that come from the proposal.
          - Middle nodes to create traffic lights. Every proposal will have nodes with id _mid. 
                - The _mid nodes need to be connected to vehicle edges on either side as well. 
                - For every _mid node, a TL logic needs to be added to the traffic_light XML file.

        For iteration = base, the end nodes are already present in the XML component files.
        For other iterations, the end nodes will require reconstruction.

        For the nodes and edges related to pedestrian network:
            Remove them from the XML component files: If they don't exist in networkx graph.
            Add them to the XML component files: If they exist in networkx graph but don't exist in component XML.

        Node attributes: 
            - <node id=" " x=" " y=" " type=" " />
            - For the middle nodes, type will "traffic_light" and an attribute tl =" " with the value same as id.
            - For the end nodes, type will be "dead_end"
        Edge attributes: 
            - From the middle node to end nodes of type "highway.footway"
            - From middle node to vehicle nodes of type "highway.tertiary"
            - Both of these are needed because the traffic light is coordinating vehicles and pedestrians in the crosswalk.
            - <edge id=" " from=" " to=" " priority="1" type="highway.footway" numLanes="1" speed="2.78" shape=" " spreadType="center" width="2.0" allow="pedestrian"> 
            - shape seems difficult to get right.
            - create a nested lane element: <lane index="0" allow="pedestrian" width=" " speed=" ">
            - create a nested param element: <param key="origId" value=" "/>
            - end with </lane></edge>
        """

        # Parse the XML files
        prefix = "original" if iteration == 'base' else "iteration_base" # Every iteration will have the same base XML files.
        
        node_file = f'{self.component_dir}/{prefix}.nod.xml'
        node_tree = ET.parse(node_file)
        node_root = node_tree.getroot()

        edge_file = f'{self.component_dir}/{prefix}.edg.xml'
        edge_tree = ET.parse(edge_file)
        edge_root = edge_tree.getroot()

        connection_file = f'{self.component_dir}/{prefix}.con.xml'
        connection_tree = ET.parse(connection_file)
        connection_root = connection_tree.getroot() 

        traffic_light_file = f'{self.component_dir}/{prefix}.tll.xml'
        traffic_light_tree = ET.parse(traffic_light_file)
        traffic_light_root = traffic_light_tree.getroot()

        type_file = f'{self.component_dir}/{prefix}.typ.xml'
        type_tree = ET.parse(type_file)

        # Find ALL the nodes and edges in the XML component files (nod.xml and edg.xml)
        nodes_in_xml = { n.get('id'): n for n in node_root.findall('node') } # save the node element itself.
        edges_in_xml = { (e.get('from'), e.get('to')): e for e in edge_root.findall('edge') } # save the from, to nodes and edge element.

        # Find PEDESTRIAN nodes and edges in the XML component .edg file. 
        pedestrian_edges_in_xml = {}
        pedestrian_nodes_in_xml = set()
        for (f, t), e in edges_in_xml.items():
            e_type = e.get('type')
            allow = e.get('allow', '')
            if e_type in ['highway.footway', 'highway.steps'] and 'pedestrian' in allow:
                pedestrian_edges_in_xml[(f, t)] = e
                pedestrian_nodes_in_xml.update([f, t]) # From node id alone we cant differentiate between vehicle and pedestrian nodes.

        # Extract pedestrian nodes and edges from networkx_graph
        pedestrian_nodes_in_graph = set(networkx_graph.nodes())
        
        # Remove PEDESTRIAN nodes that are in XML component file but not in networkx graph.
        potential_nodes_to_remove = pedestrian_nodes_in_xml - pedestrian_nodes_in_graph

        # Some edges may still access the nodes that are in potential_nodes_to_remove.
        # Find the edges that still access the nodes that are in potential_nodes_to_remove.
        edges_in_xml_that_access_removal_nodes = {}
        for (f, t) in edges_in_xml:
            if f in potential_nodes_to_remove or t in potential_nodes_to_remove:
                edges_in_xml_that_access_removal_nodes[(f, t)] = edges_in_xml[(f, t)] # These can be vehicle edges as well.

        # In the edges that access nodes in potential_nodes_to_remove, some of the edges are vehicle edges (For e.g., when the old TL was removed).
        vehicle_edges_that_access_removal_nodes = {}
        for (f, t), e in edges_in_xml_that_access_removal_nodes.items():
            e_type = e.get('type')
            disallow = e.get('disallow', '')
            if e_type == 'highway.tertiary' and 'pedestrian' in disallow: # vehicle edge attributes: highway.tertiary and disallowed pedestrian
                vehicle_edges_that_access_removal_nodes[(f, t)] = e
        # print(f"Vehicle edges that access removal nodes: Total: {len(vehicle_edges_that_access_removal_nodes)},\n {vehicle_edges_that_access_removal_nodes}\n")
        
        # Get all nodes that appear in vehicle edges
        nodes_in_vehicle_edges = set()
        for f, t in vehicle_edges_that_access_removal_nodes.keys():
            nodes_in_vehicle_edges.add(f)
            nodes_in_vehicle_edges.add(t)
        # print(f"Potential nodes to be removed: {potential_nodes_to_remove}\n Nodes in vehicle edges: {nodes_in_vehicle_edges}\n")
 
        # The nodes that appear in vehicle edges can be removed because they are not needed for the pedestrian network. Remove them
        pedestrian_nodes_to_remove = potential_nodes_to_remove - nodes_in_vehicle_edges
        # print(f"Actual pedestrian nodes to remove: Total: {len(pedestrian_nodes_to_remove)},\n {pedestrian_nodes_to_remove}\n")

        # Some pedestrian edges (at crossings) link to vehicle edges. Remove the pedestrian edges that are not linked to the vehicle edges. 
        pedestrian_edges_to_remove = {}
        for (f, t), e in edges_in_xml_that_access_removal_nodes.items():
            if (f, t) not in vehicle_edges_that_access_removal_nodes:
                pedestrian_edges_to_remove[(f, t)] = e
        # print(f"Actual pedestrian edges to remove: Total: {len(pedestrian_edges_to_remove)}, \n {pedestrian_edges_to_remove}\n")

        # Removing selected nodes and edges
        for node_id in pedestrian_nodes_to_remove:
            if node_id in nodes_in_xml:
                node_root.remove(nodes_in_xml[node_id]) # remove from nod component file
                del nodes_in_xml[node_id] # remove from dictionary

        for (f, t) in pedestrian_edges_to_remove:
            if (f, t) in edges_in_xml:
                edge_root.remove(edges_in_xml[(f, t)]) # remove from edg component file
                del edges_in_xml[(f, t)] # remove from dictionary

        # Before new nodes are added.
        # All the nodes with tl other than default tl need to have type="dead_end" and tl attribute removed.
        default_tl = ['cluster_172228464_482708521_9687148201_9687148202_#5more'] # By default in base, there will the one TL at the left intersection. present.
        for node in node_root.findall('node'):
            tl_name = node.get('tl')
            if tl_name:
                if tl_name not in default_tl:
                    node.set('type', 'dead_end')
                    del node.attrib['tl']

        # Find the pedestrian nodes to add (present in networkx graph but not in XML component file) i.e., end nodes and middle nodes
        # In iterations other than base i.e., in iteration base, there will be no new nodes to add.
        # For regular nodes: <node id=" " x=" " y=" " />
        # For the nodes with type "middle": also add attributes: type = "traffic_light" and tl = "node_id" 
        node_ids_to_add = pedestrian_nodes_in_graph - set(nodes_in_xml.keys()) 
        middle_nodes_to_add = []
        # print(f"\nNodes to add: {node_ids_to_add}")

        for nid in node_ids_to_add:
            node_data = networkx_graph.nodes[nid]
            x, y = node_data['pos']
            n_type = node_data.get('type', 'regular')
            attribs = {'id': nid, 'x': str(round(x, 2)), 'y': str(round(y, 2))}

            if n_type == 'regular':
                attribs['type'] = 'dead_end'
            elif n_type == 'middle':
                middle_nodes_to_add.append(nid)
                attribs['type'] = 'traffic_light'
                attribs['tl'] = nid

            new_node = ET.Element('node', attribs)
            new_node.tail = "\n\t"
            node_root.append(new_node)
            nodes_in_xml[nid] = new_node

        # Find the edges to add (present in networkx graph but not in XML component file).
        ped_edges_to_add = set(networkx_graph.edges()) - set(edges_in_xml.keys()) # These are all pedestrian edges.
        ped_edges_to_add = list(ped_edges_to_add)
        # print(f"\nPedestrian edges to add: Total: {len(ped_edges_to_add)},\n {ped_edges_to_add}\n")

        # The edge could be from a type = "regular" node to a type = "regular" node or from a type = "regular" node to a type = "middle" node (crossing).
        for (f, t) in ped_edges_to_add:
            # Do Regular to Regular and Regular to Middle need some different treatment?
            edge_data = networkx_graph.get_edge_data(f, t)
            edge_id = edge_data.get('id', f'edge_{f}_{t}') # Get it from the networkx graph.
            width = edge_data.get('width', None) # There should be a width for all edges.
            edge_attribs = {
                'id': edge_id,
                'from': f,
                'to': t,
                'name': 'Iterative addition',
                'priority': '1',
                'type': 'highway.footway',
                'numLanes': '1',
                'speed': '2.78', # default
                'spreadType': 'center',
                'width': str(width),
                'allow': 'pedestrian'
            }

            # positions of f and t nodes
            f_data = networkx_graph.nodes[f]
            t_data = networkx_graph.nodes[t]
            f_x, f_y = round(f_data['pos'][0], 2), round(f_data['pos'][1], 2)
            t_x, t_y = round(t_data['pos'][0], 2), round(t_data['pos'][1], 2)
            shape = f'{f_x},{f_y} {t_x},{t_y}'

            edge_element = ET.Element('edge', edge_attribs)
            edge_element.text = "\n\t\t" 

            lane_element = ET.SubElement(
                edge_element,
                'lane', 
                index='0', 
                allow='pedestrian',
                width=str(width),
                speed='2.78', 
                shape=shape)

            lane_element.text = "\n\t\t\t" 
            param_element = ET.SubElement(lane_element, 'param', key='origId', value=edge_id)
            param_element.tail = "\n\t\t" 
            lane_element.tail = "\n\t"
            edge_element.tail = "\n\t"
            edge_root.append(edge_element)

        # Every middle node (present in middle_nodes_to_add) falls on a certain vehicle edge. Split the vehicle edges into two new edges.
        # The new edge names have left and right attached to the old names (the new edges inherit respective portions of the edge shape and lane shape property of the old edge)
        # This happens iteratively (because multiple middle nodes may fall on the same vehicle edge) and is a bit complex.
        old_veh_edges_to_remove, new_veh_edges_to_add, updated_conn_root, m_node_mapping = get_new_veh_edges_connections(middle_nodes_to_add, 
                                                                                                         networkx_graph, 
                                                                                                         f'{self.component_dir}/original.edg.xml', 
                                                                                                         f'{self.component_dir}/original.nod.xml', 
                                                                                                         connection_root)

        # Add the new edges (each edge has a single nested lane) to the edge file. The width is the default road width.
        for direction in ['top', 'bottom']:
            for edge_id, edge_data in new_veh_edges_to_add[direction].items():
                edge_attribs = {
                    'id': edge_id,
                    'from': edge_data.get('from'),
                    'to': edge_data.get('to'),
                    'name': "Craver Road Iterative Addition",
                    'priority': "10",
                    'type': "highway.tertiary",
                    'numLanes': "1",
                    'speed': "8.94",
                    'disallow': "pedestrian tram rail_urban rail rail_electric rail_fast ship cable_car subway"
                }

                edge_element = ET.Element('edge', edge_attribs)
                edge_element.text = "\n\t\t"

                lane_element = ET.SubElement(edge_element, 
                                             'lane', 
                                             index='0', 
                                             disallow="pedestrian tram rail_urban rail rail_electric rail_fast ship cable_car subway", 
                                             speed="8.94", 
                                             )

                lane_element.text = "\n\t\t\t"
                param_element = ET.SubElement(lane_element, 'param', key='origId', value=edge_id.split('#')[0].replace('-', '')) # remove the negative sign and #
                param_element.tail = "\n\t\t"
                lane_element.tail = "\n\t"
                edge_element.tail = "\n\t"

                edge_root.append(edge_element)
        
        # For TL logics,
        # TL logics should come before the connections. (https://github.com/eclipse-sumo/sumo/issues/6160)
        # In order to do this, we first remove all existing TL logics except the default one.
        # We collect the connections associated with default TL and remove all connections.
        # TL 1. Remove all TLs and except the default one.
        tls_to_remove = []
        for tl in traffic_light_root.findall('tlLogic'):
            if tl.get('id') not in default_tl:
                tls_to_remove.append(tl)
        for tl in tls_to_remove:
            traffic_light_root.remove(tl)

        # TL 2. Remove all connections and store the default ones.
        tl_connections_to_add = [] # collect the connection elements.
        connections_to_remove = [] # Connections except the default TL should be removed from the connections file as well.
        for conn in traffic_light_root.findall('connection'):
            traffic_light_root.remove(conn) # remove from the TLL file whether its default or not. We will add it back later.
            if conn.get('tl') in default_tl:
                tl_connections_to_add.append(conn)
            else:
                connections_to_remove.append(conn) # remove later from the connections file.

        # The TLL file connections contains connections between edges that are left and right of every midde node.
        # Due to split of split, the names of these edges may not be symmetrical (i.e., just replace left with right and vice versa wont work).
        # Use linkIndex 0 for connecting -ve direction and linkIndex 1 for connecting +ve direction.
        for direction in ['top', 'bottom']:
            for tl_id, mapping_data in m_node_mapping.items(): # m_node is the tl_id
                linkindex = 0 if direction == 'top' else 1 # Top is -ve direction and bottom is +ve direction.
                
                # These connections should be present in both the TLL and connections files (using left as from and right as to).
                # TL 3. Add the new connections.
                tl_conn_attribs = {'from': mapping_data[direction]['from'], 'to': mapping_data[direction]['to'], 'fromLane': "0", 'toLane': "0", 'tl': tl_id, 'linkIndex': str(linkindex)} # Since inside the corridor, there is only one lane.
                tl_conn_element = ET.Element('connection', tl_conn_attribs)
                tl_connections_to_add.append(tl_conn_element)

                conn_attribs = {'from': mapping_data[direction]['from'], 'to': mapping_data[direction]['to'], 'fromLane': "0", 'toLane': "0"} # Since inside the corridor, there is only one lane.
                conn_element = ET.Element('connection', conn_attribs)
                conn_element.text = None  # Ensure there's no text content
                conn_element.tail = "\n\t\t"
                updated_conn_root.append(conn_element)

        # For the crossing tags in the Conn file ( which also dont need to be changed iteratively). # The width here needs to come from the model. 
        # They are already updated while obtaining the new edges. Nothing to do here.
        # Whereas for the crossing tags,
        # First remove all except the default ones. Then add the new ones here by making use of new_veh_edges_to_add.
        default_crossings = default_tl + ['cluster_172228408_9739966907_9739966910', '9687187500', '9687187501'] # associated with ids 0 and 10.
        for crossing in updated_conn_root.findall('crossing'):
            if crossing.get('node') not in default_crossings:
                updated_conn_root.remove(crossing)
        
        # Then deal with the existing old crossings that refer to the old edges which have been split. 
        # Can be done manually.. as in -> if the leftmost edge has been split then the intersection should now refer to the new edge.
        min_x, max_x = float('inf'), float('-inf')
        leftmost_new, rightmost_new = '', ''
        for edge_id, edge_data in new_veh_edges_to_add['top'].items(): # One of the counterparts (among -ve, +ve) is enough.
            # Also bottom has reverse direction so top is enough.
            min_x_among_nodes = min(edge_data.get('from_x'), edge_data.get('to_x'))
            if min_x_among_nodes < min_x:
                min_x = min_x_among_nodes
                leftmost_new = f'16666012#{edge_id.split("#")[1]}'
            if min_x_among_nodes > max_x:
                max_x = min_x_among_nodes
                rightmost_new = f'16666012#{edge_id.split("#")[1]}'

        # One of the counterparts (among -ve, +ve) is enough.
        self.extreme_edge_dict['leftmost']['new'] = leftmost_new
        self.extreme_edge_dict['rightmost']['new'] = rightmost_new
        
        # Updates to connections and crossings in connections file.
        for direction, direction_data in self.extreme_edge_dict.items():
            old_edge = direction_data['old']
            if old_edge in old_veh_edges_to_remove:
                new_edge = direction_data['new']
                # print(f"\n\nold_edge: {old_edge}, new_edge: {new_edge}\n\n")
                
                for crossing in updated_conn_root.findall('crossing'):
                    if crossing.get('edges') == f'{old_edge} -{old_edge}':
                        # First, a connection between the two new edges should be added.
                        connection_element = ET.Element('connection', {'from': new_edge, 'to': f'-{new_edge}', 'fromLane': '0', 'toLane': '0'})
                        connection_element.text = None  # Ensure there's no text content
                        connection_element.tail = "\n\t\t"
                        updated_conn_root.append(connection_element)
                        # Then, it can be updated in crossing.
                        crossing.set('edges', f'{new_edge} -{new_edge}')

                    elif crossing.get('edges') == f'-{old_edge} {old_edge}':
                        # First, a connection between the two new edges should be added.
                        connection_element = ET.Element('connection', {'from': f'-{new_edge}', 'to': new_edge, 'fromLane': '0', 'toLane': '0'})
                        connection_element.text = None  # Ensure there's no text content
                        connection_element.tail = "\n\t\t"
                        updated_conn_root.append(connection_element)

                        # Then, it can be updated in crossing.
                        crossing.set('edges', f'-{new_edge} {new_edge}')
        # print(f"m_node_mapping: {m_node_mapping}\n")
        # Add new connections (between top and bottom edges) and crossings (making use of new_veh_edges_to_add).
        # All tags that refer to the old edges should now refer to the new edges (if the refering edges fall to the left, they will refer to the new left edge and vice versa) 
        # They have the edges attribute (which are edges to the right) and outlineShape attribute (the shape of the crossing): 
        # outlineShape seems hard to specify, lets not specify and see what it does. They mention it as optional here: https://github.com/eclipse-sumo/sumo/issues/11668
        for m_node, mapping_data in m_node_mapping.items():
            # Retrieve the 'from' edge for the top side
            e1 = mapping_data['top']['from']
            # if e1 is None:
            #     print(f"[ERROR crossing] iteration={iteration}, m_node={m_node} - 'from' edge is None; mapping_data={mapping_data}")
            #     # Additional debug for None 'from' edge
            #     pos = networkx_graph.nodes[m_node].get('pos')
            #     print(f"[DEBUG crossing] m_node '{m_node}' position: {pos}")
            #     bottom_map = mapping_data['bottom']
            #     print(f"[DEBUG crossing] mapping_data['bottom']: {bottom_map}")
            #     # Debug horizontal segments on current graph
            #     horiz_seg = self._get_horizontal_segment_ped(
            #         self.horizontal_nodes_top_ped, 
            #         self.horizontal_nodes_bottom_ped, 
            #         networkx_graph
            #     )
            #     print(f"[DEBUG crossing] horizontal_segment top: {horiz_seg['top']}")
            #     print(f"[DEBUG crossing] horizontal_segment bottom: {horiz_seg['bottom']}")
            #     print(f"[DEBUG crossing] horizontal_nodes_top: {self.horizontal_nodes_top_ped}")
            #     print(f"[DEBUG crossing] horizontal_nodes_bottom: {self.horizontal_nodes_bottom_ped}")
            #     continue
            
            if '-' in e1:
                e2 = e1.replace('-', '')
            else:
                e2 = '-' + e1
            width = networkx_graph.nodes[m_node].get('width')
            crossing_attribs = {'node': m_node, 'edges': e1 + ' ' + e2, 'priority': '1', 'width': str(width), 'linkIndex': '2'} # Width/ Thickness needs to come from the model.
            crossing_element = ET.Element('crossing', crossing_attribs)
            crossing_element.text = None  # Ensure there's no text content
            crossing_element.tail = "\n\t\t"
            updated_conn_root.append(crossing_element)

        # Delete the old edges from the edg file i.e., just remove the tags with old edge ids.
        for edge in edge_root.findall('edge'):
            if edge.get('id') in old_veh_edges_to_remove:
                edge_root.remove(edge)

        # TL 4. Add the new TL logics.
        for nid in middle_nodes_to_add:
            tlLogic_element = ET.Element('tlLogic', id=nid, type='static', programID='0', offset='0')
            tlLogic_element.text = "\n\t\t" # Inside <tlLogic>: phases start at two tabs

            # Create phases with proper indentation
            phase1 = ET.SubElement(tlLogic_element, 'phase', duration='40', state='GGr')
            phase1.tail = "\n\t\t"
            phase2 = ET.SubElement(tlLogic_element, 'phase', duration='4', state='yyr') 
            phase2.tail = "\n\t\t"
            phase3 = ET.SubElement(tlLogic_element, 'phase', duration='16', state='rrG')
            phase3.tail = "\n\t\t"
            phase4 = ET.SubElement(tlLogic_element, 'phase', duration='2', state='rrr')
            phase4.tail = "\n\t"

            tlLogic_element.tail = "\n\t"
            traffic_light_root.append(tlLogic_element)
        
        # TL 5. Add all the new connections.
        for conn in tl_connections_to_add:
            conn.text = None  
            conn.tail = "\n\t"
            traffic_light_root.append(conn)

        # TL 6. The default crossings in TL (that were kept above) may still refer to the old edges.
        # In addition, there may also be a connection of the -ve and +ve sides of the old edges.
        for direction, direction_data in self.extreme_edge_dict.items():
            old_edge = direction_data['old']
            if old_edge in old_veh_edges_to_remove:
                new_edge = direction_data['new']
                for conn in traffic_light_root.findall('connection'):
                    if conn.get('from') == old_edge: # positive
                        conn.set('from', new_edge)
                    if conn.get('from') == f"-{old_edge}": # negative
                        conn.set('from', f"-{new_edge}") 
                    if conn.get('to') == old_edge: # positive
                        conn.set('to', new_edge)
                    if conn.get('to') == f"-{old_edge}": # negative
                        conn.set('to', f"-{new_edge}")

        # Respective changes to the connections file.
        # All the connections present in the TLL file should also be present in the connections file. But the connection file will have more of them.
        # In iteration base, there will be a bunch of connections to remove from original file (remove connections with the same from and to edges).

        # Look at the same from and to edges in the connections file and remove them.
        connections_to_remove_list = [(conn.get('from'), conn.get('to')) for conn in connections_to_remove]
        to_remove = []
        for conn in connection_root.findall('connection'):
            from_edge = conn.get('from')
            to_edge = conn.get('to')
            if (from_edge, to_edge) in connections_to_remove_list:
                to_remove.append(conn)
        for conn in to_remove:
            connection_root.remove(conn)

        # Additional stuff related to edge removals.
        # If the edge (pedestrian and vehicle) is removed, then the connections to and from that edge should also be removed.
        pedestrian_edges_to_remove_connections = []
        for (f,t), edge in pedestrian_edges_to_remove.items():
            pedestrian_edges_to_remove_connections.append(edge.get('id'))

        for conn in connection_root.findall('connection'):
            if conn.get('from') in pedestrian_edges_to_remove_connections or conn.get('to') in pedestrian_edges_to_remove_connections:
                connection_root.remove(conn)
        
        iteration_prefix = f'{self.component_dir}/iteration_{iteration}'
        node_tree.write(f'{iteration_prefix}.nod.xml', encoding='utf-8', xml_declaration=True)
        edge_tree.write(f'{iteration_prefix}.edg.xml', encoding='utf-8', xml_declaration=True)
        connection_tree.write(f'{iteration_prefix}.con.xml', encoding='utf-8', xml_declaration=True)
        type_tree.write(f'{iteration_prefix}.typ.xml', encoding='utf-8', xml_declaration=True)
        traffic_light_tree.write(f'{iteration_prefix}.tll.xml', encoding='utf-8', xml_declaration=True)

        # Generate the final net file using netconvert
        self.current_net_file_path = f'{self.network_dir}/network_iteration_{iteration}.net.xml'
        self.current_network_iteration = iteration
        netconvert_log_file = f'{self.run_dir}/netconvert_log.txt'
        command = (
            f"netconvert "
            f"--node-files={iteration_prefix}.nod.xml "
            f"--edge-files={iteration_prefix}.edg.xml "
            f"--connection-files={iteration_prefix}.con.xml "
            f"--type-files={iteration_prefix}.typ.xml "
            f"--tllogic-files={iteration_prefix}.tll.xml " # using TLL file (against advice from Jakob)
            f"--output-file={self.current_net_file_path} "
            f"--log={netconvert_log_file}"
        )

        max_attempts = 3
        attempt = 0
        while attempt < max_attempts:
            try:
                result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
                if result.stderr:
                    print(f"Warnings/Errors from netconvert: {result.stderr}")
                break
            except subprocess.CalledProcessError as e:
                print(f"Error running netconvert (attempt {attempt + 1}/{max_attempts}): {e}")
                print("Error output:", e.stderr)
                attempt += 1
                if attempt == max_attempts:
                    print("Failed all attempts to run netconvert")
                    raise

    def _initialize_normalizers(self, pedestrian_networkx_graph):
        """
        Initialize normalizers based on the graph coordinates
        """
        # Get all horizontal segments
        horizontal_segment = self._get_horizontal_segment_ped(
            self.horizontal_nodes_top_ped, 
            self.horizontal_nodes_bottom_ped, 
            pedestrian_networkx_graph)
        
        offset = 2 # Add or Subtract a small buffer.
        # Find the leftmost and rightmost valid x coordinates from both top and bottom segments
        top_x_coords = list(horizontal_segment['top'].keys())
        bottom_x_coords = list(horizontal_segment['bottom'].keys())
        
        # Get the leftmost valid x (start of first segment)
        min_x = max(min(top_x_coords), min(bottom_x_coords)) + offset
        
        # Get the rightmost valid x (end of last segment)
        max_top_x = max(x + length for x, (length, _) in horizontal_segment['top'].items())
        max_bottom_x = max(x + length for x, (length, _) in horizontal_segment['bottom'].items())
        max_x = min(max_top_x , max_bottom_x) - offset
        
        self.normalizer_x = {'min': float(min_x), 'max': float(max_x)}
        
        # Keep the y coordinate normalization as is
        coords = np.array([data['pos'] for _, data in pedestrian_networkx_graph.nodes(data=True)])
        y_coords = coords[:, 1]
        self.normalizer_y = {'min': float(np.min(y_coords)), 'max': float(np.max(y_coords))}
