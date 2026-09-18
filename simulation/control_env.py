import math
import time 
import traci
import torch
import sumolib
import sumo
import random
import gymnasium as gym
import numpy as np
import pprint
from utils import scale_demand, scale_demand_sliced_window
from .env_utils import *
from .sim_setup import *
from .signal_control import PROTOCOL as SHARED_SIGNAL_PROTOCOL, SignalExecutor

class ControlEnv(gym.Env):
    def __init__(self, control_args, run_dir, worker_id=None, network_iteration=None, current_net_file_path=None):
        """
        Lower level agent.
        Parallelizable environment, includes features:  
        - Scaling the demand.
        - Tracking and occupancy map.
        """
        super().__init__()
        self.worker_id = worker_id
        self.traci_label = f"_{worker_id}" if worker_id is not None else "default"
        self.run_dir = run_dir
        self.network_iteration = network_iteration
        self.vehicle_input_trips = control_args['vehicle_input_trips']
        self.vehicle_output_trips = control_args['vehicle_output_trips']
        self.pedestrian_input_trips = control_args['pedestrian_input_trips']
        self.pedestrian_output_trips = control_args['pedestrian_output_trips']
        self.manual_demand_veh = control_args['manual_demand_veh']
        self.manual_demand_ped = control_args['manual_demand_ped']
        self.demand_scale_min = control_args['demand_scale_min']
        self.demand_scale_max = control_args['demand_scale_max']
        self.step_length = control_args['step_length'] # SUMO allows to specify how long (in real world time) should each step be.
        self.action_duration = control_args['lower_action_duration']
        self.max_timesteps = control_args['max_timesteps']
        self.max_proposals = control_args['max_proposals']
        self.use_gui = control_args['gui']
        self.auto_start = control_args['auto_start']
        self.warmup_steps = control_args['warmup_steps'] # is a list of two values
        self.sumo_running = False
        self.signal_control_protocol = control_args.get('signal_control_protocol')
        if self.signal_control_protocol not in (None, SHARED_SIGNAL_PROTOCOL):
            raise ValueError('Unknown signal control protocol')
        if self.signal_control_protocol and (self.step_length != 1 or self.action_duration != 10):
            raise ValueError('Shared signal control requires 1 s steps and 10 s decisions')
        self.signal_executor = None
        self.sumo_seed = control_args.get('sumo_seed')
        self.demand_windows = {}
        self.step_count = 0
        
        # Modify file paths to include the unique suffix. Each worker has their own environment and hence their own copy of the trips file.
        self.total_unique_ids_veh = []
        self.total_unique_ids_ped = []
        self.vehicle_output_trips = self.vehicle_output_trips.replace('.xml', f'{self.traci_label}.xml')
        self.pedestrian_output_trips = self.pedestrian_output_trips.replace('.xml', f'{self.traci_label}.xml')
        if current_net_file_path:
            self.net = sumolib.net.readNet(current_net_file_path)

        # This list has to be gotten from the latest network. Will be populated dynamically during reset
        self.tl_ids = ['cluster_172228464_482708521_9687148201_9687148202_#5more'] # Intersection will always be present.
        self.active_slots = np.empty(0, dtype=np.int64)
        self.previous_action = None
        self.num_proposals = 0
        # Number of simulation steps that should occur for each action. 
        self.steps_per_action = int(self.action_duration / self.step_length) # This is also one of the dimensions of the size of the observation buffer
        self.per_timestep_state_dim = control_args['per_timestep_state_dim']

        self.current_action_step = 0 # To track where we are within the curret action's duration
        self.int_tl_phase_groups, self.int_crosswalk_phase_groups = get_intersection_phase_groups()
        self.direction_and_edges  = get_direction_lookup()
        self.tl_lane_dict = get_related_lanes_edges()
        self.tl_pedestrian_status = {} # For pedestrians related to crosswalks attached to TLS.
        self.corrected_occupancy_map = None

        self.pedestrian_existence_times = {}
        self.pedestrian_arrival_times = {}
        self.prev_ped_waiting_time = {}
        self.prev_vehicle_waiting_time = {}
        self.mb_ped_incoming_edges_all = []

        # Create a reverse lookup dict
        self.edge_to_direction = {}
        for direction, edges in self.direction_and_edges.items():
            for edge in edges:
                self.edge_to_direction[edge] = direction

        self.midblock_phase_groups = {0: "rrG", 1: "GGr", 2: "yyr", 3: "rrr"}
        self.directions = ["north", "east", "south", "west"]
        self.turns = ["straight", "right", "left"]
        self.direction_turn_intersection_incoming = [f"{direction}-{turn}" for direction in self.directions for turn in self.turns]
        self.direction_turn_intersection_inside = [f"{direction}-{turn}" for direction in self.directions for turn in ["straight", "left"]] # Exclude right
        self.direction_turn_midblock = ["west-straight", "east-straight"]

        # Normalization parameters (100m cutoff distance with 5m veh and 2.5m gap = approx 13 vehicles)
        self.max_pressure_vehicle = 15
        self.max_pressure_pedestrian = 25 # arbitrary
        self.intersection_cutoff_distance = 100.0
        self.midblock_cutoff_distance = 50.0
        
        # For safety conflict tracking (only in unsignalized)
        self.recorded_conflicts = set()  # Set of unique conflict identifiers to prevent double-counting
        self.total_conflicts = 0         # Running total of unique vehicle-pedestrian conflicts
        self.total_switches = 0 # Only applicable for RL. 
        self.junction_pos_cache = {}

    def _get_vehicle_direction(self, signal_state):
        # Define signal bits for left and right blinkers
        # VEH_SIGNAL_BLINKER_RIGHT = 0b1  # Bit 0
        # VEH_SIGNAL_BLINKER_LEFT = 0b10  # Bit 1

        # Check if left blinker or right blinker is on
        left_blinker = bool(signal_state & 0b10)
        right_blinker = bool(signal_state & 0b1)

        # if left_blinker and not right_blinker:
        #     return "left"
        # elif right_blinker and not left_blinker:
        #     return "right"
        # else:
        #     return "center" # cases where both blinkers are on (emergency) or off
        
        # Direct mapping is faster than multiple conditionals
        if left_blinker:
            if right_blinker:
                return "center"  # both on (emergency)
            return "left"        # left only
        elif right_blinker:
            return "right"       # right only
        return "center"          # both off

    def _step_operations(self, occupancy_map, vehicle_pos_map, print_map=False):
        """
        Some corrections have to be done to the occupancy map every step.
        1. Pedestrians wanting to cross, and pedestrians already crossed both are going to be in the same area in simulation. This mechanism helps distinguish between them. 
        - If a pedestrian is inside (present in outgoing of occupancy map) the crosswalk and hasn't been marked as 'crossed', mark them.
        - Later during tracking, only include unmarked pedestrians in incoming.
        - This mechanism fails when a pedestrian wants to cross twice (however, the probability of this happening is low, assume they wont cross twice).
        2. At the intersection, in some directions (incoming), same lanes are used for L, R, S turns (in case of vehicles and incoming). The straight lane will have repeated entries of those vehicles, remove them.  
        3. Vehicles are only included in the occupancy map if they are within a given distance (due to the length of some edges, far away vehicles may also be present). Only for incoming and outgoing.
        """
        # 1.
        for tl_id in self.tl_ids:
            if tl_id == 'cluster_172228464_482708521_9687148201_9687148202_#5more': # Intersection has 4 outgoing directions

                for direction in self.directions:
                    for area in ["main", "vicinity"]:
                        for ped_id in occupancy_map[tl_id]["pedestrian"]["outgoing"][direction][area]:
                            if self._check_crossing_status(ped_id):
                                self.tl_pedestrian_status[ped_id] = 'crossed'

            else: # Midblock has one direction (north). Also midblocks have no vicinity edges.
                for ped_id in occupancy_map[tl_id]["pedestrian"]["outgoing"]["north"]["main"]:
                    if self._check_crossing_status(ped_id):
                        self.tl_pedestrian_status[ped_id] = 'crossed'

        # 2.
        for direction_turn in self.direction_turn_intersection_incoming:
            if direction_turn not in ['south-straight', 'west-straight', 'east-straight', 'north-straight']:
                direction, _ = direction_turn.split('-')
                straight_lane = f"{direction}-straight"
                straight_lane_vehicles = occupancy_map['cluster_172228464_482708521_9687148201_9687148202_#5more']['vehicle']['incoming'][straight_lane]
                
                # If this vehicles in a non-straight lane are also in the straight lane, remove them from the straight lane.
                for veh_id in occupancy_map['cluster_172228464_482708521_9687148201_9687148202_#5more']['vehicle']['incoming'][direction_turn]:
                    if veh_id in straight_lane_vehicles:
                        occupancy_map['cluster_172228464_482708521_9687148201_9687148202_#5more']['vehicle']['incoming'][straight_lane].remove(veh_id)

        # 3.
        # faster
        for tl_id in self.tl_ids:
            junction_pos = self.junction_pos_cache[tl_id]

            if tl_id == 'cluster_172228464_482708521_9687148201_9687148202_#5more':  # Intersection
                # Incoming
                for direction_turn in self.direction_turn_intersection_incoming:
                    original_list = occupancy_map[tl_id]["vehicle"]["incoming"][direction_turn] # copy
                    occupancy_map[tl_id]["vehicle"]["incoming"][direction_turn] = [
                        veh_id for veh_id in original_list
                        if self._get_vehicle_distance_cached(junction_pos, vehicle_pos_map.get(veh_id)) <= self.intersection_cutoff_distance
                    ]

                # Outgoing
                for direction in self.directions:
                    original_list = occupancy_map[tl_id]["vehicle"]["outgoing"][direction]
                    occupancy_map[tl_id]["vehicle"]["outgoing"][direction] = [
                        veh_id for veh_id in original_list
                        if self._get_vehicle_distance_cached(junction_pos, vehicle_pos_map.get(veh_id)) <= self.intersection_cutoff_distance
                    ]
            # In the new mechanism, in mid-block "inside" vehicles may have duplicate entries in "outgoing" and "incoming" directions.
            # only keep in "incoming" and "outgoing" directions if the vehicle does not exist in "inside" direction.
            else: # Midblock
                for direction_turn in self.direction_turn_midblock:
                    inside_list = occupancy_map[tl_id]["vehicle"]["inside"][direction_turn]
                    occupancy_map[tl_id]["vehicle"]["incoming"][direction_turn] = [
                        veh_id for veh_id in occupancy_map[tl_id]["vehicle"]["incoming"][direction_turn]
                        if veh_id not in inside_list
                    ]
                    occupancy_map[tl_id]["vehicle"]["outgoing"][direction_turn] = [
                        veh_id for veh_id in occupancy_map[tl_id]["vehicle"]["outgoing"][direction_turn]
                        if veh_id not in inside_list
                    ]
            
            # Old mechanism, in the new mechanism, we enforce the cutoff distance earlier.
            # else:  # Midblock
            #     for group in ["incoming", "outgoing"]:
            #         for direction_turn in self.direction_turn_midblock:
            #             original_list = occupancy_map[tl_id]["vehicle"][group][direction_turn]
            #             occupancy_map[tl_id]["vehicle"][group][direction_turn] = [
            #                 veh_id for veh_id in original_list
            #                 if self._get_vehicle_distance_cached(junction_pos, vehicle_pos_map.get(veh_id)) <= self.midblock_cutoff_distance
            #             ]

        # for tl_id in self.tl_ids:
        #     if tl_id == 'cluster_172228464_482708521_9687148201_9687148202_#5more': # Intersection has 8 directions for incoming and 4 for outgoing
        #         # Incoming
        #         for direction_turn in self.direction_turn_intersection_incoming:
        #             for veh_id in occupancy_map[tl_id]["vehicle"]["incoming"][direction_turn]:
        #                 distance = self._get_vehicle_distance_to_junction(tl_id, veh_id)
        #                 if distance > self.intersection_cutoff_distance:
        #                     occupancy_map[tl_id]["vehicle"]["incoming"][direction_turn].remove(veh_id)

        #         # Outgoing
        #         for direction in self.directions:
        #             for veh_id in occupancy_map[tl_id]["vehicle"]["outgoing"][direction]:
        #                 distance = self._get_vehicle_distance_to_junction(tl_id, veh_id)
        #                 if distance > self.intersection_cutoff_distance:
        #                     occupancy_map[tl_id]["vehicle"]["outgoing"][direction].remove(veh_id)

        #     else: # Midblock has two directions (west-straight, east-straight) 
        #         for group in ["incoming", "outgoing"]:
        #             for direction_turn in self.direction_turn_midblock:
        #                 for veh_id in occupancy_map[tl_id]["vehicle"][group][direction_turn]:
        #                     distance = self._get_vehicle_distance_to_junction(tl_id, veh_id)
        #                     if distance > self.midblock_cutoff_distance:
        #                         occupancy_map[tl_id]["vehicle"][group][direction_turn].remove(veh_id)

        # if print_map: 
        #     print("\nOccupancy Map:")
        #     for tl_id, tl_data in occupancy_map.items():
        #         print(f"\nTraffic Light: {tl_id}")
        #         for type in ["vehicle"]:#, "pedestrian"]:
        #             print(f"  {type.capitalize()}:")

        #             for group in tl_data[type].keys():
        #                 print(f"    {group.capitalize()}:")
        #                 for direction, ids in tl_data[type][group].items():
        #                     if type == "vehicle":
        #                         print(f"      {direction.capitalize()}: {len(ids)} [{', '.join(ids)}]")
        #                         for idx in ids:
        #                             distance = self._get_vehicle_distance_to_junction(tl_id, idx)
        #                             print(f"        {idx}: {distance:.2f}m")   
        #                     else: 
        #                         for area in ids.keys():
        #                             print(f"      {direction.capitalize(), area}: {len(ids[area])} [{', '.join(ids[area])}]")

        return occupancy_map
    
    def _get_vehicle_distance_cached(self, junction_pos, vehicle_pos):
        """
        Helper
        """
        # if vehicle_pos is None:
        #     return float('inf')  # Vehicle not in simulation
        dx = junction_pos[0] - vehicle_pos[0]
        dy = junction_pos[1] - vehicle_pos[1]
        return math.hypot(dx, dy)  # More stable than sqrt(x^2 + y^2)

    def _get_vehicle_distance_to_junction(self, junction_id, vehicle_id):
        """
        Calculate the distance between a vehicle and a specific junction.

        :param junction_id: ID of the junction
        :param vehicle_id: ID of the vehicle
        :return: Distance between the vehicle and the junction in meters
        """
        try:
            # Get the x, y coordinates of the junction
            junction_pos = traci.junction.getPosition(junction_id)
            vehicle_pos = traci.vehicle.getPosition(vehicle_id)
            # Euclidean distance
            distance = math.sqrt(
                (junction_pos[0] - vehicle_pos[0])**2 + 
                (junction_pos[1] - vehicle_pos[1])**2
            )
            return distance

        except traci.TraCIException as e:
            print(f"Error calculating distance: {e}")
            return None
    
    def _get_pressure_dict(self, occupancy_map, print_dict=False):
        """
        Update the data structure that holds info about pressure (in each outgoing direction).
        For both vehicles and pedestrians:
            - Pressure = incoming - outgoing
        """
        pressure_dict = {tl_id: {"vehicle": {}, "pedestrian": {}} for tl_id in self.tl_ids}
        # Intersection
        # 1. Vehicles
        veh_pressure = {d: 0 for d in self.directions}
        for outgoing_direction in self.directions:
            incoming = 0 
            if outgoing_direction == 'north':
                incoming += len(occupancy_map["cluster_172228464_482708521_9687148201_9687148202_#5more"]["vehicle"]["incoming"]["south-straight"])
                incoming += len(occupancy_map["cluster_172228464_482708521_9687148201_9687148202_#5more"]["vehicle"]["incoming"]["east-right"])
                incoming += len(occupancy_map["cluster_172228464_482708521_9687148201_9687148202_#5more"]["vehicle"]["incoming"]["west-left"])
                incoming += len(occupancy_map["cluster_172228464_482708521_9687148201_9687148202_#5more"]["vehicle"]["inside"]["south-straight"])
                incoming += len(occupancy_map["cluster_172228464_482708521_9687148201_9687148202_#5more"]["vehicle"]["inside"]["west-left"])

            elif outgoing_direction == 'south':
                incoming += len(occupancy_map["cluster_172228464_482708521_9687148201_9687148202_#5more"]["vehicle"]["incoming"]["north-straight"])
                incoming += len(occupancy_map["cluster_172228464_482708521_9687148201_9687148202_#5more"]["vehicle"]["incoming"]["east-left"])
                incoming += len(occupancy_map["cluster_172228464_482708521_9687148201_9687148202_#5more"]["vehicle"]["incoming"]["west-right"])
                incoming += len(occupancy_map["cluster_172228464_482708521_9687148201_9687148202_#5more"]["vehicle"]["inside"]["north-straight"])
                incoming += len(occupancy_map["cluster_172228464_482708521_9687148201_9687148202_#5more"]["vehicle"]["inside"]["east-left"])

            elif outgoing_direction == 'east':
                incoming += len(occupancy_map["cluster_172228464_482708521_9687148201_9687148202_#5more"]["vehicle"]["incoming"]["west-straight"])
                incoming += len(occupancy_map["cluster_172228464_482708521_9687148201_9687148202_#5more"]["vehicle"]["incoming"]["north-left"])
                incoming += len(occupancy_map["cluster_172228464_482708521_9687148201_9687148202_#5more"]["vehicle"]["incoming"]["south-right"])
                incoming += len(occupancy_map["cluster_172228464_482708521_9687148201_9687148202_#5more"]["vehicle"]["inside"]["west-straight"])
                incoming += len(occupancy_map["cluster_172228464_482708521_9687148201_9687148202_#5more"]["vehicle"]["inside"]["north-left"])

            elif outgoing_direction == 'west':
                incoming += len(occupancy_map["cluster_172228464_482708521_9687148201_9687148202_#5more"]["vehicle"]["incoming"]["east-straight"])
                incoming += len(occupancy_map["cluster_172228464_482708521_9687148201_9687148202_#5more"]["vehicle"]["incoming"]["north-right"])
                incoming += len(occupancy_map["cluster_172228464_482708521_9687148201_9687148202_#5more"]["vehicle"]["incoming"]["south-left"])
                incoming += len(occupancy_map["cluster_172228464_482708521_9687148201_9687148202_#5more"]["vehicle"]["inside"]["east-straight"])
                incoming += len(occupancy_map["cluster_172228464_482708521_9687148201_9687148202_#5more"]["vehicle"]["inside"]["south-left"])

            outgoing = len(occupancy_map["cluster_172228464_482708521_9687148201_9687148202_#5more"]["vehicle"]["outgoing"][outgoing_direction])
            veh_pressure[outgoing_direction] = incoming - outgoing
            pressure_dict["cluster_172228464_482708521_9687148201_9687148202_#5more"]["vehicle"][outgoing_direction] = veh_pressure[outgoing_direction]

        # 2. Pedestrians 
        # Incmoing (0.7*main + 0.3*vicinity)
        ped_pressure = {d: 0 for d in self.directions}
        for outgoing_direction in self.directions:
            incoming = 0.7 * len(occupancy_map["cluster_172228464_482708521_9687148201_9687148202_#5more"]["pedestrian"]["incoming"][outgoing_direction]["main"]) + \
                0.3 * len(occupancy_map["cluster_172228464_482708521_9687148201_9687148202_#5more"]["pedestrian"]["incoming"][outgoing_direction]["vicinity"])
            outgoing = len(occupancy_map["cluster_172228464_482708521_9687148201_9687148202_#5more"]["pedestrian"]["outgoing"][outgoing_direction]["main"]) # vicinity does not exist
            ped_pressure[outgoing_direction] = incoming - outgoing
            pressure_dict["cluster_172228464_482708521_9687148201_9687148202_#5more"]["pedestrian"][outgoing_direction] = ped_pressure[outgoing_direction]

        # Midblock
        for tl_id in self.tl_ids[1:]:
            # 1. Vehicles (have two directions)
            veh_pressure = {d: 0 for d in self.direction_turn_midblock}
            for direction in self.direction_turn_midblock:
                incoming = len(occupancy_map[tl_id]["vehicle"]["incoming"][direction])
                outgoing = len(occupancy_map[tl_id]["vehicle"]["outgoing"][direction])
                veh_pressure[direction] = incoming - outgoing
                pressure_dict[tl_id]["vehicle"][direction] = veh_pressure[direction]

            # 2. Pedestrians (have one direction)
            # Vicinity does not exist
            ped_pressure = {"north": 0}
            incoming = len(occupancy_map[tl_id]["pedestrian"]["incoming"]["north"]["main"])
            outgoing = len(occupancy_map[tl_id]["pedestrian"]["outgoing"]["north"]["main"])
            ped_pressure["north"] = incoming - outgoing
            pressure_dict[tl_id]["pedestrian"]["north"] = ped_pressure["north"]

        if print_dict:
            print("\nPressure Dictionary:")
            print("===================")
            print(f"\nIntersection:")
            print(f"Vehicles: {pressure_dict['cluster_172228464_482708521_9687148201_9687148202_#5more']['vehicle']}")
            print(f"Pedestrians: {pressure_dict['cluster_172228464_482708521_9687148201_9687148202_#5more']['pedestrian']}")
            print("\nMidblock TLs:")
            for tl_id in self.tl_ids[1:]:
                print(f"{tl_id}:")
                print(f"  Vehicles: {pressure_dict[tl_id]['vehicle']}")
                print(f"  Pedestrians: {{'north': {pressure_dict[tl_id]['pedestrian']['north']}}}")

        return pressure_dict
    
    def _get_occupancy_map(self, ):
        """
        Occupancy map = dict with vehicle and pedestrian information in the neighbourhood of all TLs
        If the same lane is used for multiple turns (straight, right, left), the indicator light (turns ON about 100m as a vehicle approaches the junction) of vehicle is used to determine the actual direction. 
        occupancy_map stores the ids.
        """
        occupancy_map = {}
        for tl_id in self.tl_ids:
            occupancy_map[tl_id] = {
                "vehicle": {
                    "incoming": {},
                    "inside": {}, # Inside the junction (since mid-block TLs are small, may not apply to them)
                    "outgoing": {}
                },
                "pedestrian": {
                    "incoming": {"north": {}},
                    "outgoing": {"north": {}}
                }
            }
        
        person_ids = traci.person.getIDList()
        person_road_map = {ped_id: traci.person.getRoadID(ped_id) for ped_id in person_ids}
        vehicle_ids = traci.vehicle.getIDList()
        vehicle_pos_map = {veh_id: traci.vehicle.getPosition(veh_id) for veh_id in vehicle_ids}
        vehicle_occupancy_midblock = self._get_vehicle_occupancy_midblock(vehicle_ids, vehicle_pos_map)

        for tl_id in self.tl_ids:
            for type in ["vehicle", "pedestrian"]:

                if type == "vehicle": # Vehicles 
                    # At the intersection, they have 8 directions for incoming + inside and 4 directions for outgoing
                    if tl_id == 'cluster_172228464_482708521_9687148201_9687148202_#5more':
                        for group_key, direction_turns_value in [("incoming", self.direction_turn_intersection_incoming), ("inside", self.direction_turn_intersection_inside)]:
                            for direction_turn in direction_turns_value:
                                occupancy_map[tl_id][type][group_key][direction_turn] = []
                                lanes_edges = self.tl_lane_dict[tl_id][type][group_key][direction_turn]

                                for lane_edge in lanes_edges:
                                    if lane_edge != '-1':  # Skip lanes that are common for all directions
                                        veh_ids = traci.lane.getLastStepVehicleIDs(lane_edge) if "edge" not in lane_edge else traci.edge.getLastStepVehicleIDs(lane_edge.split('.')[1]) 
                                        occupancy_map[tl_id][type][group_key][direction_turn].extend(veh_ids)
                                    
                                    else:
                                        # Look at the indicator light of the vehicle to get the actual direction.
                                        # Get the direction, turn. Then get all vehicles in the straight lane 
                                        # If there are multiple -1s, this case can occur multiple times.
                                        direction, turn = direction_turn.split('-')
                                        straight_lane_edge = self.tl_lane_dict[tl_id][type][group_key][f"{direction}-straight"][0] # Use 0 because in the current implementation, there is only one straight lane per direction
                                        veh_ids = traci.lane.getLastStepVehicleIDs(straight_lane_edge) if "edge" not in straight_lane_edge else traci.edge.getLastStepVehicleIDs(straight_lane_edge.split('.')[1]) 

                                        if len(veh_ids) > 0:
                                            new_ids = []
                                            for veh_id in veh_ids:
                                                veh_direction = self._get_vehicle_direction(traci.vehicle.getSignals(veh_id))
                                                if veh_direction == turn:
                                                    new_ids.append(veh_id)

                                            occupancy_map[tl_id][type][group_key][direction_turn].extend(new_ids)

                        # outgoing
                        for direction in self.directions:
                            occupancy_map[tl_id][type]["outgoing"][direction] = []
                            lanes_edges = self.tl_lane_dict[tl_id][type]["outgoing"][direction]
                            for lane_edge in lanes_edges:
                                veh_ids = traci.lane.getLastStepVehicleIDs(lane_edge) if "edge" not in lane_edge else traci.edge.getLastStepVehicleIDs(lane_edge.split('.')[1]) 
                                occupancy_map[tl_id][type]["outgoing"][direction].extend(veh_ids)

                    else:  # For midblock TLs, they have 2 directions each for incoming, inside, and outgoing
                        
                        # Old mechanism has blind spots
                        # for group in ["incoming", "outgoing"]: 
                        #     for direction_turn in self.direction_turn_midblock:
                        #         occupancy_map[tl_id][type][group][direction_turn] = []
                        #         lanes_edges = self.tl_lane_dict[tl_id][type][group][direction_turn]
                        #         for lane_edge in lanes_edges:
                        #             veh_ids = traci.lane.getLastStepVehicleIDs(lane_edge) if "edge" not in lane_edge else traci.edge.getLastStepVehicleIDs(lane_edge.split('.')[1]) 
                        #             occupancy_map[tl_id][type][group][direction_turn].extend(veh_ids)

                        # New mechanism to get vehicles incoming and outgoing for midblock TLs
                        for group in ["incoming", "outgoing"]:
                            for direction in ["west-straight", "east-straight"]:
                                occupancy_map[tl_id][type][group][direction] = vehicle_occupancy_midblock[tl_id][group][direction]
                                
                        # Inside has to be done in a computationally expensive way (because its an interal edge not detected either by edge.getLastStepVehicleIDs or lane.getLastStepVehicleIDs). 
                        # Get all vehicles in the simulation and compare their road id with the lanes_edges.
                        for direction_turn in self.direction_turn_midblock:
                            occupancy_map[tl_id][type]["inside"][direction_turn] = []
                            # lanes_edges = self.tl_lane_dict[tl_id][type]["inside"][direction_turn]
                            # for veh_id in traci.vehicle.getIDList():
                            #     if traci.vehicle.getRoadID(veh_id) in lanes_edges:
                            #         occupancy_map[tl_id][type]["inside"][direction_turn].append(veh_id)

                            # faster
                            filtered_veh_ids = [veh_id for veh_id in vehicle_ids if traci.vehicle.getRoadID(veh_id) in self.tl_lane_dict[tl_id][type]["inside"][direction_turn]]
                            occupancy_map[tl_id][type]["inside"][direction_turn].extend(filtered_veh_ids)

                else: # Pedestrians: lanes or edges further divided into main and vicinity, each with incoming and outgoing
                    for group in ["incoming", "outgoing"]:
                    
                        # Intersection has four directions.
                        if tl_id == 'cluster_172228464_482708521_9687148201_9687148202_#5more':
                            for direction in self.directions:
                                occupancy_map[tl_id][type][group][direction] = {}
                                for area in ["main", "vicinity"]:
                                    occupancy_map[tl_id][type][group][direction][area] = []
                                    # lanes_edges = self.tl_lane_dict[tl_id][type][group][direction][area]
                                    # for lane_edge in lanes_edges:
                                    #     if lane_edge.startswith(':'): # Check if it's an internal lane. If yes, do it the computationally expensive way. Get all persons in the simulation
                                            
                                    #         for ped_id in traci.person.getIDList():
                                    #             if traci.person.getRoadID(ped_id) == lane_edge:
                                    #                 if group == "incoming":
                                    #                     if self._check_crossing_status(ped_id):
                                    #                         occupancy_map[tl_id][type][group][direction][area].append(ped_id)
                                    #                 else:
                                    #                     occupancy_map[tl_id][type][group][direction][area].append(ped_id)
                                    #     else:
                                    #         for ped_id in traci.edge.getLastStepPersonIDs(lane_edge):
                                    #             if group == "incoming":
                                    #                 if self._check_crossing_status(ped_id):
                                    #                     occupancy_map[tl_id][type][group][direction][area].append(ped_id)
                                    #             else:
                                    #                 occupancy_map[tl_id][type][group][direction][area].append(ped_id)
                                    
                                    # faster, Pre-fetch 
                                    for lane_edge in self.tl_lane_dict[tl_id][type][group][direction][area]:
                                        if lane_edge.startswith(':'):  # Internal lane
                                            # Filter only the people on this internal lane
                                            relevant_peds = [ped_id for ped_id, road_id in person_road_map.items() if road_id == lane_edge]
                                        else:
                                            # External lane: get people directly from edge
                                            relevant_peds = traci.edge.getLastStepPersonIDs(lane_edge)

                                        for ped_id in relevant_peds:
                                            if group == "incoming":
                                                if self._check_crossing_status(ped_id):
                                                    occupancy_map[tl_id][type][group][direction][area].append(ped_id)
                                            else:
                                                occupancy_map[tl_id][type][group][direction][area].append(ped_id)


                        else: # Midblock has one direction (north). Also midblocks have no vicinity edges.
                            occupancy_map[tl_id][type][group]["north"]["main"] = []
                            #lanes_edges = self.tl_lane_dict[tl_id][type][group]["north"]["main"]
                            # for lane_edge in lanes_edges:
                            #     if lane_edge.startswith(':'): # Check if it's an internal lane. If yes, do it the computationally expensive way. Get all persons in the simulation
                                    
                            #         for ped_id in person_ids:
                            #             if traci.person.getRoadID(ped_id) == lane_edge:
                                            
                            #                 if group == "incoming":
                            #                     if self._check_crossing_status(ped_id):
                            #                         occupancy_map[tl_id][type][group]["north"]["main"].append(ped_id)
                            #                 else:
                            #                     occupancy_map[tl_id][type][group]["north"]["main"].append(ped_id)
                            #     else:

                            #         for ped_id in traci.edge.getLastStepPersonIDs(lane_edge):
                            #             if group == "incoming":
                            #                 if self._check_crossing_status(ped_id):
                            #                     occupancy_map[tl_id][type][group]["north"]["main"].append(ped_id)
                            #             else:
                            #                 occupancy_map[tl_id][type][group]["north"]["main"].append(ped_id)

                            # faster, Pre-fetch 
                            for lane_edge in self.tl_lane_dict[tl_id][type][group]["north"]["main"]:
                                if lane_edge.startswith(':'):  # Internal lane
                                    # Filter only the people on this internal lane
                                    relevant_peds = [ped_id for ped_id, road_id in person_road_map.items() if road_id == lane_edge]
                                else:
                                    # External lane: get people directly from edge
                                    relevant_peds = traci.edge.getLastStepPersonIDs(lane_edge)

                                for ped_id in relevant_peds:
                                    if group == "incoming":
                                        if self._check_crossing_status(ped_id):
                                            occupancy_map[tl_id][type][group]["north"]["main"].append(ped_id)
                                    else:
                                        occupancy_map[tl_id][type][group]["north"]["main"].append(ped_id)

        return occupancy_map, vehicle_pos_map
    
    def _check_crossing_status(self, pedestrian_id):
        """
        There is a 1 timestep delay in marking pedestrians as crossed.
        In the first timestep, the pedestrian will not be in the tl_pedestrian_status dictionary.
        """
        return True if pedestrian_id not in self.tl_pedestrian_status or self.tl_pedestrian_status[pedestrian_id] != "crossed" else False

    @property
    def action_space(self):
        """
        * Simple action space
        The control action is represented as a 9-bit string for each traffic light:
        - First digit: Intersection signal (4 mutually exclusive configurations)
            0 = allow vehicular traffic through East-West only
            1 = allow vehicular traffic through North-South only
            2 = allow dedicated left turns through N-E and S-W only
            3 = disallow vehicular traffic in all directions
        - Next max_proposals bits: Mid-block crosswalk signals (each bit is Bernoulli)
            1  = allow vehicles to cross (pedestrians red) at a given mid-block segment
            0  = disallow vehicles to cross (pedestrians green) at that segment
        """
        # Simple action
        action_space = [4] + [2] * self.max_proposals  # First value is 0-3 for intersection, remaining are binary
        return gym.spaces.MultiDiscrete(action_space)

    @property
    def observation_space(self):
        """
        * Observation space is defined per action step (i.e. accumulated over action duration)
        """
        return gym.spaces.Box(low=-1,
                              high=np.inf,
                              shape=(self.steps_per_action, 
                                     self.per_timestep_state_dim), 
                                     dtype=np.float32)

    def train_step(self, action):
        """
        If Tl = True, operating in eval mode with TL.
        """
        if self.previous_action is None:
            self.previous_action = action

        reward = 0
        done = False
        observation_buffer = []
        action = np.array(action)
        # switch detection does not need to be done every timestep. 
        switch_state, _ = self._detect_switch(action, self.previous_action)
 
        for i in range(self.steps_per_action): # Run simulation steps for the duration of the action

            # Apply action is called every timestep (return information useful for reward calculation)
            current_phase = self._apply_action(action, i, switch_state)
            traci.simulationStep() # Step length is the simulation time that elapses when each time this is called.
            self._update_pedestrian_existence_times()
            self.step_count += 1
            obs = self._get_observation(current_phase)
            observation_buffer.append(obs)
            self._get_pedestrian_arrival_times()
            # time.sleep(1.0)
            
        # outside the loop
        # Do before reward calculation
        # pressure_dict = self._get_pressure_dict(self.corrected_occupancy_map)
        # Reward outside the loop (only once per duration)
        reward = self._get_control_reward(self.corrected_occupancy_map, switch_state, pressure_dict = None) # pressure dict used only in pressure-based reward
        self.previous_action = action

        # Check if episode is done (outside the for loop, otherwise it would create a broken observation)
        if self._check_done():
            done = True

        observation = np.asarray(observation_buffer, dtype=np.float32) 
        #print(f"\nObservation shape: {observation.shape}, type: {type(observation)}, value: {observation}")
        return observation, reward, done, False, {} # info is empty
    
    def eval_step(self, action, tl= False, unsignalized= False):
        """
        Step during evaluation.
        """
        if not self.sumo_running:
            raise Exception("Environment is not running. Call reset() to start the environment.")
        if self.previous_action is None:
            self.previous_action = action

        reward = 0
        info = {"vehicle_wait": 0.0, "pedestrian_wait": 0.0}
        done = False
        observation_buffer = []
        action = np.array(action)
        # switch detection does not need to be done every timestep. 
        switch_state, full_switch_state = self._detect_switch(action, self.previous_action)
        
        #TODO: Placing an if-else here (at almost the innermost loop) is very inefficient.
        if tl: 
            if unsignalized: # Set all Midblock as green.
                for i in range(self.steps_per_action): # Run simulation steps for the duration of the action
                    current_phase = [1]*len(self.tl_ids) # random phase
                    # Midblock
                    for j in range(1, len(self.tl_ids)):
                        tl_id = self.tl_ids[j]
                        mb_phase_string = "GgggggGG" if tl_id == "cluster_9740157181_9740483933" else "GGG"
                        traci.trafficlight.setRedYellowGreenState(tl_id, mb_phase_string)
                    traci.simulationStep() # Step length is the simulation time that elapses when each time this is called.
                    self._update_pedestrian_existence_times()
                    self.step_count += 1
                    obs = self._get_observation(current_phase)
                    observation_buffer.append(obs)

                    # Count near-conflicts after each step (after each simulation step has been applied)
                    self.total_conflicts += self._count_near_conflicts(self.corrected_occupancy_map)
                    self._get_pedestrian_arrival_times()
                    info["vehicle_wait"] += self.get_vehicle_waiting_time()
                    info["pedestrian_wait"] += self.get_pedestrian_waiting_time()

            else: # Apply action for TL (For evaluation)
                for i in range(self.steps_per_action): # Run simulation steps for the duration of the action
                    current_phase = [1]*len(self.tl_ids) # random phase
                    traci.simulationStep() # Step length is the simulation time that elapses when each time this is called.
                    self._update_pedestrian_existence_times()
                    self.step_count += 1
                    obs = self._get_observation(current_phase)
                    observation_buffer.append(obs)
                    self._get_pedestrian_arrival_times()
                    info["vehicle_wait"] += self.get_vehicle_waiting_time()
                    info["pedestrian_wait"] += self.get_pedestrian_waiting_time()
        else:
            self.total_switches += sum(full_switch_state)
            # Only apply the action from policy if not TL.   
            for i in range(self.steps_per_action): # Run simulation steps for the duration of the action

                # Apply action is called every timestep (return information useful for reward calculation)
                current_phase = self._apply_action(action, i, switch_state)
                traci.simulationStep() # Step length is the simulation time that elapses when each time this is called.
                self._update_pedestrian_existence_times()
                self.step_count += 1
                obs = self._get_observation(current_phase)
                observation_buffer.append(obs)
                
                self._get_pedestrian_arrival_times()
                info["vehicle_wait"] += self.get_vehicle_waiting_time()
                info["pedestrian_wait"] += self.get_pedestrian_waiting_time()

        # Outside the loop, before reward calculation
        # pressure_dict = self._get_pressure_dict(self.corrected_occupancy_map)
        # Reward outside the loop (only once per duration)
        reward = self._get_control_reward(self.corrected_occupancy_map, switch_state, pressure_dict = None) # pressure dict used only in pressure-based reward
        self.previous_action = action

        # Check if episode is done (outside the for loop, otherwise it would create a broken observation)
        if self._check_done():
            done = True

        observation = np.asarray(observation_buffer, dtype=np.float32) 
        return observation, reward, done, False, info

    def _detect_switch(self, current_action, previous_action):
        """
        Detect a switch in all the components (the vehicle part at the intersection considered as one component) i.e., total 8 components
        For Intersection:
            - A switch is detected if action changes from: 
                - 0 to 1: E-W green to N-S green
                - 1 to 0: N-S green to E-W green
                - 2 to 0: N-E + S-W green (Dedicated left) to E-W green
        For Midblock: 
            - Action = 1 means vehicle green (pedestrian red)
            - Action = 0 means vehicle red (pedestrian green)
            - We only enfore a mandatory yellow phase if there is a 1 to 0 transition i.e., only consider this as a switch.
        """

        current_action = ''.join(map(str, current_action))
        previous_action = ''.join(map(str, previous_action))
        current_intersection_action = current_action[0:1]
        current_mid_block_action = current_action[1:]  
        previous_intersection_action = previous_action[0:1]
        previous_mid_block_action = previous_action[1:]
        
        switch_state = []
        intersection_switch = [int(c1 == '0' and c2 == '1') or int(c1 == '1' and c2 == '0') or int(c1 == '2' and c2 == '0') for c1, c2 in zip(previous_intersection_action, current_intersection_action)]
        switch_state.extend(intersection_switch)
        midblock_switch = [int(c1 == '1' and c2 == '0') for c1, c2 in zip(previous_mid_block_action, current_mid_block_action)] # only detect 1->0 transitions
        switch_state.extend(midblock_switch)

        # For the plot, also detect all the switches (full = as long as the phase changes).
        # full_switch_state = []
        # for i in range(len(current_action)):
        #     full_switch_state.append(int(current_action[i] != previous_action[i]))
        # print(f"Full switch state: {full_switch_state}\n")
        return switch_state, []

    def _get_observation(self, current_phase, print_map=True):
        """
        * Per step observation size = 123
        * Assume max_proposals = 10
        * Composed of: 
            - Current phase information (1 + 10 elements that transitions throughout the action duration)
            - Intersection (32 elements):
                - Vehicles: 
                    - Incoming (12 directions)
                    - Inside (8 directions)
                    - Outgoing (4 directions)
                - Pedestrians
                    - Incoming (4 directions)
                    - Outgoing (4 directions)
            - max_proposals Midblocks (10 x 8 = 80 elements):
                - Vehicles: 
                    - Incoming (2 directions)
                    - Inside (2 directions)
                    - Outgoing (2 directions)
                - Pedestrians
                    - Incoming (1 direction)
                    - Outgoing (1 direction)
            - 11 + 32 + 80 = 123

        * Each action persists for a number of timesteps (transitioning though phases); observation is collected at each timestep.
        * Pressure itself is not a part of the observation (only used for reward calculation).
        * Add full info about one TL at a time (preserve locality).
        * Signal locality: each signal's fields occupy its declared slot, even after layout edits.
        * Temporal locality: Observations are stacked verticalled one timestep to the next.
        * Normalize everything with a max normalizer.

        * Main and vicinity are given equal weightage (1). A single person in observation areas should represent same val.
        """

        # initialize a full observation
        observation = -1 * np.ones(self.per_timestep_state_dim, dtype=np.float32)
        occupancy_map, vehicle_pos_map = self._get_occupancy_map()
        self.corrected_occupancy_map = self._step_operations(occupancy_map, vehicle_pos_map, print_map=print_map)
        observation[0] = current_phase[0]
        observation[1 + self.active_slots] = np.asarray(current_phase[1:], dtype=np.float32)
        length = self.max_proposals + 1  # Reserve every phase slot before occupancy.
        
        # Intersection
        # - vehicles
        int_incoming = [] # 12 directions
        for direction_turn in self.direction_turn_intersection_incoming:
            int_incoming.append(len(self.corrected_occupancy_map['cluster_172228464_482708521_9687148201_9687148202_#5more']["vehicle"]["incoming"][direction_turn]))
        observation[length:length + len(int_incoming)] = np.array(int_incoming, dtype=np.float32)
        length += len(int_incoming)

        int_inside = [] # 8 directions
        for direction_turn in self.direction_turn_intersection_inside:
            int_inside.append(len(self.corrected_occupancy_map['cluster_172228464_482708521_9687148201_9687148202_#5more']["vehicle"]["inside"][direction_turn]))
        observation[length:length + len(int_inside)] = np.array(int_inside, dtype=np.float32)
        length += len(int_inside)
  
        int_outgoing = [] # 4 directions
        for direction in self.directions:
            int_outgoing.append(len(self.corrected_occupancy_map['cluster_172228464_482708521_9687148201_9687148202_#5more']["vehicle"]["outgoing"][direction]))
        observation[length:length + len(int_outgoing)] = np.array(int_outgoing, dtype=np.float32)
        length += len(int_outgoing)

        # - pedestrians
        int_incoming_ped = [] # 4 directions. This contains both main and vicinity areas. Sum (70% = main, 30% = vicinity)
        for direction in self.directions:
            main = len(self.corrected_occupancy_map['cluster_172228464_482708521_9687148201_9687148202_#5more']["pedestrian"]["incoming"][direction]["main"])
            vicinity = len(self.corrected_occupancy_map['cluster_172228464_482708521_9687148201_9687148202_#5more']["pedestrian"]["incoming"][direction]["vicinity"])
            int_incoming_ped.append(main + vicinity)
        observation[length:length + len(int_incoming_ped)] = np.array(int_incoming_ped, dtype=np.float32)
        length += len(int_incoming_ped)

        int_outgoing_ped = [] # 4 directions
        for direction in self.directions:
            int_outgoing_ped.append(len(self.corrected_occupancy_map['cluster_172228464_482708521_9687148201_9687148202_#5more']["pedestrian"]["outgoing"][direction]["main"]))
        observation[length:length + len(int_outgoing_ped)] = np.array(int_outgoing_ped, dtype=np.float32)
        length += len(int_outgoing_ped)

        # Midblock fields occupy the same sparse slots as their phases and action heads.
        for slot, tl_id in zip(self.active_slots, self.tl_ids[1:]):
            length = self.max_proposals + 1 + 32 + 8 * slot
            # - vehicles
            mb_incoming = [] # 2 directions
            for direction_turn in self.direction_turn_midblock:
                mb_incoming.append(len(self.corrected_occupancy_map[tl_id]["vehicle"]["incoming"][direction_turn]))
            observation[length:length + len(mb_incoming)] = np.array(mb_incoming, dtype=np.float32)
            length += len(mb_incoming)

            mb_inside = [] # 2 directions
            for direction_turn in self.direction_turn_midblock:
                mb_inside.append(len(self.corrected_occupancy_map[tl_id]["vehicle"]["inside"][direction_turn]))
            observation[length:length + len(mb_inside)] = np.array(mb_inside, dtype=np.float32)
            length += len(mb_inside)

            mb_outgoing = [] # 2 directions
            for direction in self.direction_turn_midblock:
                mb_outgoing.append(len(self.corrected_occupancy_map[tl_id]["vehicle"]["outgoing"][direction]))
            observation[length:length + len(mb_outgoing)] = np.array(mb_outgoing, dtype=np.float32)
            length += len(mb_outgoing)

            # - pedestrians
            # Incoming, 1 direction
            mb_incoming_ped_count = len(self.corrected_occupancy_map[tl_id]["pedestrian"]["incoming"]["north"]["main"])
            observation[length] = np.float32(mb_incoming_ped_count) 
            length += 1 # Increment by 1

            # Outgoing, 1 direction
            mb_outgoing_ped_count = len(self.corrected_occupancy_map[tl_id]["pedestrian"]["outgoing"]["north"]["main"])
            observation[length] = np.float32(mb_outgoing_ped_count) 
            length += 1 # 
        
        # Normalize with running mean and std
        observation = np.asarray(observation, dtype=np.float32)

        # # Breakdown print of observation components with sub-step info
        # # Determine sub-step within the current action if possible
        # try:
        #     substep = (self.step_count - 1) % self.steps_per_action + 1
        #     print(f"\nSub-step within action: {substep}/{self.steps_per_action}")
        # except Exception:
        #     pass
        # offset = 0
        # # Phase information
        # phase_len = len(current_phase)
        # print("Phase:", ' '.join(str(int(x)) for x in observation[offset:offset+phase_len]))
        # offset += phase_len
        # # Intersection vehicle incoming
        # inc_len = len(self.direction_turn_intersection_incoming)
        # inc_vals = observation[offset:offset+inc_len]
        # print("Intersection veh incoming:", ' '.join(f"{x:.4f}" for x in inc_vals))
        # offset += inc_len
        # # Intersection vehicle inside
        # inside_len = len(self.direction_turn_intersection_inside)
        # inside_vals = observation[offset:offset+inside_len]
        # print("Intersection veh inside:", ' '.join(f"{x:.4f}" for x in inside_vals))
        # offset += inside_len
        # # Intersection vehicle outgoing
        # out_len = len(self.directions)
        # out_vals = observation[offset:offset+out_len]
        # print("Intersection veh outgoing:", ' '.join(f"{x:.4f}" for x in out_vals))
        # offset += out_len
        # # Intersection pedestrian incoming
        # ped_in_len = len(self.directions)
        # ped_in_vals = observation[offset:offset+ped_in_len]
        # print("Intersection ped incoming:", ' '.join(f"{x:.4f}" for x in ped_in_vals))
        # offset += ped_in_len
        # # Intersection pedestrian outgoing
        # ped_out_len = len(self.directions)
        # ped_out_vals = observation[offset:offset+ped_out_len]
        # print("Intersection ped outgoing:", ' '.join(f"{x:.4f}" for x in ped_out_vals))
        # offset += ped_out_len
        # # Midblock traffic lights breakdown
        # for idx_tl in range(1, len(self.tl_ids)):
        #     tl_id = self.tl_ids[idx_tl]
        #     # Vehicles incoming
        #     mb_inc = observation[offset:offset+len(self.direction_turn_midblock)]
        #     offset += len(self.direction_turn_midblock)
        #     # Vehicles inside
        #     mb_ins = observation[offset:offset+len(self.direction_turn_midblock)]
        #     offset += len(self.direction_turn_midblock)
        #     # Vehicles outgoing
        #     mb_out = observation[offset:offset+len(self.direction_turn_midblock)]
        #     offset += len(self.direction_turn_midblock)
        #     # Pedestrians incoming
        #     mb_p_in = observation[offset]
        #     offset += 1
        #     # Pedestrians outgoing
        #     mb_p_out = observation[offset]
        #     offset += 1
        #     print(f"Midblock {tl_id}: veh inc [{' '.join(f'{x:.4f}' for x in mb_inc)}] | veh inside [{' '.join(f'{x:.4f}' for x in mb_ins)}] | veh out [{' '.join(f'{x:.4f}' for x in mb_out)}] | ped inc {mb_p_in:.4f} | ped out {mb_p_out:.4f}")
        
        return observation

    def _apply_action(self, action, current_action_step, switch_state):
        """
        apply_action is the enforcement of the chosen action (9-bit string) to the traffic lights and crosswalks at every step.
        Based on the switch_state, a 4-second mandatory yellow phase is enforced to the direction/ light that is turning red, before starting the new phase in the other light.
        For Intersection: 

        For each Midblock:
        - Action = 0: rrG: Red for both vehicle directions, green for pedestrian crosswalk
        - Action = 1: GGr: Green for the 2 vehicle directions, red for pedestrian crosswalk
        - If switch state is 1 (that means current action is 0):
             - Action 2: 4 timesteps of yyr: Yellow for the 2 vehicle directions, red for pedestrian crosswalk
             - Action 3: 1 timestep of rrr (all-red)
             - Action 0: 5 timestep of rrG

        * It does not matter what phases are specified in the Tlogic in net file, we override it from here.
        -----
        In the design env, there are variable number of Mid-block TLs. 
        
        """
        if self.signal_control_protocol == SHARED_SIGNAL_PROTOCOL:
            if self.signal_executor is None:
                self.signal_executor = SignalExecutor(self)
            return self.signal_executor.apply(action)
        #print(f"Action: {action}, switch_state: {switch_state}, type: {type(switch_state)}")
        current_phase = []

        # Intersection
        int_action = action[0]
        int_switch_state = switch_state[0]
        if int_switch_state == 0:
            int_vehicle_phase_group = int_action
            vehicle_phase_string = self.int_tl_phase_groups[int_vehicle_phase_group]
        else:
            # If switch_state is 1 and current_action is 1, select phase group 4
            # If switch_state is 1 and current_action is 0, select phase group 5
            if int_action == 1:
                int_vehicle_phase_group = 4
                vehicle_phase_string = self.int_tl_phase_groups[int_vehicle_phase_group][current_action_step]
            elif int_action == 0:
                int_vehicle_phase_group = 5
                vehicle_phase_string = self.int_tl_phase_groups[int_vehicle_phase_group][current_action_step]
            else: # coming from dedicated left turn (keep as is)
                int_vehicle_phase_group = int_action
                vehicle_phase_string = self.int_tl_phase_groups[int_vehicle_phase_group]

        current_phase.append(int_vehicle_phase_group)
        pedestrian_phase_abcd = self.int_crosswalk_phase_groups[int_action]
        # ArBCrD
        pedestrian_phase_string = pedestrian_phase_abcd['A'] + 'r' + pedestrian_phase_abcd['B'] + pedestrian_phase_abcd['C'] + 'r' + pedestrian_phase_abcd['D']
        int_state = vehicle_phase_string + pedestrian_phase_string
        traci.trafficlight.setRedYellowGreenState(self.tl_ids[0], int_state)
        
        # Midblock
        for i in range(1, len(self.tl_ids)):
            tl_id = self.tl_ids[i]
            mb_switch_state = switch_state[i]

            if mb_switch_state == 1:
                if current_action_step < 4:
                    mb_action = 2
                elif current_action_step == 4:
                    mb_action = 3
                elif current_action_step >= 5:
                    mb_action = 0
            else:
                mb_action = action[i]
            
            current_phase.append(mb_action)
            mb_phase_string = self.midblock_phase_groups[mb_action]

            if tl_id == "cluster_9740157181_9740483933": # For this TL, add 4 small 'g's after the first letter
                mb_phase_string = mb_phase_string[0] + "g"*5 + mb_phase_string[1:]

            #print(f"\nSetting phase: {mb_phase_string} for Midblock TL: {tl_id}\n")
            traci.trafficlight.setRedYellowGreenState(tl_id, mb_phase_string)

        return current_phase
    
    def _get_design_reward(self, num_proposals):
        """
        Every iteration, only to be called once at the end of the lower agent's episode.

        Design reward:
        - Design reward is obtained at the end of the lower agent's episode. i.e., it is obtained from each parallel worker.
        - Pedestrians: 
            - How much time "on average per pedestrian" did it take to "arrive" at the crossing
            - The "on average per pedestrian" part normalizes the demand scaling 
            - Due to SUMO's routing logic, pedestrians will always take the shortest path (to arrive at the nearest crosswalk).
            - This will be averaged across all parallel workers in the main process. 
        - Vehicles: 
            - Vehicle delay is strongly correlated with the number of crosswalks (proposals)
            - i.e., num_proposals itself is a proxy for delay
            - Penalize high num_proposals. Which is also not dependent on the demand scaling.

        - General: 
            - TODO: If a proposed crosswalk is not used (does not get foot traffic), penalize. (This only makes sense in case of deterministic sampling)
            - TODO: Penalty for pedestrian jams. Be careful: this jam has to be caused by design, not control.
        """
        design_reward = 0

        # Pedestrian
        total_ped_arrival_time = sum(self.pedestrian_arrival_times.values())
        average_arrival_time_per_ped = total_ped_arrival_time / (len(self.pedestrian_arrival_times) + 1e-6) # Avoid division by zero (due to some bad-faith design)
        design_reward -= average_arrival_time_per_ped
        print(f"Total pedestrian arrival time: {total_ped_arrival_time}")
        print(f"Average arrival time per pedestrian: {average_arrival_time_per_ped}") # Generally in the range of 100 seconds.
        
        # Vehicle
        design_reward -= num_proposals*2.0 # Maximum number of proposals is 10. So -20.

        # General 
        
        design_reward = np.clip(design_reward, -200, 200)
        return design_reward
    
    def _get_control_reward(self, corrected_occupancy_map, switch_state, pressure_dict=None):
        """ 
        wrapper
        """
        return self._get_mwaq_reward_exponential(corrected_occupancy_map, switch_state, print_reward=False)

    def _get_mwaq_reward(self, corrected_occupancy_map, switch_state, print_reward=False):
        """
        * Maximum wait aggregated queue (MWAQ)
        * MWAQ = For each TL, For both veh and ped: [sum of queue lengths of all incoming directions x maximum waiting time among all]
        * Penalize high MWAQ, i.e. make it negative.
        * Individually normalize MWAQ for each TL.
        Other components (Optional): 
            - Penalize frequent changes of action based on switch_state 
        Vehicles:
        - getWaitingTime: The waiting time (in seconds) of a vehicle spent with speed below 0.1 m/s. Reset to 0 every time it moves.
        - getAccumulatedWaitingTime: The accumulated waiting time (in seconds) of a vehicle over a certain time interval (interval length is set with --waiting-time-memory)
        Pedestrians:
        - getWaitingTime: The waiting time (in seconds) of a pedestrian spent with speed below 0.1 m/s. Reset to 0 every time it moves.

        Normalizers:
        - A fixed normalizer multiplied by the number of incoming directions

        # TODO: Should we consider vicinity for pedestrians as well?
        """
        MWAQ_VEH_NORMALIZER = 100
        MWAQ_PED_NORMALIZER = 100
        VEH_THRESHOLD_SPEED = 0.2 # m/s
        PED_THRESHOLD_SPEED = 0.5 # m/s

        # Intersection 
        # Vehicle
        int_veh_mwaq = 0
        # queue length only starts counting if the vehicles are below 0.1 m/s
        # set thsi to 0.5 so that it does not have a zero value if vehicles  moving but very slow (upto the threshold speed)
        max_wait_time_veh_int = 0.5 
        veh_queue_length = 0

        for direction_turn in self.direction_turn_intersection_incoming:
            int_vehicles = corrected_occupancy_map["cluster_172228464_482708521_9687148201_9687148202_#5more"]["vehicle"]["incoming"][direction_turn]
            for veh_id in int_vehicles:
                # queue length
                if traci.vehicle.getSpeed(veh_id) < VEH_THRESHOLD_SPEED:
                    veh_queue_length += 1
                # wait time
                wait_time = traci.vehicle.getWaitingTime(veh_id)
                if wait_time > max_wait_time_veh_int:
                    max_wait_time_veh_int = wait_time

        int_veh_mwaq = veh_queue_length * max_wait_time_veh_int
        norm_int_veh_mwaq = int_veh_mwaq / (MWAQ_VEH_NORMALIZER * len(self.direction_turn_intersection_incoming))

        # Pedestrian
        int_ped_mwaq = 0
        max_wait_time_ped_int = 0.5
        ped_queue_length = 0
        for direction in self.directions:
            int_pedestrians = corrected_occupancy_map["cluster_172228464_482708521_9687148201_9687148202_#5more"]["pedestrian"]["incoming"][direction]["main"]
            for ped_id in int_pedestrians:
                # queue length
                if traci.person.getSpeed(ped_id) < PED_THRESHOLD_SPEED:
                    ped_queue_length += 1
                # wait time
                wait_time = traci.person.getWaitingTime(ped_id)
                if wait_time > max_wait_time_ped_int:

                    max_wait_time_ped_int = wait_time
        int_ped_mwaq = ped_queue_length * max_wait_time_ped_int
        norm_int_ped_mwaq = int_ped_mwaq / (MWAQ_PED_NORMALIZER * len(self.directions))

        # Midblock
        # Vehicle
        norm_mb_veh_mwaq = 0
        norm_mb_veh_mwaq_per_tl = {}
        for tl_id in self.tl_ids[1:]:
            tl_veh_mwaq = 0
            max_wait_time_veh_mb = 0.5
            veh_queue_length = 0
            for direction in self.direction_turn_midblock:
                mb_vehicles = corrected_occupancy_map[tl_id]["vehicle"]["incoming"][direction]

                # queue length
                for veh_id in mb_vehicles:
                    if traci.vehicle.getSpeed(veh_id) < VEH_THRESHOLD_SPEED:
                        veh_queue_length += 1

                    # wait time
                    wait_time = traci.vehicle.getWaitingTime(veh_id)
                    if wait_time > max_wait_time_veh_mb:
                        max_wait_time_veh_mb = wait_time

            tl_veh_mwaq = veh_queue_length * max_wait_time_veh_mb
            norm_tl_veh_mwaq = tl_veh_mwaq / (MWAQ_VEH_NORMALIZER * len(self.direction_turn_midblock))
            norm_mb_veh_mwaq_per_tl[tl_id] = norm_tl_veh_mwaq
            norm_mb_veh_mwaq += norm_tl_veh_mwaq
        norm_mb_veh_mwaq = norm_mb_veh_mwaq / len(self.tl_ids[1:])


        # Pedestrian    
        norm_mb_ped_mwaq = 0
        norm_mb_ped_mwaq_per_tl = {}
        for tl_id in self.tl_ids[1:]:
            tl_ped_mwaq = 0
            max_wait_time_ped_mb = 0.5
            ped_queue_length = 0
            mb_pedestrians = corrected_occupancy_map[tl_id]["pedestrian"]["incoming"]["north"]["main"] # only one direction # 
            for ped_id in mb_pedestrians:
                # queue length
                if traci.person.getSpeed(ped_id) < PED_THRESHOLD_SPEED:
                    ped_queue_length += 1
                # wait time
                wait_time = traci.person.getWaitingTime(ped_id)
                if wait_time > max_wait_time_ped_mb:

                    max_wait_time_ped_mb = wait_time

            tl_ped_mwaq = ped_queue_length * max_wait_time_ped_mb
            norm_tl_ped_mwaq = tl_ped_mwaq / MWAQ_PED_NORMALIZER
            norm_mb_ped_mwaq_per_tl[tl_id] = norm_tl_ped_mwaq
            norm_mb_ped_mwaq += norm_tl_ped_mwaq
        norm_mb_ped_mwaq = norm_mb_ped_mwaq / len(self.tl_ids[1:])  

        # Frequency of switch state changes
        # norm_switch_penalty = sum(switch_state) / len(self.tl_ids)
        
        reward = -1 * (norm_int_veh_mwaq + norm_int_ped_mwaq + norm_mb_veh_mwaq + norm_mb_ped_mwaq)

        if print_reward:
            print(f"Intersection Reward Components:\n"
                  f"\tVehicle MWAQ: {norm_int_veh_mwaq}\n"
                  f"\tPedestrian MWAQ: {norm_int_ped_mwaq}")

            for tl_id in self.tl_ids[1:]:
                print(f"Midblock TL {tl_id} Reward Components:\n"
                      f"\tVehicle MWAQ: {norm_mb_veh_mwaq_per_tl[tl_id]}\n"
                      f"\tPedestrian MWAQ: {norm_mb_ped_mwaq_per_tl[tl_id]}")
            # print(f"Switch penalty: {norm_switch_penalty}")
            print(f"Total Reward: {reward}\n\n")

        return reward

    def _get_mwaq_reward_exponential(self, corrected_occupancy_map, switch_state, print_reward=False):
        """
        - Exponentially Increasing Normalized Maximum wait aggregated queue (EI-MWAQ) 
        * MWAQ = For each TL, For both veh and ped: [sum of queue lengths of all incoming directions x maximum waiting time among all]
        * Penalize high MWAQ, i.e. make it negative.
        * Individually normalize MWAQ for each TL.
        Other components (Optional): 
            - Penalize frequent changes of action based on switch_state 

        Vehicles:
        - getWaitingTime: The waiting time (in seconds) of a vehicle spent with speed below 0.1 m/s. Reset to 0 every time it moves.
        - getAccumulatedWaitingTime: The accumulated waiting time (in seconds) of a vehicle over a certain time interval (interval length is set with --waiting-time-memory)
        Pedestrians:
        - getWaitingTime: The waiting time (in seconds) of a pedestrian spent with speed below 0.1 m/s. Reset to 0 every time it moves.

        Normalizers:
        - A fixed normalizer multiplied by the number of incoming directions

        # TODO: Should we consider vicinity for pedestrians as well?
        """
        MWAQ_VEH_NORMALIZER = 2.0 # Number of pedestrians per crosswalk increases as the number of crosswalks is reduced by design agent. But it remains same for vehicles.
        MWAQ_PED_NORMALIZER = 10.0
        VEH_THRESHOLD_SPEED = 0.2 # m/s
        PED_THRESHOLD_SPEED = 0.5 # m/s # 0.1 is the threshold in SUMO by default (i.e. wait time is counted when speed is below 0.1 m/s)

        # Intersection 
        # Vehicle
        int_veh_mwaq = 0
        max_wait_time_veh_int = 0.5
        veh_queue_length = 0

        for direction_turn in self.direction_turn_intersection_incoming:
            int_vehicles = corrected_occupancy_map["cluster_172228464_482708521_9687148201_9687148202_#5more"]["vehicle"]["incoming"][direction_turn]
            for veh_id in int_vehicles:
                # queue length
                if traci.vehicle.getSpeed(veh_id) < VEH_THRESHOLD_SPEED:
                    veh_queue_length += 1
                # wait time
                wait_time = traci.vehicle.getWaitingTime(veh_id)
                if wait_time > max_wait_time_veh_int:
                    max_wait_time_veh_int = wait_time

        int_veh_mwaq = veh_queue_length * max_wait_time_veh_int
        # In practice, using len(self.direction_turn_intersection_incoming) (with value 12) in the division, is too harsh, and will lead to pedestrians getting extreme favor.
        # Only 7 incoming directions are physically present.
        norm_int_veh_mwaq = int_veh_mwaq / (MWAQ_VEH_NORMALIZER * len(self.directions))

        # Pedestrian
        int_ped_mwaq = 0
        max_wait_time_ped_int = 0.5
        ped_queue_length = 0
        for direction in self.directions:
            int_pedestrians = corrected_occupancy_map["cluster_172228464_482708521_9687148201_9687148202_#5more"]["pedestrian"]["incoming"][direction]["main"]
            for ped_id in int_pedestrians:
                # queue length
                if traci.person.getSpeed(ped_id) < PED_THRESHOLD_SPEED:
                    ped_queue_length += 1
                # wait time
                wait_time = traci.person.getWaitingTime(ped_id)
                if wait_time > max_wait_time_ped_int:

                    max_wait_time_ped_int = wait_time
        int_ped_mwaq = ped_queue_length * max_wait_time_ped_int
        norm_int_ped_mwaq = int_ped_mwaq / (MWAQ_PED_NORMALIZER * len(self.directions))

        # Midblock
        # Vehicle
        norm_mb_veh_mwaq_per_tl = {}
        for tl_id in self.tl_ids[1:]:
            tl_veh_mwaq = 0
            max_wait_time_veh_mb = 0.5
            veh_queue_length = 0
            for direction in self.direction_turn_midblock:
                mb_vehicles = corrected_occupancy_map[tl_id]["vehicle"]["incoming"][direction]

                # queue length
                for veh_id in mb_vehicles:
                    if traci.vehicle.getSpeed(veh_id) < VEH_THRESHOLD_SPEED:
                        veh_queue_length += 1

                    # wait time
                    wait_time = traci.vehicle.getWaitingTime(veh_id)
                    if wait_time > max_wait_time_veh_mb:
                        max_wait_time_veh_mb = wait_time

            tl_veh_mwaq = veh_queue_length * max_wait_time_veh_mb
            norm_tl_veh_mwaq = tl_veh_mwaq / (MWAQ_VEH_NORMALIZER * len(self.direction_turn_midblock))
            norm_mb_veh_mwaq_per_tl[tl_id] = norm_tl_veh_mwaq

        # Pedestrian    
        norm_mb_ped_mwaq_per_tl = {}
        for tl_id in self.tl_ids[1:]:
            tl_ped_mwaq = 0
            max_wait_time_ped_mb = 0.5
            ped_queue_length = 0
            mb_pedestrians = corrected_occupancy_map[tl_id]["pedestrian"]["incoming"]["north"]["main"] # only one direction # 
            for ped_id in mb_pedestrians:
                # queue length
                if traci.person.getSpeed(ped_id) < PED_THRESHOLD_SPEED:
                    ped_queue_length += 1
                # wait time
                wait_time = traci.person.getWaitingTime(ped_id)
                if wait_time > max_wait_time_ped_mb:
                    max_wait_time_ped_mb = wait_time

            tl_ped_mwaq = ped_queue_length * max_wait_time_ped_mb
            norm_tl_ped_mwaq = tl_ped_mwaq / MWAQ_PED_NORMALIZER
            norm_mb_ped_mwaq_per_tl[tl_id] = norm_tl_ped_mwaq

        # Frequency of switch state changes
        # norm_switch_penalty = sum(switch_state) / len(self.tl_ids)
        
        alpha = 0.5
        # Final reward calculation
        # For the intersection, exponent the current normalized value (for both veh and ped)
        final_int_veh = np.exp(alpha * norm_int_veh_mwaq) # Alpha value of 0.5 to reduce sensitivity 
        final_int_ped = np.exp(alpha * norm_int_ped_mwaq) 

        # For the midblock, compute an L2 norm over the vector of normalized values for each TL, then exponentiate (for both veh and ped)
        norm_mb_veh_l2 = np.linalg.norm(np.array(list(norm_mb_veh_mwaq_per_tl.values())))
        norm_mb_ped_l2 = np.linalg.norm(np.array(list(norm_mb_ped_mwaq_per_tl.values())))

        # TODO: For Midblock: Can further, effectively "normalize" the L2 aggregation. By dividing ny sqrt(N). Reduce the effect of outliers.
        # Perhaps preserving outliers is beneficial.
        final_mb_veh = np.exp(alpha * norm_mb_veh_l2)
        final_mb_ped = np.exp(alpha * norm_mb_ped_l2)

        # Final reward is the negative sum of the four exponentiated values
        reward = -1 * (final_int_veh + final_int_ped + final_mb_veh + final_mb_ped)

        # Clip the reward (In an appropriately chosen range) before returning. 
        clipped_reward = np.clip(reward, -2500, 2500) # IMPORTANT: If alpha is changed, this range needs to be changed as well.

        # if print_reward:
        #     print(f"Intersection Reward Components:\n"
        #         f"\tVehicle MWAQ: {norm_int_veh_mwaq} (exp: {final_int_veh})\n"
        #         f"\tPedestrian MWAQ: {norm_int_ped_mwaq} (exp: {final_int_ped})")

        #     for tl_id in self.tl_ids[1:]:
        #         print(f"Midblock TL {tl_id} Reward Components:\n"
        #             f"\tVehicle MWAQ: {norm_mb_veh_mwaq_per_tl[tl_id]}\n"
        #             f"\tPedestrian MWAQ: {norm_mb_ped_mwaq_per_tl[tl_id]}")
        #     print(f"Midblock Overall L2 Norms:\n"
        #         f"\tVehicle L2 Norm: {norm_mb_veh_l2} (exp: {final_mb_veh})\n"
        #         f"\tPedestrian L2 Norm: {norm_mb_ped_l2} (exp: {final_mb_ped})")
        #     # print(f"Switch penalty: {norm_switch_penalty}")
        #     print(f"Total Reward: {reward}\n\n")
        #     print(f"Clipped Reward: {clipped_reward}\n\n")

        return clipped_reward

    def _count_near_conflicts(self, corrected_occupancy_map, threshold_speed=0.8, distance_threshold=6.0):
        """
        Count potential vehicle-pedestrian conflicts at signalized/ unsignalized crosswalks.
        Only meaningful at high volumes. 

        A "near-conflict" represents a safety-critical event where a vehicle must yield
        to a pedestrian at a crosswalk. Each unique vehicle-pedestrian interaction is
        counted only once, even if it spans multiple timesteps.

        Features:
        - Tracks unique conflicts using vehicle+pedestrian+location identifiers
        - Only counts each unique interaction once during the simulation
        - Records yielding behavior as a proxy for potential safety incidents
        - Focuses specifically on mid-block crosswalks where conflicts are most likely

        Parameters:
        - threshold_speed: Speed below which a vehicle is considered to be yielding (m/s)
        - distance_threshold: Maximum distance between vehicle and pedestrian to qualify as conflict (m)

        Returns:
        - Number of new unique conflicts detected in the current timestep
        """
        new_conflicts = 0

        # For each midblock location
        for tl_id in self.tl_ids[1:]:  # Skip intersection, focus on midblocks
            # Get pedestrians at this crosswalk (incoming or outgoing. Most likely outgoing)
            pedestrians = corrected_occupancy_map[tl_id]["pedestrian"]["incoming"]["north"]["main"] + corrected_occupancy_map[tl_id]["pedestrian"]["outgoing"]["north"]["main"]

            if not pedestrians:  # Skip if no pedestrians present
                continue

            # Check vehicles approaching this location
            for direction in self.direction_turn_midblock:
                vehicles = corrected_occupancy_map[tl_id]["vehicle"]["incoming"][direction]

                for veh_id in vehicles:
                    # Current vehicle speed
                    current_speed = traci.vehicle.getSpeed(veh_id)

                    # If vehicle speed is still high
                    if current_speed > threshold_speed:
                        veh_pos = traci.vehicle.getPosition(veh_id)

                        # Check proximity to pedestrians
                        for ped_id in pedestrians:
                            ped_pos = traci.person.getPosition(ped_id)
                            distance = math.sqrt((veh_pos[0]-ped_pos[0])**2 +
                                            (veh_pos[1]-ped_pos[1])**2)

                            if distance < distance_threshold:
                                # Create a unique conflict identifier
                                conflict_id = f"{veh_id}_{ped_id}_{tl_id}"

                                # If this is a new conflict we haven't seen before
                                if conflict_id not in self.recorded_conflicts:
                                    self.recorded_conflicts.add(conflict_id)
                                    new_conflicts += 1

        return new_conflicts

    def _check_done(self):
        """
        TODO: What more conditions can be added here?
        - Gridlock? Jam? of vehicles or pedestrians? Crashes?
        """
        return self.step_count >= self.max_timesteps

    def _set_signal_slots(self, new_tl_ids, signal_slots=None):
        if signal_slots is None:
            ordered = sorted(new_tl_ids, key=lambda tid: (traci.junction.getPosition(tid)[0], tid))
            signal_slots = {tid: slot for slot, tid in enumerate(ordered)}
        if set(signal_slots) != set(new_tl_ids):
            raise ValueError("Signal slots must cover exactly the current crossing IDs.")
        slots = list(signal_slots.values())
        if any(not isinstance(slot, (int, np.integer)) or slot < 0 or slot >= self.max_proposals for slot in slots):
            raise ValueError("Signal slots must be integer indices within max_proposals.")
        if len(set(slots)) != len(slots):
            raise ValueError("Signal slots must be unique.")
        ordered = sorted(new_tl_ids, key=signal_slots.__getitem__)
        self.tl_ids = self.tl_ids[:1] + ordered
        self.active_slots = np.asarray([signal_slots[tid] for tid in ordered], dtype=np.int64)

    def reset(self, extreme_edge_dict, num_proposals, tl=False, real_world=False, eval_mode=False, signal_slots=None):
        """
        """
        # useful when running multiple iterations of the same env (as in eval)
        if self.sumo_running:
            self.close()

        self.tl_ids = ['cluster_172228464_482708521_9687148201_9687148202_#5more']
        self.tl_lane_dict = get_related_lanes_edges()
        self.tl_pedestrian_status.clear()
        self.pedestrian_existence_times.clear()
        self.pedestrian_arrival_times.clear()
        self.total_unique_ids_veh.clear()
        self.total_unique_ids_ped.clear()
        self.recorded_conflicts.clear()
        self.previous_action = None
        self.signal_executor = None
        self.demand_windows = {}

        window_size = self.max_timesteps * self.step_length + self.warmup_steps[1]

        if self.manual_demand_veh is not None : 
            #scaling = convert_demand_to_scale_factor(self.manual_demand_veh, "vehicle", self.vehicle_input_trips) # Convert the demand to scaling factor first
            # scale_demand(self.vehicle_input_trips, self.vehicle_output_trips, self.manual_demand_veh, demand_type="vehicle") # directly scaling factor given
            self.demand_windows['vehicle'] = scale_demand_sliced_window(self.vehicle_input_trips, self.vehicle_output_trips, self.manual_demand_veh, demand_type="vehicle", window_size=window_size, evaluation=eval_mode)
        else: 
            # Automatically scale demand 
            scale_factor_vehicle = random.uniform(self.demand_scale_min, self.demand_scale_max)
            # scale_demand(self.vehicle_input_trips, self.vehicle_output_trips, scale_factor_vehicle, demand_type="vehicle")
            self.demand_windows['vehicle'] = scale_demand_sliced_window(self.vehicle_input_trips, self.vehicle_output_trips, scale_factor_vehicle, demand_type="vehicle", window_size=window_size, evaluation=eval_mode)

        if self.manual_demand_ped is not None:
            # scaling = convert_demand_to_scale_factor(self.manual_demand_ped, "pedestrian", self.pedestrian_input_trips)
            # scale_demand(self.pedestrian_input_trips, self.pedestrian_output_trips, self.manual_demand_ped, demand_type="pedestrian") # directly scaling factor given
            self.demand_windows['pedestrian'] = scale_demand_sliced_window(self.pedestrian_input_trips, self.pedestrian_output_trips, self.manual_demand_ped, demand_type="pedestrian", window_size=window_size, evaluation=eval_mode)
        else: 
            scale_factor_pedestrian = random.uniform(self.demand_scale_min, self.demand_scale_max)
            # scale_demand(self.pedestrian_input_trips, self.pedestrian_output_trips, scale_factor_pedestrian, demand_type="pedestrian")
            self.demand_windows['pedestrian'] = scale_demand_sliced_window(self.pedestrian_input_trips, self.pedestrian_output_trips, scale_factor_pedestrian, demand_type="pedestrian", window_size=window_size, evaluation=eval_mode)


        # Workers share read-only geometry; each keeps its own SUMO logs.
        sumo_cmd = [os.path.join(sumo.SUMO_HOME, "bin", "sumo-gui" if self.use_gui else "sumo"),
                    "--quit-on-end",
                    "--net-file", f"{self.run_dir}/network_iterations/network_iteration_{self.network_iteration}.net.xml",
                    "--log", f"{self.run_dir}/sumo_logfile{self.traci_label}.txt",
                    "--error-log", f"{self.run_dir}/sumo_errorlog{self.traci_label}.txt",
                    "--step-length", str(self.step_length),
                    "--route-files", f"{self.vehicle_output_trips},{self.pedestrian_output_trips}"]
        if self.sumo_seed is not None:
            sumo_cmd.extend(["--seed", str(self.sumo_seed)])
        if self.auto_start:
            sumo_cmd.append("--start")
        max_retries = 1 if self.signal_control_protocol == SHARED_SIGNAL_PROTOCOL else 3
        try:
            for attempt in range(max_retries):
                try:
                    traci.start(sumo_cmd, label=self.traci_label)
                    break
                except traci.exceptions.FatalTraCIError:
                    if attempt < max_retries - 1:
                        print(f"TraCI connection failed. Retrying... (Attempt {attempt + 1}/{max_retries})")
                        time.sleep(3)
                    else:
                        print(f"Failed to start TraCI after {max_retries} attempts.")
                        raise
        except Exception as e:
            print(f"An unexpected error occurred: {str(e)}")
            raise

        self.sumo_running = True
        self.step_count = 0 # This counts the timesteps in an episode. Needs reset.
        
        # Get the actual traffic light IDs from the current network (add new ones)
        new_tl_ids = [tl_id for tl_id in traci.trafficlight.getIDList() if tl_id not in self.tl_ids]

        if num_proposals != len(new_tl_ids):
            raise ValueError(f"Number of proposals ({num_proposals}) does not match the number of traffic light IDs ({len(new_tl_ids)}, {new_tl_ids})")
        
        self._set_signal_slots(new_tl_ids, signal_slots)
        self.junction_pos_cache = {jid: traci.junction.getPosition(jid) for jid in self.tl_ids}

        self.num_proposals = num_proposals
        self.dynamically_populate_edges_lanes(extreme_edge_dict, real_world)

        temp_mb_ped_incoming_edges_all = [self.tl_lane_dict[tl_id]['pedestrian']['incoming']['north']['main'] \
                                          for tl_id in self.tl_ids if tl_id != 'cluster_172228464_482708521_9687148201_9687148202_#5more']
        self.mb_ped_incoming_edges_all = [edge for edges in temp_mb_ped_incoming_edges_all for edge in edges]

        # Warmup period
        # How many actions to take during warmup
        warmup = random.randint(self.warmup_steps[0], self.warmup_steps[1])
        num_actions_warmup = int(warmup // self.action_duration)
        #print(f"Number of actions during warmup: {num_actions_warmup}")
        observation_buffer = []
        for i in range(num_actions_warmup):

            # Randomly sample actions (1 digit for intersection, the rest of the bits for mid-block crosswalks)
            action = np.concatenate([np.random.randint(4, size=1), np.random.randint(2, size= len(self.tl_ids) - 1)]).astype(np.int32)
            #print(f"\nWarmup action {i}: {action}\n")
            if i==0:
                prev_action = action
            switch_state, _ = self._detect_switch(action, prev_action)

            # TODO: Inefficient to do it this way.
            if tl:
                for j in range(self.steps_per_action):
                    current_phase = [1]*len(self.tl_ids) # random phase
                    traci.simulationStep() 
                    self._update_pedestrian_existence_times()
                    obs = self._get_observation(current_phase)
                    # _ = self._get_pressure_dict(self.corrected_occupancy_map)
                    observation_buffer.append(obs)
                    self._get_pedestrian_arrival_times()
                    # No reward calculation
                    # self.step_count += 1 # We are not counting the warmup steps in the total simulation steps
            else: 
                for j in range(self.steps_per_action):
                    current_phase = self._apply_action(action, j, switch_state)
                    traci.simulationStep() 
                    self._update_pedestrian_existence_times()
                    obs = self._get_observation(current_phase)
                    # _ = self._get_pressure_dict(self.corrected_occupancy_map)
                    observation_buffer.append(obs)
                    self._get_pedestrian_arrival_times()
                    # No reward calculation
                    # self.step_count += 1 # We are not counting the warmup steps in the total simulation steps

            prev_action = action

        if not tl:
            self.previous_action = prev_action

        # print(f"\n{warmup} steps of warmup ended.")
        observation_buffer = observation_buffer[-self.steps_per_action:] # Only keep the observation of thelast action
        observation = np.asarray(observation_buffer, dtype=np.float32)
        #print(f"\nObservation (in reset): {observation.shape}")

        # reset the waiting times
        self.prev_vehicle_waiting_time = {vid: traci.vehicle.getWaitingTime(vid)
                                          for vid in traci.vehicle.getIDList()}
        self.prev_ped_waiting_time = {pid: traci.person.getWaitingTime(pid)
                                    for pid in traci.person.getIDList()}
        self.total_conflicts = 0 # Running total of unique vehicle-pedestrian conflicts
        self.total_switches = 0 # Only applicable for RL.

        return observation, {} # info is empty

    # Old mechanism for vehicle occupancy
    # def _straight_links(self, ctrl_links):
    #     west_in = east_in = west_out = east_out = None

    #     for phase in ctrl_links:
    #         for frm, to, via in phase:
    #             if via == "" or not (frm.endswith("_0") and to.endswith("_0")):
    #                 continue                        # skip ped rows & turns

    #             if frm.startswith("-"):            # vehicle coming from WEST
    #                 west_in  = frm
    #                 west_out = to                  # <-- keep WEST tag
    #             else:                              # coming from EAST
    #                 east_in  = frm
    #                 east_out = to                  # <-- keep EAST tag

    #     if None in (west_in, east_in, west_out, east_out):
    #         raise RuntimeError("Could not find all four straight vehicle links")

    #     return west_in, east_in, west_out, east_out
    
    def _get_vehicle_occupancy_midblock(self, vehicle_ids, vehicle_pos_map):
        """
        A new mechanism for midblock TLs ("incoming" and "outgoing"). 
        Get the vehicle occupancy for all midblock TLs.
        For the ones below, we are using the old mechanism.
         - pedestrians in all TLs
         - "inside" for vehicles in midblock TLs
         - vehicles in intersection
        Enforces the cutoff distance.
        The mechanism also allows for same vehicles to be detected by multiple TLs.
        Duplication of vehicles between is corrected later.
        """
        midblock_tl_ids = self.tl_ids[1:]
        vehicle_occupancy_midblock = {
            tl_id: {
                "incoming": {"west-straight": [], "east-straight": []},
                "outgoing": {"west-straight": [], "east-straight": []}
            }
            for tl_id in midblock_tl_ids
        }

        # Angle 90 is East-bound, 0 North-bound, 270 West-bound, 180 South-bound.
        angle_tolerance = 40

        for veh_id in vehicle_ids:
            veh_pos = vehicle_pos_map.get(veh_id)

            if veh_pos is None:
                continue

            try: # Vehicle might have just left or not fully entered
                angle = traci.vehicle.getAngle(veh_id) 
            except traci.TraCIException:
                continue

            direction = None 
            if angle >= (90 - angle_tolerance) and angle <= (90 + angle_tolerance):
                direction = "east-straight"
            elif angle >= (270 - angle_tolerance) and angle <= (270 + angle_tolerance):
                direction = "west-straight"
            
            if direction is None: # If not clearly East or Westbound, skip (likely turning or vertical)
                continue
            
            veh_pos_x, veh_pos_y = veh_pos[0], veh_pos[1]
            for tl_id in midblock_tl_ids:
                junction_pos = self.junction_pos_cache.get(tl_id)
                distance = self._get_vehicle_distance_cached(junction_pos, veh_pos)
                if distance >= self.midblock_cutoff_distance:
                    continue
                    
                # get the incoming or outgoing direction based on x-coordinate
                tl_pos_x = junction_pos[0]
                # Assign based on relative position and direction
                if direction == "east-straight": # Moving East
                    if veh_pos_x < tl_pos_x: # Vehicle is West of TL
                        vehicle_occupancy_midblock[tl_id]["incoming"]["east-straight"].append(veh_id)
                    elif veh_pos_x > tl_pos_x: # Vehicle is East of TL
                        vehicle_occupancy_midblock[tl_id]["outgoing"]["east-straight"].append(veh_id)
                elif direction == "west-straight": 
                    if veh_pos_x > tl_pos_x: # Vehicle is East of TL
                        vehicle_occupancy_midblock[tl_id]["incoming"]["west-straight"].append(veh_id)
                    elif veh_pos_x < tl_pos_x: # Vehicle is West of TL
                        vehicle_occupancy_midblock[tl_id]["outgoing"]["west-straight"].append(veh_id)

        return vehicle_occupancy_midblock
    
    # Old mechanism for vehicle occupancy
    # helper – returns True if the lane is “verticalish”
    # def _is_vertical(self, lane, thresh=30):                  # thresh = degrees from 90°
    #     x0, y0 = lane.getShape()[0]
    #     x1, y1 = lane.getShape()[-1]
    #     dx, dy = x1 - x0, y1 - y0
    #     if dx == dy == 0:                               # zero‑length, ignore
    #         return True
    #     angle = abs(math.degrees(math.atan2(dy, dx)))   #   0° = east, 90° = north
    #     angle = min(angle, 180 - angle)                 # mirror so 90° is “up/down”
    #     return angle >= 90 - thresh                     # within thresh of vertical

    def _internal_edges(self, via, outgoing):
        """Follow a movement across internal waiting junctions to its exit lane."""
        while via:
            yield traci.lane.getEdgeID(via)
            via = next(link[4] for link in traci.lane.getLinks(via, extended=True)
                       if link[0] == outgoing)

    def dynamically_populate_edges_lanes(self, extreme_edge_dict, real_world=False):
        """
        Get midblock TLs lanes/edges for occupancy map to use. i.e., update self.tl_lane_dict with latest network iteration.
        - For vehicle + inside, follow the controlled connections' actual internal edges.
        - For vehicle  + incoming or outgoing, use controlled_links but need to add more until either total length is equal to cut off distance or we encounter another TL. 
        - For pedestrians + outgoing, use the mid-block TL id with _c0 
        - For pedestrians + incoming, use the mid-block TL id with _w0 and _w1 (there will be more.) 
        - For vehicles, in both incoming and outgoing, west-straight will have a negative sign in edge id. east-straight will have a positive sign.
        - Sometimes the returned controlled links are empty. Because for pedestrians, we are using a single direction, the other direction is not present.
        """
    
        # For each midblock traffic light (everything except the intersection)
        for tl_id in self.tl_ids:
            if tl_id == 'cluster_172228464_482708521_9687148201_9687148202_#5more':
                if extreme_edge_dict['rightmost']['new'] is not None: # Just check one (if it has been split)
                    self.tl_lane_dict[tl_id]["vehicle"]["incoming"]["east-straight"] = [f"-{extreme_edge_dict['leftmost']['new']}_0"]
                    # counterpart
                    self.tl_lane_dict[tl_id]["vehicle"]["outgoing"]["east"] = [f"{extreme_edge_dict['leftmost']['new']}_0"]
                controlled_links = traci.trafficlight.getControlledLinks(tl_id)
                inside = self.tl_lane_dict[tl_id]["vehicle"]["inside"]
                for offset, direction in enumerate(("east", "south", "west", "north")):
                    for index, turn in enumerate(("right", "straight", "left")):
                        inside[f"{direction}-{turn}"] = [
                            f"edge.{edge}" for _, outgoing, via in controlled_links[4 * offset + index]
                            for edge in self._internal_edges(via, outgoing)]
            
            # Skip if the traffic light ID already exists in the dictionary
            if tl_id in self.tl_lane_dict:
                continue
            else: 
                inside_edges = {"west-straight": [], "east-straight": []}
                for links in traci.trafficlight.getControlledLinks(tl_id):
                    for incoming, outgoing, via in links:
                        if not via or incoming.startswith(':') or incoming.startswith('-') != outgoing.startswith('-'):
                            continue
                        direction = "west-straight" if incoming.startswith('-') else "east-straight"
                        inside_edges[direction].extend(self._internal_edges(via, outgoing))

                tl_internal = f":{tl_id}"
                self.tl_lane_dict[tl_id] = {
                    "vehicle": {
                        # Old mechanism for vehicle occupancy
                        # "incoming": {
                        #     "west-straight": [west_in],
                        #     "east-straight": [east_in]
                        # },
                        # "outgoing": {
                        #     "west-straight": [west_out],
                        #     "east-straight": [east_out]
                        # }
                        "inside": inside_edges,
                        
                    },
                    "pedestrian": {
                        "incoming": {
                            "north": {
                                "main": [f"{tl_internal}_w0", f"{tl_internal}_w1"]
                            }
                        },
                        "outgoing": {
                            "north": {
                                "main": [f"{tl_internal}_c0"]
                            }
                        }
                    }
                }
                
                # Old mechanism for vehicle occupancy
                # TODO: For ONLY vehicle incoming + outgoing, keep adding until the total length becomes just higher than cutoff distance or we encounter another TL. DONE
                # Do this for east-straight (only for outgoing) of the intersection as well.
                # for group in ["incoming", "outgoing"]:
                #     for dir_key in ["west-straight", "east-straight"]:

                #         expanded   = self.tl_lane_dict[tl_id]["vehicle"][group][dir_key].copy()
                #         seed_sign  = expanded[0].split('_')[0].startswith('-')   # True if '‑'
                #         total_dist = self.net.getLane(expanded[-1]).getLength()
                #         visited    = set(expanded)


                #         while total_dist < self.midblock_cutoff_distance:
                #             last_lane_obj = self.net.getLane(expanded[-1])
                #             successors    = (last_lane_obj.getIncoming()
                #                             if group == "incoming"
                #                             else last_lane_obj.getOutgoingLanes())

                #             next_lane_id  = None
                #             for ln in successors:
                #                 lid = ln.getID()

                #                 # 1) skip lanes we have already taken
                #                 # 2) skip “verticalish” lanes
                #                 # 3) keep the original sign (− for westbound, + for eastbound)
                #                 if (lid in visited or
                #                     self._is_vertical(ln) or
                #                     lid.split('_')[0].startswith('-') != seed_sign):
                #                     continue

                #                 next_lane_id = lid
                #                 break          # only take **one** successor each step

                #             # No suitable successor found → stop expanding
                #             if next_lane_id is None or "iter" in next_lane_id:
                #                 break

                #             # -------------------------------------------------------------
                #             # If the successor’s *from* (or *to*) node is a **different**
                #             # traffic‑light, we stop **before** adding it.
                #             # -------------------------------------------------------------
                #             next_node = (self.net.getLane(next_lane_id).getEdge().getFromNode()
                #                         if group == "incoming"
                #                         else self.net.getLane(next_lane_id).getEdge().getToNode())

                #             if next_node.getID() in self.tl_ids and next_node.getID() != tl_id:
                #                 break      # reached another TL – do NOT cross it

                #             # Otherwise: append, mark visited, update distance, continue
                #             expanded.append(next_lane_id)
                #             visited.add(next_lane_id)
                #             total_dist += self.net.getLane(next_lane_id).getLength()

                #         self.tl_lane_dict[tl_id]["vehicle"][group][dir_key] = expanded

                if real_world: # Hard code for Craver
                    if tl_id == '9727816850_c0_mid':
                        self.tl_lane_dict['9727816850_c0_mid']["pedestrian"]["incoming"]["north"]["main"].extend(['edge_9727816851_9727816850_c0_mid', 'edge_9727816846_9727816850_c0_mid'])
                    elif tl_id == '9727816623_c0_mid': 
                        self.tl_lane_dict['9727816623_c0_mid']["pedestrian"]["incoming"]["north"]["main"].extend(['edge_9727816625_9727816623_c0_mid', 'edge_9666274798_9727816623_c0_mid'])
                    elif tl_id == '9740157155_c0_mid':
                        self.tl_lane_dict['9740157155_c0_mid']["pedestrian"]["incoming"]["north"]["main"].extend(['edge_9740157154_9740157155_c0_mid', 'edge_9666274886_9740157155_c0_mid'])
                    elif tl_id == 'cluster_9740157181_9740483933_c0_mid':
                        self.tl_lane_dict['cluster_9740157181_9740483933_c0_mid']["pedestrian"]["incoming"]["north"]["main"].extend(['edge_9655154530_cluster_9740157181_9740483933_c0_mid', 'edge_9740157180_cluster_9740157181_9740483933_c0_mid'])
                    elif tl_id == '9740157194_c0_mid':
                        self.tl_lane_dict['9740157194_c0_mid']["pedestrian"]["incoming"]["north"]["main"].extend(['edge_9740157195_9740157194_c0_mid', 'edge_9740157204_9740157194_c0_mid'])
                    elif tl_id == '9740157209_c0_mid':
                        self.tl_lane_dict['9740157209_c0_mid']["pedestrian"]["incoming"]["north"]["main"].extend(['edge_9740157210_9740157209_c0_mid', 'edge_9740484420_9740157209_c0_mid'])
                    elif tl_id == '9740484527_c0_mid':
                        self.tl_lane_dict['9740484527_c0_mid']["pedestrian"]["incoming"]["north"]["main"].extend(['edge_9740484528_9740484527_c0_mid', 'edge_9740484524_9740484527_c0_mid'])
                else: 
                    stem = tl_id.removesuffix('_mid')
                    top_edge = f"edge_{stem}_top_{tl_id}"
                    bottom_edge = f"edge_{stem}_bottom_{tl_id}"
                    # print(f"Top edge: {top_edge}, Bottom edge: {bottom_edge}")
                    self.tl_lane_dict[tl_id]["pedestrian"]["incoming"]["north"]["main"].append(top_edge)
                    self.tl_lane_dict[tl_id]["pedestrian"]["incoming"]["north"]["main"].append(bottom_edge)
        
        # print("TL lane dict:")
        # pprint.pprint(self.tl_lane_dict)

    def close(self):
        if self.sumo_running:
            traci.switch(self.traci_label)# switch to this worker's context
            traci.close(False) #https://sumo.dlr.de/docs/TraCI/Interfacing_TraCI_from_Python.html
            self.sumo_running = False

    def _get_pedestrian_arrival_times(self):
        """
        Each pedestrian has a unique id and also a unique arrival time (for the first arrival at the crosswalk).
        A lot of pedestrians can be accomodated in walking areas (w0, w1).
        As this is part of the design agent, this tracking also needs to be done during warmup.
        """

        # print(f"\nPedestrian incoming edges: {self.mb_ped_incoming_edges_all}\n")
        for ped_id in traci.person.getIDList():
            edge_id = '_'.join(traci.person.getLaneID(ped_id).split('_')[:-1])
            # print(f"Pedestrian {ped_id} is on edge: {edge_id}")
            if edge_id in self.mb_ped_incoming_edges_all: 
                # only log the first arrival time
                if ped_id not in self.pedestrian_arrival_times:
                    existed_time = self.pedestrian_existence_times[ped_id]
                    # print(f"Pedestrian {ped_id} has arrived to crosswalk at {existed_time} seconds")
                    self.pedestrian_arrival_times[ped_id] = existed_time

    def _update_pedestrian_existence_times(self):
        """
        Get the existence time (measured in number of seconds) of a pedestrian.
        """
        for ped_id in traci.person.getIDList():
            if ped_id not in self.pedestrian_existence_times:
                self.pedestrian_existence_times[ped_id] = 0.0
            else:
                self.pedestrian_existence_times[ped_id] += 1 * self.step_length
    
    # Eval specific methods
    def get_vehicle_waiting_time(self,):
        """
        The cumulative incremental wait time for all vehicles.
        Instead of summing the cumulative waiting time each step, we add only the difference since the last call.
        """
        incremental_wait = 0.0
        current_vehicle_ids = traci.vehicle.getIDList()
        for veh_id in current_vehicle_ids:
            current_wait = traci.vehicle.getWaitingTime(veh_id)
            last_wait = self.prev_vehicle_waiting_time.get(veh_id, 0.0)
            diff = current_wait - last_wait
            # In case the waiting time was reset (e.g. when a vehicle moves) and diff becomes negative, use current_wait.
            if diff < 0:
                diff = current_wait
            incremental_wait += diff
            self.prev_vehicle_waiting_time[veh_id] = current_wait
            if veh_id not in self.total_unique_ids_veh:
                self.total_unique_ids_veh.append(veh_id)
        return incremental_wait

    def get_pedestrian_waiting_time(self):
        """
        The cumulative incremental wait time for all pedestrians.
        Only the increase in waiting time since the last call is accumulated.
        """
        incremental_wait = 0.0
        current_ped_ids = traci.person.getIDList()
        for ped_id in current_ped_ids:
            current_wait = traci.person.getWaitingTime(ped_id)
            last_wait = self.prev_ped_waiting_time.get(ped_id, 0.0)
            diff = current_wait - last_wait
            if diff < 0:
                diff = current_wait
            incremental_wait += diff
            self.prev_ped_waiting_time[ped_id] = current_wait
            if ped_id not in self.total_unique_ids_ped:
                self.total_unique_ids_ped.append(ped_id)
        return incremental_wait
    
    def total_unique_ids(self):
        """
        So far in the simulation, how many total unique ids were seen. 
        """
        return len(self.total_unique_ids_veh),  len(self.total_unique_ids_ped)
