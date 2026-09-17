import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

import networkx as nx
import numpy as np

from simulation.control_env import ControlEnv
from simulation.design_env import DesignEnv
from config import classify_and_return_args, get_config


INTERSECTION = 'cluster_172228464_482708521_9687148201_9687148202_#5more'


def observation_env(crossings):
    env = ControlEnv.__new__(ControlEnv)
    env.max_proposals = 10
    env.per_timestep_state_dim = 123
    env.tl_ids = [INTERSECTION] + [f'cross{i}' for i in range(crossings)]
    env.active_slots = np.arange(crossings, dtype=np.int64)
    env.direction_turn_intersection_incoming = range(12)
    env.direction_turn_intersection_inside = range(8)
    env.directions = range(4)
    env.direction_turn_midblock = range(2)
    occupancy = {INTERSECTION: {
        'vehicle': {group: {i: [0] * 137 if group == 'incoming' and i == 0 else []
                            for i in range(n)}
                    for group, n in [('incoming', 12), ('inside', 8), ('outgoing', 4)]},
        'pedestrian': {group: {i: {'main': [], 'vicinity': []} for i in range(4)}
                       for group in ['incoming', 'outgoing']}}}
    for tid in env.tl_ids[1:]:
        occupancy[tid] = {
            'vehicle': {group: {i: [0] * 23 if group == 'incoming' and i == 0 else []
                                for i in range(2)} for group in ['incoming', 'inside', 'outgoing']},
            'pedestrian': {group: {'north': {'main': []}} for group in ['incoming', 'outgoing']}}
    env._get_occupancy_map = lambda: ({}, {})
    env._step_operations = lambda *args, **kwargs: occupancy
    return env


class ControlObservationTests(unittest.TestCase):
    def test_workers_start_without_a_shared_writable_configuration(self):
        _, control, _, _, _ = classify_and_return_args(get_config(), 'cpu')
        with tempfile.TemporaryDirectory() as directory:
            commands = []
            for worker, auto_start in [(0, True), (1, False)]:
                control['auto_start'] = auto_start
                env = ControlEnv(control, directory, worker_id=worker, network_iteration=7)
                with patch('simulation.control_env.scale_demand_sliced_window'), \
                     patch('simulation.control_env.traci.start', side_effect=RuntimeError('stop before SUMO')) as start:
                    with self.assertRaisesRegex(RuntimeError, 'stop before SUMO'):
                        env.reset({}, 4)
                command = start.call_args.args[0]
                commands.append(command)
                self.assertNotIn('-c', command)
                self.assertEqual(command[command.index('--net-file') + 1],
                                 f'{directory}/network_iterations/network_iteration_7.net.xml')
                self.assertEqual('--start' in command, auto_start)
            for flag in ['--log', '--error-log']:
                paths = [command[command.index(flag) + 1] for command in commands]
                self.assertEqual(len(set(paths)), 2)
            self.assertEqual(list(Path(directory).glob('*.sumocfg')), [])

    def test_occupancy_offsets_do_not_shift_with_crossing_count(self):
        for crossings in [0, 4, 5, 10]:
            with self.subTest(crossings=crossings):
                env = observation_env(crossings)
                state = env._get_observation([1] * (crossings + 1), print_map=False)
                self.assertEqual(state.shape, (123,))
                self.assertEqual(state[11], 137)
                np.testing.assert_array_equal(state[crossings + 1:11], -1)
                if crossings:
                    self.assertEqual(state[43], 23)
                np.testing.assert_array_equal(state[43 + 8 * crossings:], -1)

    def test_sparse_slots_pack_each_phase_and_all_eight_crossing_fields(self):
        env = observation_env(2)
        occupancy = env._step_operations({}, {})
        for tid, start in [('cross0', 11), ('cross1', 21)]:
            for offset, group in enumerate(['incoming', 'inside', 'outgoing']):
                occupancy[tid]['vehicle'][group] = {
                    direction: list(range(start + 2 * offset + direction)) for direction in range(2)}
            for offset, group in enumerate(['incoming', 'outgoing']):
                occupancy[tid]['pedestrian'][group]['north']['main'] = list(range(start + 6 + offset))
        env._set_signal_slots(['cross0', 'cross1'], {'cross0': 7, 'cross1': 2})
        phases = {'cross0': 0, 'cross1': 1}
        state = env._get_observation([3] + [phases[tid] for tid in env.tl_ids[1:]], print_map=False)
        expected = np.full(123, -1, dtype=np.float32)
        expected[0], expected[3], expected[8] = 3, 1, 0
        expected[11:43] = 0
        expected[11] = 137
        expected[59:67] = [21, 22, 23, 24, 25, 26, 27, 28]
        expected[99:107] = [11, 12, 13, 14, 15, 16, 17, 18]
        np.testing.assert_array_equal(state, expected)

    def test_survivor_slots_do_not_shift_when_crossings_are_added_or_removed(self):
        env = observation_env(0)
        for ids, slots, expected_ids, expected_slots in [
            (['east', 'west'], {'west': 1, 'east': 7}, ['west', 'east'], [1, 7]),
            (['east', 'new', 'west'], {'west': 1, 'new': 4, 'east': 7},
             ['west', 'new', 'east'], [1, 4, 7]),
            (['east', 'new'], {'new': 4, 'east': 7}, ['new', 'east'], [4, 7]),
        ]:
            env._set_signal_slots(ids, slots)
            self.assertEqual(env.tl_ids, [INTERSECTION] + expected_ids)
            np.testing.assert_array_equal(env.active_slots, expected_slots)
        # An unrelated layout without a map starts fresh, not with the preceding sparse slots.
        positions = {'right': (90, 0), 'left': (10, 0)}
        with patch('simulation.control_env.traci.junction.getPosition', side_effect=positions.__getitem__):
            env._set_signal_slots(['right', 'left'])
        self.assertEqual(env.tl_ids, [INTERSECTION, 'left', 'right'])
        np.testing.assert_array_equal(env.active_slots, [0, 1])

    def test_inside_occupancy_follows_physical_links_not_internal_edge_numbers(self):
        env = observation_env(0)
        signal = 'midblock_west_mid'
        env.tl_ids = [signal]
        env.direction_turn_midblock = ['west-straight', 'east-straight']
        env.tl_lane_dict = {}
        links = [
            [('-road_right_0', '-road_left_0', f':{signal}_0_0')],
            [('road_left_0', 'road_right_0', f':{signal}_2_0')],
            [(f':{signal}_w1_0', f':{signal}_c0_0', '')],
        ]
        roads = {'west': f':{signal}_0', 'east': f':{signal}_2',
                 'east_continuation': f':{signal}_5', 'turn': f':{signal}_1'}
        continuations = {
            f':{signal}_0_0': [('-road_left_0', True, True, False, '', 'G', 's', 3.)],
            f':{signal}_2_0': [('road_right_0', False, True, False, f':{signal}_5_0', 'm', 's', 3.)],
            f':{signal}_5_0': [('road_right_0', True, True, False, '', 'M', 's', 3.)],
        }
        with patch('simulation.control_env.traci.trafficlight.getControlledLinks', return_value=links), \
             patch('simulation.control_env.traci.lane.getEdgeID', side_effect=lambda lane: lane.rsplit('_', 1)[0]), \
             patch('simulation.control_env.traci.lane.getLinks', side_effect=lambda lane, **kwargs: continuations[lane]):
            env.dynamically_populate_edges_lanes({})
        env._get_vehicle_occupancy_midblock = lambda *args: {
            signal: {group: {direction: [] for direction in env.direction_turn_midblock}
                     for group in ['incoming', 'outgoing']}}
        with patch('simulation.control_env.traci.person.getIDList', return_value=[]), \
             patch('simulation.control_env.traci.edge.getLastStepPersonIDs', return_value=[]), \
             patch('simulation.control_env.traci.vehicle.getIDList', return_value=list(roads)), \
             patch('simulation.control_env.traci.vehicle.getPosition', return_value=(0, 0)), \
             patch('simulation.control_env.traci.vehicle.getRoadID', side_effect=roads.__getitem__):
            occupancy, _ = ControlEnv._get_occupancy_map(env)
        self.assertEqual(occupancy[signal]['vehicle']['inside'],
                         {'west-straight': ['west'], 'east-straight': ['east', 'east_continuation']})

    def test_generated_layouts_keep_explicit_identities_and_start_fresh_without_metadata(self):
        env = DesignEnv.__new__(DesignEnv)
        env.max_proposals = 10
        env.design_args = {'save_graph_images': False}
        env.normalizer_x = {'min': 0, 'max': 100}
        env.min_thickness, env.max_thickness = 2, 15
        env.base_networkx_graph = nx.Graph()
        for side, y in [('top', 10), ('bottom', -10)]:
            for end, x in [('left', 0), ('right', 100)]:
                env.base_networkx_graph.add_node(f'{end}_{side}', pos=(x, y), type='regular', width=-1)
            env.base_networkx_graph.add_edge(f'left_{side}', f'right_{side}', width=2,
                                            shape=[(0, y), (100, y)], shape_from=f'left_{side}')
        env.horizontal_nodes_top_ped = ['left_top', 'right_top']
        env.horizontal_nodes_bottom_ped = ['left_bottom', 'right_bottom']
        env.horizontal_edges_veh_original_data = {
            side: {'road': {'from_x': 0, 'to_x': 100, 'shape': [(0, y), (100, y)]}}
            for side, y in [('top', 1), ('bottom', -1)]}
        variants = [
            [('east', .8, .2, 7), ('midblock_west', .2, .3, 1)],
            [('new', .5, .4, 4), ('midblock_west', .2, .3, 1), ('east', .8, .2, 7)],
            [('east', .8, .2, 7), ('new', .5, .4, 4)],
            [('east', .1, .6, 7), ('new', .9, .4, 4)],
        ]
        with patch.object(env, '_update_xml_files'):
            for iteration, variant in enumerate(variants):
                ids = [tid for tid, _, _, _ in variant]
                slots = {f'{tid}_mid': slot for tid, _, _, slot in variant}
                proposals = np.asarray([(x, width) for _, x, width, _ in variant])
                env._apply_action(proposals, iteration, crossing_ids=ids, signal_slots=slots)
                graph = env.iterative_networkx_graph
                self.assertEqual({tid for tid, data in graph.nodes(data=True) if data['type'] == 'middle'},
                                 set(slots))
                for tid, x, width, slot in variant:
                    self.assertEqual(set(graph.neighbors(f'{tid}_mid')), {f'{tid}_top', f'{tid}_bottom'})
                    self.assertAlmostEqual(graph.nodes[f'{tid}_mid']['pos'][0], 100 * x)
                    self.assertAlmostEqual(graph.nodes[f'{tid}_mid']['width'], 2 + 13 * width)
                    self.assertEqual(env.signal_slots[f'{tid}_mid'], slot)
                self.assertEqual(env.crossing_ids, ids)
            env._apply_action(np.asarray([[.7, .2], [.25, .4]]), 13)
        self.assertEqual(env.crossing_ids, ['iter13_0', 'iter13_1'])
        self.assertEqual(env.signal_slots, {'iter13_1_mid': 0, 'iter13_0_mid': 1})
        self.assertNotIn('east_mid', env.iterative_networkx_graph)

    def test_elapsed_age_starts_at_zero_on_first_observation(self):
        env = ControlEnv.__new__(ControlEnv)
        env.step_length = 1
        env.pedestrian_existence_times = {}
        with patch('simulation.control_env.traci.person.getIDList', return_value=['p']):
            for _ in range(11):
                env._update_pedestrian_existence_times()
        self.assertEqual(env.pedestrian_existence_times['p'], 10)

    def test_wait_within_a_decision_is_not_lost(self):
        env = ControlEnv.__new__(ControlEnv)
        env.sumo_running = True
        env.previous_action = None
        env.steps_per_action = 10
        env.step_count = 0
        env.max_timesteps = 10
        env.tl_ids = [INTERSECTION]
        env.corrected_occupancy_map = {}
        env.prev_vehicle_waiting_time = {}
        env.prev_ped_waiting_time = {}
        env.total_unique_ids_veh = []
        env.total_unique_ids_ped = []
        env._detect_switch = lambda *args: ([], [])
        env._get_observation = lambda *args: np.zeros(123)
        env._update_pedestrian_existence_times = lambda: None
        env._get_pedestrian_arrival_times = lambda: None
        env._get_control_reward = lambda *args, **kwargs: 0
        tick = [0]
        def step():
            tick[0] += 1
        with patch('simulation.control_env.traci.simulationStep', side_effect=step), \
             patch('simulation.control_env.traci.vehicle.getIDList', return_value=['v']), \
             patch('simulation.control_env.traci.person.getIDList', return_value=[]), \
             patch('simulation.control_env.traci.vehicle.getWaitingTime',
                   side_effect=lambda _: tick[0] if tick[0] <= 2 else 0):
            _, _, done, _, info = env.eval_step(np.array([0]), tl=True)
        self.assertTrue(done)
        self.assertEqual(info['vehicle_wait'], 2)
        self.assertEqual(info['pedestrian_wait'], 0)


if __name__ == '__main__':
    unittest.main()
