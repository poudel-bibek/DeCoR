import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np

from simulation.control_env import ControlEnv
from config import classify_and_return_args, get_config


INTERSECTION = 'cluster_172228464_482708521_9687148201_9687148202_#5more'


def observation_env(crossings):
    env = ControlEnv.__new__(ControlEnv)
    env.max_proposals = 10
    env.per_timestep_state_dim = 123
    env.tl_ids = [INTERSECTION] + [f'cross{i}' for i in range(crossings)]
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
