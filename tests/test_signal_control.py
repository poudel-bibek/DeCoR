import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from simulation import signal_control as control


class SignalExecutorTest(unittest.TestCase):
    def setUp(self):
        self.traci = self.enterContext(patch.object(control, 'traci'))
        self.now = 0
        self.occupied = False
        vehicles, pedestrians = control.get_intersection_phase_groups()
        p = pedestrians[1]
        self.physical = {'intersection': vehicles[1] + p['A'] + 'r' + p['B'] + p['C'] + 'r' + p['D'],
                         'crossing': 'GGr'}
        self.traci.simulation.getTime.side_effect = lambda: self.now
        self.traci.trafficlight.getRedYellowGreenState.side_effect = self.physical.__getitem__
        self.traci.trafficlight.setRedYellowGreenState.side_effect = self.physical.__setitem__
        self.traci.trafficlight.getControlledLinks.return_value = [()] * 22
        self.traci.edge.getLastStepPersonIDs.side_effect = lambda edge: ['walking'] if self.occupied else []
        self.traci.edge.getLastStepVehicleNumber.return_value = 0
        env = SimpleNamespace(tl_ids=list(self.physical), step_length=1,
                              tl_lane_dict={tid: {'pedestrian': {'outgoing': {'north': {'main': [tid + '_c']}}}}
                                            for tid in self.physical}, _internal_edges=Mock(return_value=[]))
        self.executor = control.SignalExecutor(env)
        for now in range(10):
            self.apply(now, [1, 1])

    def apply(self, now, actions):
        self.now = now
        return self.executor.apply(actions)

    def test_protected_left_transition_has_yellow_then_occupied_all_red(self):
        self.occupied = True
        for now in range(10, 14):
            self.apply(now, [2, 0])
            state = self.physical['intersection']
            self.assertIn('y', state[:16])
            self.assertNotIn('G', state)
            self.assertEqual(state[16:], 'r' * 6)
        for now in range(14, 22):
            self.apply(now, [2, 0])
            self.assertEqual(self.physical['intersection'], 'r' * 22)
            self.assertEqual(self.physical['crossing'], 'rrr')
        self.occupied = False
        phases = self.apply(22, [2, 0])
        self.assertEqual(phases, [2, 0])
        self.assertEqual(self.physical['intersection'], self.executor.targets[0][2])
        self.assertEqual(self.physical['crossing'], 'rrG')

    def test_pending_service_is_not_skipped_when_requests_change_during_clearance(self):
        self.occupied = True
        for now in range(10, 20):
            self.apply(now, [2, 0])
        for now in range(20, 24):
            self.apply(now, [0, 1])
        self.occupied = False
        for now in range(24, 29):
            self.assertEqual(self.apply(now, [0, 1]), [2, 0])
            self.assertEqual(self.physical['crossing'], 'rrG')
        self.assertEqual(self.apply(29, [0, 1]), [4, 3])
        self.assertEqual(self.physical['crossing'], 'rrr')

    def test_missing_signal_request_is_rejected_before_any_command(self):
        self.traci.trafficlight.setRedYellowGreenState.reset_mock()
        with self.assertRaises(ValueError):
            self.apply(10, [2])
        self.traci.trafficlight.setRedYellowGreenState.assert_not_called()


if __name__ == '__main__':
    unittest.main()
