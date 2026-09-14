import copy
import queue
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from config import classify_and_return_args, get_config
from ppo.ppo import PPO
from ppo.ppo_utils import Memory
from simulation.design_env import DesignEnv
from simulation.worker import parallel_eval_worker, parallel_train_worker


class FakeControlEnv:
    def __init__(self, config, *args, **kwargs):
        self.config = config
        self.step_count = 0
        self.decisions = 0
        self.late_wait_queries = 0
        self.total_conflicts = self.total_switches = 0
        self.pedestrian_arrival_times = {"pedestrian": 2.0}

    def reset(self, *args, **kwargs):
        return np.zeros((1, 1)), {}

    def eval_step(self, *args, **kwargs):
        self.decisions += 1
        self.step_count += int(self.config["lower_action_duration"] / self.config["step_length"])
        done = self.step_count >= self.config["max_timesteps"]
        return np.zeros((1, 1)), 0, done, False, {"vehicle_wait": 2.0, "pedestrian_wait": 3.0}

    def get_vehicle_waiting_time(self):
        self.late_wait_queries += 1
        return 0

    def get_pedestrian_waiting_time(self):
        self.late_wait_queries += 1
        return 0

    def total_unique_ids(self):
        return 2, 3

    def close(self):
        pass


class EvaluationWorkerTests(unittest.TestCase):
    def setUp(self):
        self.control_args = {
            "max_timesteps": 360, "lower_action_duration": 10, "step_length": 1.0,
            "manual_demand_veh": None, "manual_demand_ped": None,
        }
        self.config = {
            "control_args": self.control_args, "worker_demand_scale": 1.5,
            "lower_policy": SimpleNamespace(act=lambda *args, **kwargs: (torch.zeros(1), None)),
            "lower_state_normalizer": SimpleNamespace(normalize=lambda state: state),
            "worker_device": "cpu", "run_dir": "/tmp", "network_iteration": 0,
            "num_proposals": 4, "n_iterations": 1, "total_action_timesteps_per_episode": 45,
        }

    def run_worker(self):
        instances = []

        def make_env(*args, **kwargs):
            env = FakeControlEnv(*args, **kwargs)
            instances.append(env)
            return env

        results = queue.SimpleQueue()
        with patch("simulation.worker.ControlEnv", side_effect=make_env):
            parallel_eval_worker(0, self.config, results, None, {})
        scale, episodes = results.get_nowait()
        self.assertEqual(scale, 1.5)
        return instances[0], episodes[0]

    def test_evaluation_runs_all_requested_decisions_without_mutating_training_config(self):
        original = copy.deepcopy(self.control_args)
        env, result = self.run_worker()
        self.assertEqual(env.decisions, 45)
        self.assertEqual(env.step_count, 450)
        self.assertEqual(self.control_args, original)
        self.assertIsNot(env.config, self.control_args)
        self.assertEqual(result["requested_measurement_duration_s"], 450)
        self.assertEqual(result["executed_measurement_duration_s"], 450)

    def test_waits_use_decision_info_without_querying_consumed_counters(self):
        env, result = self.run_worker()
        self.assertEqual(result["total_veh_waiting_time"], 90)
        self.assertEqual(result["total_ped_waiting_time"], 135)
        self.assertEqual(env.late_wait_queries, 0)

    def test_subsecond_steps_preserve_requested_seconds(self):
        self.control_args["step_length"] = 0.5
        env, result = self.run_worker()
        self.assertEqual(env.decisions, 45)
        self.assertEqual(env.config["max_timesteps"], 900)
        self.assertEqual(result["executed_measurement_duration_s"], 450)

    def test_action_duration_cannot_silently_truncate_simulation_steps(self):
        self.control_args["step_length"] = 0.3
        with self.assertRaisesRegex(ValueError, "integer multiple"):
            self.run_worker()

    def test_training_accepts_intersection_plus_ten_crossing_actions(self):
        policy = SimpleNamespace(
            act=lambda *args: (torch.zeros(11), torch.tensor(0.0)),
            critic=lambda *args: torch.tensor(0.0),
        )
        config = dict(self.control_args, total_action_timesteps_per_episode=1)
        env = FakeControlEnv(config)
        env.train_step = lambda action: (np.zeros((1, 1)), 0, True, False, {})
        env._get_design_reward = lambda count: 0
        results = queue.SimpleQueue()
        with patch("simulation.worker.ControlEnv", return_value=env):
            parallel_train_worker(0, "/tmp", policy, config, results, 123,
                                  torch.tensor(10), 10, self.config["lower_state_normalizer"],
                                  {}, "cpu", 0, None)
        rank, memory, reward, executed_decisions = results.get_nowait()
        self.assertEqual(executed_decisions, 1)
        self.assertEqual(memory.actions[0].shape, (11,))


class TrainingCollectionTests(unittest.TestCase):
    def worker_packets(self, decisions, rank):
        _, config, _, _, _ = classify_and_return_args(get_config(), "cpu")
        config.update(max_timesteps=decisions * 10, total_action_timesteps_per_episode=decisions)
        env = FakeControlEnv(config)

        def train_step(action):
            state, _, done, truncated, info = env.eval_step(action)
            return state, float(rank * 100), done, truncated, info

        env.train_step = train_step
        env._get_design_reward = lambda count: float(rank)
        policy = SimpleNamespace(
            act=lambda *args: (torch.zeros(5), torch.tensor(0.0)),
            critic=lambda *args: torch.tensor(0.0),
        )
        results = queue.SimpleQueue()
        with patch("simulation.worker.ControlEnv", return_value=env):
            parallel_train_worker(rank, "/tmp", policy, config, results, 123 + rank,
                                  torch.tensor(4), 10, SimpleNamespace(normalize=lambda x: x),
                                  {}, "cpu", 0, None)
        packets = []
        while not results.empty():
            packets.append(results.get_nowait())
        return packets

    def test_worker_returns_one_complete_episode_at_different_horizons(self):
        for decisions in [23, 36]:
            with self.subTest(decisions=decisions):
                packets = self.worker_packets(decisions, 0)
                self.assertEqual(len(packets), 1)
                _, memory, design_reward, executed_decisions = packets[0]
                self.assertEqual(executed_decisions, decisions)
                self.assertEqual(len(memory.states), decisions)
                self.assertEqual(memory.is_terminals, [False] * (decisions - 1) + [True])
                self.assertEqual(design_reward, 0)

    def test_fixed_controller_runs_without_policy_normalizer_or_ppo_memory(self):
        for requested, horizon in [(23, 23), (36, 36), (36, 23)]:
            with self.subTest(requested=requested, horizon=horizon):
                config = {"max_timesteps": horizon * 10, "lower_action_duration": 10,
                          "step_length": 1.0, "total_action_timesteps_per_episode": requested}
                env = FakeControlEnv(config)
                env.reset = Mock(wraps=env.reset)
                env.eval_step = Mock(wraps=env.eval_step)
                env.train_step = Mock(side_effect=AssertionError("Learned control used"))
                env._get_design_reward = Mock(return_value=-73.0)
                policy = Mock()
                normalizer = Mock()
                results = queue.SimpleQueue()
                with patch("simulation.worker.ControlEnv", return_value=env), \
                     patch("simulation.worker.Memory", side_effect=AssertionError("PPO memory created")):
                    parallel_train_worker(0, "/tmp", policy, config, results, 123,
                                          torch.tensor(4), 10, normalizer, {}, "cpu", 0, None,
                                          fixed_control=True)
                self.assertEqual(results.get_nowait(), (0, None, -73.0, horizon))
                self.assertTrue(results.empty())
                env.reset.assert_called_once_with({}, 4, tl=True, eval_mode=False)
                self.assertEqual(env.eval_step.call_count, horizon)
                for call in env.eval_step.call_args_list:
                    self.assertEqual(call.kwargs, {"tl": True})
                    np.testing.assert_array_equal(call.args[0], np.zeros(5, dtype=np.int32))
                env._get_design_reward.assert_called_once_with(4)
                self.assertEqual(policy.mock_calls, [])
                self.assertEqual(normalizer.mock_calls, [])
                env.train_step.assert_not_called()

    def test_fixed_design_counts_work_without_learning_or_rebuilding_frozen_layout(self):
        env = DesignEnv.__new__(DesignEnv)
        env.control_args = {"lower_num_processes": 2, "global_seed": 123,
                            "lower_update_freq": 1, "lower_action_duration": 10}
        env.lower_ppo_args = {"device": "cpu"}
        env.run_dir, env.max_proposals = "/tmp", 10
        env.current_network_iteration = 3
        env.current_net_file_path = "/tmp/network_iterations/network_iteration_3.net.xml"
        env.extreme_edge_dict = {}
        env.lower_ppo = Mock()
        env.lower_state_normalizer = Mock()
        env.lower_reward_normalizer = Mock()
        env.higher_reward_normalizer = Mock()
        env.higher_reward_normalizer.normalize.side_effect = torch.as_tensor
        env.lower_memories = Memory()
        env.lower_memories.states.append("existing trajectory")
        original_memory = copy.deepcopy(vars(env.lower_memories))
        env.action_timesteps, env.global_step, env.lower_update_count = 7, 100, 2
        env.info = {"unchanged": True}
        env._apply_action = Mock()
        env.iterative_networkx_graph = None
        env._convert_to_torch_geometric = lambda graph: SimpleNamespace(
            x=torch.zeros((1, 2)), edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_attr=torch.empty((0, 2)))
        results = queue.SimpleQueue()
        for _ in range(2):
            results.put((0, None, -50.0, 23))
            results.put((1, None, -90.0, 17))
        processes = [Mock() for _ in range(4)]
        with patch("simulation.design_env.mp.Queue", return_value=results), \
             patch("simulation.design_env.mp.Process", side_effect=processes) as spawn:
            for iteration in [8, 9]:
                _, reward, raw_reward, _, info = env.step(
                    torch.zeros((1, 10, 2)), torch.tensor(4), iteration,
                    fixed_control=True, update_layout=False)
                self.assertEqual((reward, raw_reward), (-70.0, -70.0))
                self.assertIs(info, env.info)
        self.assertEqual(env.global_step, 100 + 2 * (23 + 17) * 10)
        self.assertEqual((env.action_timesteps, env.lower_update_count), (7, 2))
        self.assertEqual(vars(env.lower_memories), original_memory)
        self.assertEqual(env.lower_ppo.mock_calls, [])
        self.assertEqual(env.lower_state_normalizer.mock_calls, [])
        self.assertEqual(env.lower_reward_normalizer.mock_calls, [])
        self.assertEqual(env.higher_reward_normalizer.normalize.call_count, 4)
        env._apply_action.assert_not_called()
        self.assertEqual(env.current_network_iteration, 3)
        for call, iteration, rank in zip(spawn.call_args_list, [8, 8, 9, 9], [0, 1, 0, 1]):
            args = call.kwargs["args"]
            self.assertIsNone(args[2])
            self.assertEqual(args[5], 123 + iteration * 1000 + rank)
            self.assertEqual(args[11:14], (3, env.current_net_file_path, True))
        for process in processes:
            process.start.assert_called_once()
            process.join.assert_called_once()

    def test_update_waits_for_every_worker_and_preserves_episode_advantages(self):
        for decisions in [23, 36]:
            with self.subTest(decisions=decisions):
                packets = [self.worker_packets(decisions, rank) for rank in range(2)]
                results = queue.SimpleQueue()
                # A legal arrival order also interleaves the old partial transfers.
                for index in range(max(map(len, packets))):
                    for worker_packets in packets:
                        if index < len(worker_packets):
                            results.put(worker_packets[index])
                processes = [Mock(), Mock()]
                policy = Mock()
                policy.to.return_value = policy
                env = DesignEnv.__new__(DesignEnv)
                env.control_args = {
                    "lower_num_processes": 2, "global_seed": 123, "lower_update_freq": 1,
                    "lower_anneal_lr": False, "lower_action_duration": 10,
                }
                env.lower_ppo_args = {"device": "cpu", "lr": 0.001}
                env.run_dir, env.max_proposals = "/tmp", 10
                env.lower_state_normalizer = None
                env.extreme_edge_dict, env.current_net_file_path = {}, None
                env.current_network_iteration = 0
                env.lower_memories = Memory()
                env.global_step = env.action_timesteps = env.lower_update_count = 0
                # Identity reward normalization isolates trajectory collection and GAE.
                env.lower_reward_normalizer = env.higher_reward_normalizer = SimpleNamespace(
                    normalize=torch.as_tensor)
                env._apply_action = lambda *args: None
                env.iterative_networkx_graph = None
                env._convert_to_torch_geometric = lambda graph: SimpleNamespace(
                    x=torch.zeros((1, 2)), edge_index=torch.empty((2, 0), dtype=torch.long),
                    edge_attr=torch.empty((0, 2)))

                def update(memory, num_proposals):
                    self.assertTrue(all(process.join.called for process in processes),
                                    "PPO updated before every worker finished")
                    self.assertEqual(len(memory.rewards), 2 * decisions)
                    rewards = torch.tensor(memory.rewards)
                    values = torch.tensor(memory.values)
                    terminals = torch.tensor(memory.is_terminals)
                    actual = PPO.compute_gae(None, rewards, values, terminals, 0.99, 0.95)
                    expected = torch.cat([
                        PPO.compute_gae(None, rewards[start:start + decisions],
                                        values[start:start + decisions],
                                        terminals[start:start + decisions], 0.99, 0.95)
                        for start in [0, decisions]])
                    torch.testing.assert_close(actual, expected)
                    torch.testing.assert_close(actual[:decisions], torch.zeros(decisions))
                    return dict.fromkeys(["policy_loss", "value_loss", "entropy_loss",
                                          "total_loss", "approx_kl"], 0.0)

                env.lower_ppo = SimpleNamespace(policy_old=policy, update=Mock(side_effect=update))
                with patch("simulation.design_env.mp.Queue", return_value=results), \
                     patch("simulation.design_env.mp.Process", side_effect=processes):
                    env.step(torch.zeros((1, 10, 2)), torch.tensor(4), 0)
                env.lower_ppo.update.assert_called_once()
                self.assertEqual(env.global_step, 2 * decisions * 10)
                self.assertEqual(env.action_timesteps, 0)


if __name__ == "__main__":
    unittest.main()
