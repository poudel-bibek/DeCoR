import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch_geometric.data import Data

import review_training
from config import classify_and_return_args, get_config
from ppo.ppo import PPO
from ppo.ppo_utils import Memory

WORKERS = 1


def graph(value):
    return Data(x=torch.full((2, 2), float(value)), edge_index=torch.tensor([[0], [1]]),
                edge_attr=torch.zeros((1, 2)))


class FakeDesignEnv:
    """Returns a different layout graph after every round; no SUMO."""
    last = None

    def __init__(self, *args, **kwargs):
        FakeDesignEnv.last = self
        self.lower_ppo = SimpleNamespace(policy=torch.nn.Linear(1, 1))
        self.lower_reward_normalizer = SimpleNamespace(count=SimpleNamespace(value=0))
        self.lower_update_count = self.action_timesteps = self.global_step = 0
        self.lower_memories = Memory()
        self.current_net_file_path = "/tmp/network_iteration_0.net.xml"
        self.current_network_iteration = 0
        self.extreme_edge_dict = {}
        self.normalizer_x = self.normalizer_y = {"min": 0.0, "max": 1.0}
        self.reset_graph = graph(0)
        self.generated = []

    def reset(self):
        return self.reset_graph

    def step(self, proposals, count, iteration, fixed_control=False, update_layout=True):
        self.global_step += WORKERS * 360
        self.lower_state_normalizer.count.value += WORKERS * 36
        self.generated.append(graph(iteration))
        return self.generated[-1], -float(iteration), -float(iteration), False, {}

    def _apply_action(self, proposals, iteration):
        pass

    def close(self):
        pass


class FakePPO:
    last = None

    def __init__(self, **kwargs):
        FakePPO.last = self
        self.eps_clip = 0.2
        self.policy = torch.nn.Linear(1, 1)
        self.policy_old = SimpleNamespace(eval=lambda: None, act=self.act, critic=self.critic)
        self.inputs = []
        self.updates = []

    def act(self, state, iteration, clamp_min, clamp_max, device, training=True, visualize=False):
        self.inputs.append(("act", state, training))
        proposals = torch.full((1, 10, 2), -1.0)
        proposals[0, :2] = 0.5
        return proposals.clone(), proposals, torch.tensor(2), torch.tensor(0.0)

    def critic(self, state, device):
        self.inputs.append(("critic", state, None))
        return torch.tensor([0.0])

    def update(self, memory, **kwargs):
        self.updates.append((list(memory.states), kwargs))
        return dict.fromkeys(["policy_loss", "value_loss", "entropy_loss", "total_loss", "approx_kl"], 0.0)

    def update_learning_rate(self, *args):
        return 0.0


class OneShotDesignTests(unittest.TestCase):
    def test_configured_design_gamma_returns_immediate_reward_only(self):
        _, _, higher, lower, _ = classify_and_return_args(get_config(), "cpu")
        rewards = torch.tensor([1.0, -2.0, 3.0, -4.0])
        values = torch.tensor([0.5, 0.25, -1.0, 2.0])
        continuing = torch.zeros(4, dtype=torch.bool)
        design = PPO.compute_gae(None, rewards, values, continuing, higher["gamma"],
                                 higher["gae_lambda"], bootstrap_value=100.0)
        torch.testing.assert_close(design, rewards - values, rtol=0, atol=0)
        later = rewards.clone()
        later[1:] += 50.0
        perturbed = PPO.compute_gae(None, later, values, continuing, higher["gamma"],
                                    higher["gae_lambda"], bootstrap_value=-100.0)
        self.assertEqual(perturbed[0].item(), design[0].item())
        # The controller keeps its continuing-fragment objective.
        control = PPO.compute_gae(None, rewards, values, continuing, lower["gamma"],
                                  lower["gae_lambda"], bootstrap_value=100.0)
        cut = PPO.compute_gae(None, rewards, values, continuing, lower["gamma"],
                              lower["gae_lambda"], bootstrap_value=0.0)
        self.assertGreater((control - cut).abs().min().item(), 0.0)

    def test_joint_arm_designs_and_extracts_from_the_reset_graph_only(self):
        settings = dict(smoke=True, development=True, rounds=2, design_horizon_rounds=2, workers=WORKERS)
        with TemporaryDirectory() as directory, \
             patch("review_training.DesignEnv", FakeDesignEnv), \
             patch("review_training.PPO", FakePPO), \
             patch("review_training.design_diagnostics", return_value={}), \
             patch("review_training.save_policy"), \
             patch("review_training.digest", return_value="0" * 64):
            record = review_training.run_arm(Path(directory), "joint", 14100, settings, {})
        env, ppo = FakeDesignEnv.last, FakePPO.last
        self.assertEqual(len(env.generated), 2)
        self.assertFalse(torch.equal(env.generated[0].x, env.reset_graph.x))
        for kind, state, _ in ppo.inputs:
            with self.subTest(kind=kind):
                self.assertTrue(torch.equal(state.x, env.reset_graph.x))
        self.assertEqual([kind for kind, _, _ in ppo.inputs], ["act", "critic", "act", "critic", "act"])
        self.assertEqual([training for _, _, training in ppo.inputs][-1], False)
        [(states, kwargs)] = ppo.updates
        self.assertEqual(len(states), 2)
        self.assertEqual(kwargs, {})
        self.assertEqual(record["rounds"][-1]["design_updates"], 1)
        self.assertEqual(record["layout"]["num_proposals"], 2)


if __name__ == "__main__":
    unittest.main()
