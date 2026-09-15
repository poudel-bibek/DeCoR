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

    def __init__(self, *args, **kwargs):
        self.lower_ppo = SimpleNamespace(policy=torch.nn.Linear(1, 1))
        self.lower_reward_normalizer = SimpleNamespace(count=SimpleNamespace(value=0))
        self.lower_update_count = self.action_timesteps = self.global_step = 0
        self.lower_memories = Memory()
        self.current_net_file_path = "/tmp/network_iteration_0.net.xml"
        self.current_network_iteration = 0
        self.extreme_edge_dict = {}
        self.normalizer_x = self.normalizer_y = {"min": 0.0, "max": 1.0}
        self.reset_graph = graph(0)

    def reset(self):
        return self.reset_graph

    def step(self, proposals, count, iteration, fixed_control=False, update_layout=True):
        self.global_step += WORKERS * 360
        if not fixed_control:
            self.lower_state_normalizer.count.value += WORKERS * 36
        return graph(iteration), -float(iteration), -float(iteration), False, {}

    def _apply_action(self, proposals, iteration):
        pass

    def close(self):
        pass


class FakePPO:

    def __init__(self, **kwargs):
        self.eps_clip = 0.2
        self.policy = torch.nn.Linear(1, 1)
        self.policy_old = SimpleNamespace(eval=lambda: None, act=self.act, critic=self.critic)

    def act(self, state, iteration, clamp_min, clamp_max, device, training=True, visualize=False):
        proposals = torch.full((1, 10, 2), -1.0)
        proposals[0, :2] = torch.tensor([[0.25, 0.5], [0.75, 0.5]])
        proposals[0, :2, 0] += state.x.mean() * 0.1
        return proposals.clone(), proposals, torch.tensor(2), torch.tensor(0.0)

    def critic(self, state, device):
        return state.x.mean().reshape(1)

    def update(self, memory, **kwargs):
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

    def test_design_trials_and_export_ignore_preceding_layouts(self):
        settings = dict(smoke=True, development=True, rounds=2, design_horizon_rounds=2, workers=WORKERS)
        expected = [[0.25, 0.5], [0.75, 0.5]]
        for arm in ("joint", "sequential"):
            with self.subTest(arm=arm), TemporaryDirectory() as directory, \
                 patch("review_training.DesignEnv", FakeDesignEnv), \
                 patch("review_training.PPO", FakePPO), \
                 patch("review_training.design_diagnostics", return_value={}), \
                 patch("review_training.save_policy"), \
                 patch("review_training.digest", return_value="0" * 64):
                record = review_training.run_arm(Path(directory), arm, 14100, settings, {})
            # A state-sensitive policy must produce the canonical design even after
            # step() returns a different graph, including at the sequential boundary.
            for trial in record["rounds"]:
                self.assertEqual(trial["proposals"], expected)
            self.assertEqual(record["layout"]["proposals"], expected)


if __name__ == "__main__":
    unittest.main()
