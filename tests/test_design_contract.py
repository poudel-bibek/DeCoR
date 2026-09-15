import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch_geometric.data import Batch, Data

import review_training
from config import classify_and_return_args, get_config
from ppo.models import GAT_v2_ActorCritic, MLP_ActorCritic
from ppo.ppo import PPO
from ppo.ppo_utils import Memory, WelfordNormalizer
from utils import LearnedEvaluationUnavailable, load_design_policy, load_policy, require_exposed_heads, save_policy

WORKERS = 1


def graph(value):
    return Data(x=torch.full((2, 2), float(value)), edge_index=torch.tensor([[0], [1]]),
                edge_attr=torch.zeros((1, 2)))


class FakeDesignEnv:
    """Returns a different layout graph after every round; no SUMO."""

    def __init__(self, *args, **kwargs):
        self.max_proposals = 10
        self.lower_ppo = SimpleNamespace(policy=torch.nn.Linear(1, 1))
        self.lower_ppo.policy.observation_version = 3
        self.lower_reward_normalizer = SimpleNamespace(count=SimpleNamespace(value=0))
        self.lower_update_count = self.action_timesteps = self.global_step = 0
        self.lower_memories = Memory()
        self.control_head_decisions = np.zeros(self.max_proposals, dtype=np.int64)
        self.control_head_updates = np.zeros(self.max_proposals, dtype=np.int64)
        self.current_net_file_path = "/tmp/network_iteration_0.net.xml"
        self.current_network_iteration = 0
        self.extreme_edge_dict = {}
        self.normalizer_x = self.normalizer_y = {"min": 0.0, "max": 1.0}
        self.reset_graph = graph(0)

    def reset(self):
        return self.reset_graph

    def step(self, proposals, count, iteration, fixed_control=False, update_layout=True):
        self.global_step += WORKERS * 360
        if update_layout:
            self._apply_action(proposals[0, :int(count)].numpy(), iteration)
        if not fixed_control:
            self.lower_state_normalizer.count.value += WORKERS * 36
            active = np.fromiter(self.signal_slots.values(), dtype=np.int64)
            self.control_head_decisions[active] += WORKERS * 36
            self.control_head_updates[active] += 1
        return graph(iteration), -float(iteration), -float(iteration), False, {}

    def _apply_action(self, proposals, iteration, crossing_ids=None, signal_slots=None):
        # Mirror the default contract: layout-local IDs and deterministic x-rank slots.
        ids = [f"iter{iteration}_{i}" for i in range(len(proposals))] if crossing_ids is None else list(crossing_ids)
        if signal_slots is None:
            order = sorted(range(len(proposals)), key=lambda i: proposals[i][0])
            signal_slots = {f"{ids[i]}_mid": slot for slot, i in enumerate(order)}
        self.crossing_ids, self.signal_slots = ids, dict(signal_slots)

    def close(self):
        pass


class FakePPO:

    def __init__(self, **kwargs):
        self.eps_clip = 0.2
        self.policy = torch.nn.Linear(1, 1)
        self.policy.readout_version = GAT_v2_ActorCritic.readout_version
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


class CheckpointReadoutTests(unittest.TestCase):
    def setUp(self):
        self.directory = self.enterContext(TemporaryDirectory())
        self.enterContext(torch.random.fork_rng(devices=[]))
        torch.manual_seed(173)
        self.higher = GAT_v2_ActorCritic(
            2, 10, run_dir=self.directory, num_mixtures=2, hidden_channels=4,
            out_channels=4, initial_heads=1, second_heads=1, edge_dim=2,
            readout_k=3, activation="tanh", model_size="small").eval()
        self.lower = MLP_ActorCritic(
            1, 14, action_duration=10, per_timestep_state_dim=123,
            activation="tanh", model_size="small").eval()
        self.normalizer = WelfordNormalizer((10, 123))
        self.normalizer.manual_load(torch.full((10, 123), 0.4), torch.full((10, 123), 2.0), 3)
        self.normalizer.eval()
        self.states = Batch.from_data_list([graph(0.2), graph(0.7)])
        self.observations = torch.linspace(-1, 1, 2 * 10 * 123).reshape(2, 10, 123)
        self.actions = torch.full((2, 11), -1)
        self.actions[:, [0, 2, 8]] = torch.tensor([[1, 0, 1], [2, 1, 0]])
        self.counts = torch.tensor([2, 2])
        self.norm_x, self.norm_y = {"min": 10.0, "max": 100.0}, {"min": -5.0, "max": 5.0}
        decisions, updates = np.zeros(10, dtype=int), np.zeros(10, dtype=int)
        decisions[[1, 7]], updates[[1, 7]] = 36, 1
        self.path = Path(self.directory) / "policy.pth"
        save_policy(self.higher, self.lower, self.normalizer, self.norm_x, self.norm_y,
                    self.path, decisions, updates)

    def predictions(self):
        with torch.no_grad():
            points = torch.tensor([[0.25, 0.4], [0.75, 0.6]])
            densities = torch.stack([gmm.log_prob(points) for gmm in
                                     self.higher.get_gmm_distribution(self.states, "cpu")])
            return (densities, self.higher.critic(self.states, "cpu"),
                    self.lower.evaluate(self.normalizer.normalize(self.observations), self.actions, self.counts))

    def test_checkpoint_roundtrip_preserves_design_and_sparse_controller_predictions(self):
        expected = self.predictions()
        checkpoint = torch.load(self.path)
        self.assertEqual(checkpoint["higher"]["readout_version"], 2)
        self.assertEqual(checkpoint["lower"]["observation_version"], 3)
        with torch.no_grad():
            for policy in (self.higher, self.lower):
                for parameter in policy.parameters():
                    parameter.add_(0.25)
        self.normalizer.manual_load(torch.zeros((10, 123)), torch.ones((10, 123)), 1)
        load_policy(self.higher, self.lower, self.normalizer, self.path)
        torch.testing.assert_close(self.predictions(), expected, rtol=0, atol=0)

    def test_incompatible_design_headers_reject_before_policy_or_normalizer_mutation(self):
        expected = self.predictions()
        checkpoint = torch.load(self.path)
        for section in ("higher", "lower"):
            for parameter in checkpoint[section]["state_dict"].values():
                parameter.add_(0.5)
        checkpoint["lower"]["state_normalizer_mean"].fill(9)
        for version in (None, 1, 3):
            if version is None:
                checkpoint["higher"].pop("readout_version", None)
            else:
                checkpoint["higher"]["readout_version"] = version
            torch.save(checkpoint, self.path)
            with self.subTest(version=version, loader="design"):
                with self.assertRaises(ValueError):
                    load_design_policy(self.higher, checkpoint["higher"])
                torch.testing.assert_close(self.predictions(), expected, rtol=0, atol=0)
            with self.subTest(version=version, loader="combined"):
                with self.assertRaises(ValueError):
                    load_policy(self.higher, self.lower, self.normalizer, self.path)
                torch.testing.assert_close(self.predictions(), expected, rtol=0, atol=0)


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

    def test_explicit_maps_survive_reuse_and_gate_learned_evaluation(self):
        settings = dict(smoke=True, development=False, rounds=2, design_horizon_rounds=2, workers=WORKERS)
        family = {"iter3_1_mid": 7, "iter3_0_mid": 2}  # Non-rank map shared by the layout family.
        with TemporaryDirectory() as directory, \
             patch("review_training.DesignEnv", FakeDesignEnv), \
             patch("review_training.PPO", FakePPO), \
             patch("review_training.design_diagnostics", return_value={}), \
             patch("review_training.digest", return_value="0" * 64):
            joint = review_training.run_arm(Path(directory), "joint", 14100, settings, {})
            self.assertEqual(joint["layout"]["crossing_ids"], ["iter3_0", "iter3_1"])
            self.assertEqual(joint["layout"]["signal_slots"], {"iter3_0_mid": 0, "iter3_1_mid": 1})
            joint["layout"]["signal_slots"] = family
            (Path(directory) / "14100" / "joint" / "training.json").write_text(json.dumps(joint))
            records = {arm: review_training.run_arm(Path(directory), arm, 14100, settings, {})
                       for arm in ("fixed_layout", "random_layout")}
            loaded_provenance = {}
            for arm, record in records.items():
                with self.subTest(arm=arm):
                    # Reuse must export the joint identities and the declared map, never a rank-derived one.
                    self.assertEqual(record["layout"]["crossing_ids"], joint["layout"]["crossing_ids"])
                    self.assertEqual(record["layout"]["signal_slots"], family)
                    _, _, provenance = load_policy(FakePPO().policy, FakeDesignEnv().lower_ppo.policy,
                                                   WelfordNormalizer((10, 123)), record["checkpoint"])
                    loaded_provenance[arm] = provenance
        # Fixed-layout control trained on slots 2 and 7; random-layout control only on rank slots 0 and 1.
        require_exposed_heads(loaded_provenance["fixed_layout"], family)
        with self.assertRaisesRegex(LearnedEvaluationUnavailable, r"heads \[2, 7\]"):
            require_exposed_heads(loaded_provenance["random_layout"], family)
        with self.assertRaisesRegex(LearnedEvaluationUnavailable, "explicit signal-slot map"):
            require_exposed_heads(loaded_provenance["fixed_layout"], None)
        with self.assertRaisesRegex(LearnedEvaluationUnavailable, "provenance"):
            require_exposed_heads({"head_updates": loaded_provenance["fixed_layout"]["head_updates"]}, family)


if __name__ == "__main__":
    unittest.main()
