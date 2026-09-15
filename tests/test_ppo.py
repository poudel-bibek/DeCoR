import unittest
from tempfile import TemporaryDirectory

import torch
from torch_geometric.data import Batch, Data

from config import classify_and_return_args, get_config
from ppo.models import GAT_v2_ActorCritic
from ppo.ppo import PPO
from ppo.ppo_utils import Memory


class PPOValueLossTests(unittest.TestCase):
    def test_graph_pool_ignores_padding_when_real_node_scores_are_negative(self):
        features = -torch.arange(1.0, 7.0).unsqueeze(1).repeat(1, 2)
        batch = torch.tensor([0, 0, 1, 1, 1, 1])
        for k in [1, 3, 6]:
            with self.subTest(k=k):
                pooled = GAT_v2_ActorCritic.custom_global_sort_pool(None, features, batch, k)
                for i in [0, 1]:
                    nodes = features[batch == i]
                    expected = torch.nn.functional.pad(nodes[:k], (0, 0, 0, max(0, k - len(nodes))))
                    alone = GAT_v2_ActorCritic.custom_global_sort_pool(
                        None, nodes, torch.zeros(len(nodes), dtype=torch.long), k)
                    torch.testing.assert_close(pooled[i], expected.flatten(), rtol=0, atol=0)
                    torch.testing.assert_close(pooled[i], alone[0], rtol=0, atol=0)

    def test_graph_pool_is_continuous_at_node_score_rank_swaps(self):
        features = torch.tensor([[1., -1.], [-1., 1. - 2e-7]], dtype=torch.float64)
        perturbed = features.clone()
        perturbed[1, 1] += 4e-7
        batch = torch.zeros(2, dtype=torch.long)
        for k in [1, 2]:
            with self.subTest(k=k):
                before = GAT_v2_ActorCritic.custom_global_sort_pool(None, features, batch, k)
                after = GAT_v2_ActorCritic.custom_global_sort_pool(None, perturbed, batch, k)
                self.assertLessEqual((after - before).abs().max().item(),
                                     (perturbed - features).abs().max().item())
                reordered = GAT_v2_ActorCritic.custom_global_sort_pool(None, features.flip(0), batch, k)
                torch.testing.assert_close(before, reordered, rtol=0, atol=0)

    def test_graph_pool_selects_and_differentiates_each_feature_independently(self):
        features = torch.tensor([[-4., -1.], [-2., -3.], [-7., -5.]], requires_grad=True)
        pooled = GAT_v2_ActorCritic.custom_global_sort_pool(
            None, features, torch.tensor([0, 0, 1]), 1)
        torch.testing.assert_close(pooled, torch.tensor([[-2., -1.], [-7., -5.]]))
        gradients, = torch.autograd.grad(pooled.sum(), features)
        torch.testing.assert_close(gradients, torch.tensor([[0., 1.], [1., 0.], [1., 1.]]))
        padded = GAT_v2_ActorCritic.custom_global_sort_pool(
            None, features, torch.tensor([0, 0, 1]), 3)
        gradients, = torch.autograd.grad(padded.square().sum(), features)
        torch.testing.assert_close(gradients, 2 * features)

    def test_continuing_fragment_bootstraps_from_next_state(self):
        rewards, values = torch.ones(4), torch.full((4,), 100.0)
        terminals = torch.zeros(4, dtype=torch.bool)
        advantages = PPO.compute_gae(None, rewards, values, terminals, 0.99, 0.97,
                                     bootstrap_value=100.0)
        torch.testing.assert_close(advantages, torch.zeros(4), rtol=0, atol=0)

    def test_terminal_overrides_bootstrap_and_fragments_match_full_trajectory(self):
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
        values = torch.tensor([0.5, 1.5, 2.5, 3.5])
        terminals = torch.tensor([False, True, False, False])
        full = PPO.compute_gae(None, rewards, values, terminals, 0.99, 0.97,
                               bootstrap_value=5.0)
        first = PPO.compute_gae(None, rewards[:2], values[:2], terminals[:2], 0.99, 0.97,
                                bootstrap_value=100.0)
        last = PPO.compute_gae(None, rewards[2:], values[2:], terminals[2:], 0.99, 0.97,
                               bootstrap_value=5.0)
        torch.testing.assert_close(full, torch.cat([first, last]), rtol=0, atol=0)
        # With lambda=0, a nonterminal cut also matches a single uncut TD calculation.
        continuing = torch.zeros(4, dtype=torch.bool)
        full_td = PPO.compute_gae(None, rewards, values, continuing, 0.99, 0.0,
                                  bootstrap_value=5.0)
        prefix_td = PPO.compute_gae(None, rewards[:2], values[:2], continuing[:2], 0.99, 0.0,
                                    bootstrap_value=values[2].item())
        torch.testing.assert_close(prefix_td, full_td[:2], rtol=0, atol=0)

    def test_perfect_per_state_predictions_have_zero_value_loss(self):
        torch.set_num_threads(1)
        _, _, higher_args, lower_args, _ = classify_and_return_args(get_config(), "cpu")
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        higher_args["model_kwargs"]["run_dir"] = directory.name
        for agent_type, args in [("lower", lower_args), ("higher", higher_args)]:
            for batch_size in [1, 3]:
                with self.subTest(agent_type=agent_type, batch_size=batch_size):
                    torch.manual_seed(42)
                    agent = PPO(**dict(args, lr=0.0, K_epochs=1,
                                       batch_size=batch_size, ent_coef=0.0))
                    memory = Memory()
                    for _ in range(batch_size):
                        if agent_type == "lower":
                            state = torch.randn(10, 123)
                            with torch.no_grad():
                                action, logprob = agent.policy_old.act(state, 4)
                                action = torch.nn.functional.pad(action, (0, 11 - action.numel()), value=-1)
                                value = agent.policy_old.critic(state.unsqueeze(0)).item()
                        else:
                            state = Data(x=torch.randn(3, 2),
                                         edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
                                         edge_attr=torch.randn(4, 2))
                            batch = Batch.from_data_list([state])
                            with torch.no_grad():
                                distribution = agent.policy_old.get_gmm_distribution(batch, "cpu")[0]
                                action = distribution.sample((10,)).unsqueeze(0)
                                logprob = distribution.log_prob(action).sum()
                                value = agent.policy_old.critic(batch, "cpu").item()
                        # A terminal reward equal to the prediction makes each target exact.
                        memory.append(state, action, 4, value, logprob.item(), value, True)
                    if batch_size > 1:
                        self.assertGreater(torch.tensor(memory.values).std().item(), 0.01)
                    result = agent.update(memory)
                    self.assertLess(result["value_loss"], 1e-9)
                    # The update also forwards a nonterminal fragment's final value to GAE.
                    following_values = memory.values[1:] + [5.0]
                    memory.rewards = [v - agent.gamma * nxt
                                      for v, nxt in zip(memory.values, following_values)]
                    memory.is_terminals = [False] * batch_size
                    result = agent.update(memory, bootstrap_value=5.0)
                    self.assertLess(result["value_loss"], 1e-9)

    def test_controller_update_reports_whole_rollout_with_sparse_heads(self):
        torch.set_num_threads(1)
        torch.manual_seed(73)
        lower_args = classify_and_return_args(get_config(), "cpu")[3]
        agent = PPO(**dict(lower_args, lr=1e-4, K_epochs=2, batch_size=3))
        memory = Memory()
        for index, slots in enumerate([[0, 2], [7], [2, 5, 9], [0], [1, 6], [4], [0, 9]]):
            state = torch.randn(10, 123)
            with torch.no_grad():
                action, logprob = agent.policy_old.act(state, len(slots), active_slots=slots)
                value = agent.policy_old.critic(state.unsqueeze(0)).item()
            padded = torch.full((11,), -1, dtype=action.dtype)
            padded[0], padded[1 + torch.tensor(slots)] = action[0], action[1:]
            memory.append(state, padded, len(slots), value, logprob.item(), (-1.) ** index, True)
        states, actions = torch.stack(memory.states), torch.stack(memory.actions)
        with torch.no_grad():
            original_logits = agent.policy_old.actor(states)
        result = agent.update(memory)
        with torch.no_grad():
            logits = agent.policy.actor(states)
            logprobs, _, _ = agent.policy.evaluate(states, actions, torch.tensor(memory.num_proposals))
            delta = logprobs - torch.tensor(memory.logprobs)
            exact = torch.distributions.kl_divergence(
                torch.distributions.Categorical(logits=original_logits[:, :4]),
                torch.distributions.Categorical(logits=logits[:, :4]))
            for index, action in enumerate(actions):
                active = action[1:] >= 0
                exact[index] += torch.distributions.kl_divergence(
                    torch.distributions.Bernoulli(logits=original_logits[index, 4:][active]),
                    torch.distributions.Bernoulli(logits=logits[index, 4:][active])).sum()
        torch.testing.assert_close(result["approx_kl"], ((delta.exp() - 1) - delta).mean())
        torch.testing.assert_close(result["exact_kl"], exact.mean())
        torch.testing.assert_close(result["clip_fraction"],
                                   ((delta.exp() - 1).abs() > agent.eps_clip).float().mean())
        self.assertLess(float(result["preupdate_max_abs_logratio"]), 1e-4)

    def test_controller_kl_excludes_changes_to_inactive_heads(self):
        torch.set_num_threads(1)
        torch.manual_seed(74)
        lower_args = classify_and_return_args(get_config(), "cpu")[3]
        agent = PPO(**dict(lower_args, lr=0.0, K_epochs=1, batch_size=2))
        with torch.no_grad():
            agent.policy.actor_logits.weight.zero_()
            agent.policy.actor_logits.bias.zero_()
            agent.policy_old.load_state_dict(agent.policy.state_dict())
        memory = Memory()
        for slots in [[0, 2], [0], [2]]:
            state = torch.randn(10, 123)
            with torch.no_grad():
                action, logprob = agent.policy_old.act(state, len(slots), active_slots=slots)
                value = agent.policy_old.critic(state.unsqueeze(0)).item()
            padded = torch.full((11,), -1, dtype=action.dtype)
            padded[0], padded[1 + torch.tensor(slots)] = action[0], action[1:]
            memory.append(state, padded, len(slots), value, logprob.item(), value, True)
        with torch.no_grad():
            agent.policy.actor_logits.bias[5] = 9.0  # Slot 1 is inactive in every transition.
        result = agent.update(memory)
        self.assertEqual(float(result["exact_kl"]), 0.0)
        self.assertEqual(float(result["clip_fraction"]), 0.0)


if __name__ == "__main__":
    unittest.main()
