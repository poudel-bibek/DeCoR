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


if __name__ == "__main__":
    unittest.main()
