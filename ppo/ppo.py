import math
import torch
import torch.optim as optim
from .ppo_utils import GraphDataset, collate_fn
from torch.utils.data import TensorDataset, DataLoader
from .models import MLP_ActorCritic, GAT_v2_ActorCritic

class PPO:
    """
    PPO for both higher and lower level agents.
    """
    def __init__(self, 
                 model_dim, # could be state_dim for mlp, in_channels for cnn, in_channels for gatv2
                 action_dim, # is max_proposals for gatv2
                 device, 
                 lr, 
                 gamma, 
                 K_epochs, 
                 eps_clip, 
                 ent_coef, 
                 vf_coef, 
                 batch_size, 
                 gae_lambda,
                 max_grad_norm,
                 vf_clip_param,
                 agent_type,
                 model_kwargs):
        
        self.model_dim = model_dim
        self.action_dim = action_dim
        self.device = device
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.batch_size = batch_size
        self.gae_lambda = gae_lambda
        self.max_grad_norm = max_grad_norm
        self.vf_clip_param = vf_clip_param
        self.agent_type = agent_type

        if self.agent_type == 'lower':
            self.policy = MLP_ActorCritic(self.model_dim, self.action_dim, **model_kwargs).to(self.device)
            self.policy_old = MLP_ActorCritic(self.model_dim, self.action_dim, **model_kwargs).to(self.device) 

        elif self.agent_type == "higher":
            self.policy = GAT_v2_ActorCritic(self.model_dim, self.action_dim, **model_kwargs).to(self.device)
            self.policy_old = GAT_v2_ActorCritic(self.model_dim, self.action_dim, **model_kwargs).to(self.device)

        else:
            raise ValueError(f"Invalid agent type: {agent_type}. Must be 'lower' or 'higher'.")
        
        param_counts = self.policy.param_count()
        print(f"\nModel parameters:")
        for k, v in param_counts.items():
            print(f"\t{k}: {v}")

        # Copy the parameters from the current policy to the old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        # Set up the optimizer for the current policy network
        self.initial_lr = lr
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.initial_lr, eps=1e-5)

    def update_learning_rate(self, update, total_updates):
        """
        Linear annealing. At the end of training, the learning rate is 0.
        """
        frac = 1.0 - (update - 1.0) / total_updates
        new_lr = frac * self.initial_lr
        self.optimizer.param_groups[0]['lr'] = new_lr
        return new_lr

    def compute_gae(self, rewards, values, is_terminals, gamma, gae_lambda, bootstrap_value=0.0):
        """
        Bootstrap an unfinished fragment from its next-state value.
        True terminal transitions always use zero future value.
        """ 

        advantages = torch.zeros_like(rewards, dtype=torch.float32)
        gae = 0.0
        next_value = bootstrap_value
        # First, we iterate through the rewards in reverse order.
        for step in reversed(range(rewards.shape[0])):
            # If its the terminal step (which has no future) or if its the last step in our collected experiences (which may not be terminal).
            if is_terminals[step]:
                next_value = 0.0
                gae = 0.0

            # For each step, we calculate the TD error (delta). Equation 12 in the paper. delta = r + γV(s') - V(s)
            delta = rewards[step] + gamma * next_value * (1 - int(is_terminals[step])) - values[step]

            # Equation 11 in the paper. GAE(t) = δ(t) + (γλ)δ(t+1) + (γλ)²δ(t+2) + ...
            gae = delta + gamma * gae_lambda * (1 - int(is_terminals[step])) * gae 
            advantages[step] = gae

            # Update the next value for the next step
            next_value = values[step]
            #print(f"Next value: {next_value}")
        return advantages

    @torch.no_grad()
    def _control_diagnostics(self, states, actions, old_logprobs, reference_logits):
        """Measure the complete rollout, including only each transition's active heads."""
        logits = self.policy.actor(states)
        active = actions[:, 1:] >= 0
        intersection = torch.distributions.Categorical(logits=logits[:, :4])
        crossings = torch.distributions.Bernoulli(logits=logits[:, 4:])
        logprobs = intersection.log_prob(actions[:, 0]) + (
            crossings.log_prob(actions[:, 1:].to(dtype=logits.dtype, copy=True).clamp_min_(0)) * active).sum(dim=1)
        logratios = logprobs - old_logprobs
        ratios = logratios.exp()
        exact_kl = torch.distributions.kl_divergence(
            torch.distributions.Categorical(logits=reference_logits[:, :4]), intersection)
        exact_kl += (torch.distributions.kl_divergence(
            torch.distributions.Bernoulli(logits=reference_logits[:, 4:]), crossings) * active).sum(dim=1)
        # Actual joint entropy of the active heads over its maximum ln4 + (active crossings)·ln2.
        entropy = intersection.entropy() + (crossings.entropy() * active).sum(dim=1)
        max_entropy = math.log(4) + active.sum(dim=1) * math.log(2)
        return {
            'approx_kl': ((ratios - 1) - logratios).mean(),
            'exact_kl': exact_kl.mean(),
            'clip_fraction': ((ratios - 1).abs() > self.eps_clip).float().mean(),
            'max_abs_logratio': logratios.abs().max(),
            'entropy_fraction': (entropy / max_entropy).mean(),
        }

    def update(self, memories, num_proposals = None, bootstrap_value=0.0):
        """
        Update the policy and value networks using the collected experiences.
        memories = combined memories from all processes. 
        - Includes GAE
        - For the choice between KL divergence vs. clipping, we use clipping.
    
        The paper expresses the loss as maximization objective. We convert it to minimization by changing the sign.
        num_proposals used for lower level agent during evaluate.
        """

        # print(f"\nMemories.states: {memories.states}, type: {type(memories.states)}")
        # print(f"\nMemories.actions: {memories.actions}")
        # print(f"\nMemories.values: {memories.values}")
        # print(f"\nMemories.logprobs: {memories.logprobs}")
        # print(f"\nMemories.rewards: {memories.rewards}")
        # print(f"\nMemories.is_terminals: {memories.is_terminals}")

        # states = torch.stack(memories.states, dim=0)

        actions = torch.stack(memories.actions, dim=0) # Each action is padded to max_proposals
        num_proposals = torch.tensor(memories.num_proposals, dtype=torch.int32) # should just work for both agent types
        old_values = torch.tensor(memories.values, dtype=torch.float32)
        old_logprobs = torch.tensor(memories.logprobs, dtype=torch.float32)
        rewards = torch.tensor(memories.rewards, dtype=torch.float32)
        is_terminals = torch.tensor(memories.is_terminals, dtype=torch.bool)

        # print(f"\nStates shape: {states.shape}")
        # print(f"\nActions shape: {actions.shape}")
        # print(f"\nOld Values shape: {old_values.shape}")
        # print(f"\nOld logprobs shape: {old_logprobs.shape}")
        # print(f"\nRewards shape: {rewards.shape}")
        # print(f"\nIs terminals shape: {is_terminals.shape}")

        # Compute GAE
        advantages = self.compute_gae(rewards, old_values, is_terminals, self.gamma, self.gae_lambda, bootstrap_value)

        # Advantage = how much better is it to take a specific action compared to the average action. 
        # GAE = difference between the empirical return and the value function estimate.
        # advantages + val = Reconstruction of empirical returns. Because we want the critic to predict the empirical returns.
        returns = advantages + old_values
        # print(f"\nAdvantages: {advantages}, shape: {advantages.shape}")
        # print(f"\nReturns: {returns}, shape: {returns.shape}")

        # Normalize the advantages (only for use in policy loss calculation) after they have been added to get returns.
        # Check if advantages tensor has more than one element before normalizing
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) # Small constant to prevent division by zero
        elif advantages.numel() == 1:
             # Handle single element case (std is 0), maybe set advantage to 0 or keep as is? Let's set to 0.
             print("Warning: Only one advantage value found, setting normalized advantage to 0.")
             advantages = torch.zeros_like(advantages)
        # print(f"\nAdvantages after normalization: {advantages}, shape: {advantages.shape}")

        # Create a dataloader for mini-batching 
        # dataset = TensorDataset(states, actions, old_logprobs, advantages, returns, old_values)
        # dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        if self.agent_type == 'lower':
            states = torch.stack(memories.states, dim=0)
            dataset = TensorDataset(states, actions, num_proposals, old_logprobs, advantages, returns, old_values)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            diagnostic_states = states.to(self.device)
            diagnostic_actions = actions.to(self.device)
            diagnostic_logprobs = old_logprobs.to(self.device)
            with torch.no_grad():
                reference_logits = self.policy_old.actor(diagnostic_states)
            control_before = self._control_diagnostics(
                diagnostic_states, diagnostic_actions, diagnostic_logprobs, reference_logits)
            # Disjoint modules; measured only, never clipped or stepped separately.
            actor_parameters = list(self.policy.actor_layers.parameters()) + list(self.policy.actor_logits.parameters())
            critic_parameters = list(self.policy.critic_layers.parameters()) + list(self.policy.critic_value.parameters())
            avg_actor_grad_norm = 0.0
            avg_critic_grad_norm = 0.0
        else:  # higher level agent
            states = memories.states  # Already a list of DataBatch objects
            dataset = GraphDataset(states, actions, num_proposals, old_logprobs, advantages, returns, old_values)
            # Make sure batch_size does not exceed the number of items in the dataset
            # actual_batch_size = min(self.batch_size, len(dataset))
            # if actual_batch_size == 0:
            #     print("Warning: Empty dataset for PPO update. Skipping update.")
            #     return {} # Return empty dict or handle as appropriate
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)


        avg_policy_loss = 0
        avg_value_loss = 0
        avg_entropy_loss = 0
        avg_total_loss = 0
        approx_kl = torch.tensor(0.0) # Initialize approx_kl

        # Optimize policy for K epochs (terminology used in PPO paper)
        for _ in range(self.K_epochs):
            for batch_data in dataloader:

                states_batch, actions_batch, num_proposals_batch, old_logprobs_batch, advantages_batch, returns_batch, old_values_batch = batch_data
                states_batch = states_batch.to(self.device) # Move DataBatch to device
                actions_batch = actions_batch.to(self.device)
                num_proposals_batch = num_proposals_batch.to(self.device)
                old_logprobs_batch = old_logprobs_batch.to(self.device)
                advantages_batch = advantages_batch.to(self.device)
                returns_batch = returns_batch.to(self.device)
                old_values_batch = old_values_batch.to(self.device)

                # print(f"\nOld logprobs batch shape: {old_logprobs_batch.shape}")
                # print(f"\nAdvantages batch shape: {advantages_batch.shape}")
                # print(f"\nReturns batch shape: {returns_batch.shape}")
                # print(f"\nOld values batch shape: {old_values_batch.shape}")

                # Evaluating old actions and values using current policy network
                if self.agent_type == 'lower':
                    logprobs, state_values, dist_entropy = self.policy.evaluate(states_batch, actions_batch, num_proposals_batch, device = self.device) # Removed .to(self.device) for states_batch and actions_batch as they are moved above
                else: # higher agent
                    logprobs, state_values, dist_entropy = self.policy.evaluate(states_batch, actions_batch, device = self.device) 

                # Match one value per state for both MLP (B, 1) and GAT (B,) critics.
                state_values = state_values.reshape(-1)

                # Finding the ratio (pi_theta / pi_theta_old) for importance sampling (we want to use the samples obtained from old policy to get the new policy)
                # Ensure shapes match for subtraction
                logratios = logprobs - old_logprobs_batch # Removed squeeze, assuming logprobs and old_logprobs_batch have compatible shapes (B,)
                ratios = logratios.exp()
                # print(f"\nLogprobs: {logprobs.shape}")
                # print(f"\nOld logprobs: {old_logprobs_batch.squeeze(-1).shape}")
                # print(f"\nRatios: {ratios.shape}: {ratios}")
                # print(f"\nOld values batch: {old_values_batch.shape}: {old_values_batch}")
                # print(f"\nState values: {state_values.shape}: {state_values}")
                # print(f"\nReturns batch: {returns_batch.shape}: {returns_batch}")

                # Finding Surrogate Loss
                surr1 = ratios * advantages_batch
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages_batch
                # print(f"\nSurrogate 1: {surr1.shape}: {surr1.mean()}")
                # print(f"\nSurrogate 2: {surr2.shape}: {surr2.mean()}")

                # Calculate policy and value losses
                policy_loss = torch.min(surr1, surr2).mean() # Equation 7 in the paper
                # print(f"\nPolicy loss: {policy_loss.item()}")

                # Value function clipping
                clipped_state_values = old_values_batch + torch.clamp(state_values - old_values_batch, -self.vf_clip_param, self.vf_clip_param)

                # compute both the clipped and unclipped value losses (MSE)
                clipped_value_loss = (clipped_state_values - returns_batch) ** 2 # square first
                unclipped_value_loss = (state_values - returns_batch) ** 2
                # print(f"\nClipped value loss: {clipped_value_loss.mean()}")
                # print(f"\nUnclipped value loss: {unclipped_value_loss.mean()}")

                # Value loss is scaled by 0.5 
                value_loss = 0.5 * (torch.max(clipped_value_loss, unclipped_value_loss).mean()) # then mean
                # print(f"\nValue loss: {value_loss.item()}")

                # print(f"\nDist entropy: {dist_entropy.shape}")
                entropy_loss = dist_entropy.mean()
                # print(f"\nEntropy loss: {entropy_loss.item()}")

                # Minimize policy loss and value loss, maximize entropy loss.
                # The signs are negated (wrt to the equation in the paper). This ensures that the optimizer maximizes the PPO objective by minimizing the loss function. It is correct and necessary.
                loss = -policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy_loss # Equation 9 in the paper
                # print(f"\nTotal Loss: {loss.item()}")
                # print("--------------------------------")
                # Take gradient step
                self.optimizer.zero_grad()
                loss.backward()
                if self.agent_type == 'lower':
                    # Pre-clipping L2 norm per module; reads gradients only.
                    with torch.no_grad():
                        avg_actor_grad_norm += torch.stack([p.grad.norm() for p in actor_parameters if p.grad is not None]).norm().item()
                        avg_critic_grad_norm += torch.stack([p.grad.norm() for p in critic_parameters if p.grad is not None]).norm().item()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm) # Clipping to prevent exploding gradients
                self.optimizer.step()

                # Accumulate losses
                avg_policy_loss += policy_loss.item()
                avg_value_loss += value_loss.item()
                avg_entropy_loss += entropy_loss.item()
                avg_total_loss += loss.item()

                # Debug method 1: KL divergence (http://joschu.net/blog/kl-approx.html)
                # How much the new policy diverges from the old policy.
                with torch.no_grad():
                    if self.agent_type == 'higher':
                        approx_kl = ((ratios - 1) - logratios).mean()
                    # print(f"\nApprox KL: {approx_kl.item()}")
                    # print("--------------------------------\n")
                    # # TODO: Early stopping (at the minibatch level) based on KL divergence? Do it in main.


        num_batches = len(dataloader)
        if num_batches > 0:
            avg_policy_loss /= (num_batches * self.K_epochs)
            avg_value_loss /= (num_batches * self.K_epochs)
            avg_entropy_loss /= (num_batches * self.K_epochs)
            avg_total_loss /= (num_batches * self.K_epochs)
            if self.agent_type == 'lower':
                avg_actor_grad_norm /= (num_batches * self.K_epochs)
                avg_critic_grad_norm /= (num_batches * self.K_epochs)
        else:
             avg_policy_loss, avg_value_loss, avg_entropy_loss, avg_total_loss = 0, 0, 0, 0


        # print("\nPolicy New params:")
        # for name, param in self.policy.named_parameters():
        #     print(f"{name}: {param.data}")

        # print("\n\n\nPolicy Old params:")
        # for name, param in self.policy_old.named_parameters():
        #     print(f"{name}: {param.data}")

        if self.agent_type == 'lower':
            # policy_old still holds the pre-update parameters: whole-update actor displacement.
            with torch.no_grad():
                old_actor_parameters = list(self.policy_old.actor_layers.parameters()) + list(self.policy_old.actor_logits.parameters())
                actor_update_norm = torch.stack([(new - old).norm() for new, old in zip(actor_parameters, old_actor_parameters)]).norm().item()

        # Copy new weights into old policy
        # self.policy_old = deepcopy(self.policy)
        self.policy_old.load_state_dict(self.policy.state_dict())
        print(f"\n{self.agent_type} policy updated with avg_policy_loss: {avg_policy_loss}\n") 

        # Return the average batch loss per epoch
        result = {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'entropy_loss': avg_entropy_loss,
            'total_loss': avg_total_loss,
            'approx_kl': approx_kl
        }
        if self.agent_type == 'lower':
            result.update(self._control_diagnostics(
                diagnostic_states, diagnostic_actions, diagnostic_logprobs, reference_logits))
            result['preupdate_max_abs_logratio'] = control_before['max_abs_logratio']
            result['actor_grad_norm'] = avg_actor_grad_norm
            result['critic_grad_norm'] = avg_critic_grad_norm
            result['actor_update_norm'] = actor_update_norm
        return result
    