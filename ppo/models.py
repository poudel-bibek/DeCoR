import os
import pickle
import torch
import numpy as np
import torch.nn as nn
from matplotlib import cm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import global_mean_pool
from torch.distributions import MixtureSameFamily, MultivariateNormal, Categorical, Bernoulli
from .ppo_utils import gmm_entropy_monte_carlo, gmm_entropy_legendre
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Orthogoal initialization of weights and Constant initialization of biases.
    https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
    """
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class MLP_ActorCritic(nn.Module):
    def __init__(self, in_channels, action_dim, **kwargs):
        """
        MLP Actor-Critic network in two sizes: small, medium. 
        - Two separate networks for actor and critic (No shared backbone as there is no feature extraction).
        - Expects inputs of shape (B, in_channels, action_duration, per_timestep_state_dim) then flattens to (B, -1).
        """
        super(MLP_ActorCritic, self).__init__()
        action_duration = kwargs.get('action_duration')
        per_timestep_state_dim = kwargs.get('per_timestep_state_dim')

        with torch.no_grad():
            sample_input = torch.zeros(1, in_channels, action_duration, per_timestep_state_dim)
            self.input_dim = sample_input.numel()  # total number of features, e.g. 1 * c * d * s

        if kwargs.get('activation') == "tanh":
            activation = nn.Tanh
        elif kwargs.get('activation') == "relu":
            activation = nn.ReLU
        elif kwargs.get('activation') == "leakyrelu":
            activation = nn.LeakyReLU
        # dropout_rate = kwargs.get('dropout_rate')

        model_size = kwargs.get('model_size')
        if model_size == 'small':
            actor_hidden_sizes = [256, 128, 64, 32]
            critic_hidden_sizes = [256, 128, 64, 32]
        elif model_size == 'medium':
            actor_hidden_sizes = [512, 256, 128, 64, 32]
            critic_hidden_sizes = [512, 256, 128, 64, 32]
        
        # actor
        actor_layers = []
        input_size_actor = self.input_dim
        for h in actor_hidden_sizes:
            actor_layers.append(layer_init(nn.Linear(input_size_actor, h)))
            actor_layers.append(nn.LayerNorm(h))  # Add LayerNorm after linear layer
            actor_layers.append(activation())
            # actor_layers.append(nn.Dropout(dropout_rate)) # Disabled for now
            input_size_actor = h
        self.actor_layers = nn.Sequential(*actor_layers)
        self.actor_logits = layer_init(nn.Linear(input_size_actor, action_dim)) # Last layer, no activation

        # critic 
        critic_layers = []
        input_size_critic = self.input_dim
        for h in critic_hidden_sizes:
            critic_layers.append(layer_init(nn.Linear(input_size_critic, h)))
            critic_layers.append(nn.LayerNorm(h))  # Add LayerNorm after linear layer
            critic_layers.append(activation())
            # critic_layers.append(nn.Dropout(dropout_rate)) # Disabled for now
            input_size_critic = h
        self.critic_layers = nn.Sequential(*critic_layers)
        self.critic_value = layer_init(nn.Linear(input_size_critic, 1)) # Last layer, no activation

    def actor(self, state):
        """
        First Flatten the input from 4D (B, C, D, S) to 2D (B, -1)
        Returns the raw action logits from the actor head.
        """
        bsz = state.size(0)
        flat = state.view(bsz, -1)  # shape: (B, in_channels*action_duration*per_timestep_state_dim)
        return self.actor_logits(self.actor_layers(flat))

    def critic(self, state):
        """
        First Flatten the input from 4D (B, C, D, S) to 2D (B, -1)
        Returns the scalar state-value V(s).
        """
        bsz = state.size(0)
        flat = state.view(bsz, -1)  # shape: (B, in_channels*action_duration*per_timestep_state_dim)
        return self.critic_value(self.critic_layers(flat))

    def act(self, state, num_proposals, training=True):
        """
        Sample an action exactly like in the CNN version:
          - intersection action from first 4 logits (Categorical)
          - midblock from next num_proposals logits (Bernoulli)
          - Ignore the rest of the logits.

        If training=False, selects the most likely action deterministically.
        Otherwise, samples stochastically.

        TODO: Is there a bias in log_prob because of the number of proposals?
        How to propoerly handle the rest (is ignoring them good?)
        """
        # print(f"Sampling actions for intersection and {num_proposals} midblock proposals...")
        state = state.reshape(1, 1, state.shape[0], state.shape[1])
        action_logits = self.actor(state)

        # The first 4 logits => intersection (Categorical)
        intersection_logits = action_logits[:, :4]
        # intersection_probs = torch.softmax(intersection_logits, dim=1)
        intersection_dist = Categorical(logits=intersection_logits)
        if training:
            intersection_action = intersection_dist.sample()  # [1]
        else:
            intersection_action = torch.argmax(intersection_logits, dim=1) # [1]

        # The next num_proposals logits => midblock (Bernoulli)
        midblock_logits = action_logits[:, 4: 4 + num_proposals]
        # midblock_probs = torch.sigmoid(midblock_logits)
        midblock_dist = Bernoulli(logits=midblock_logits)
        if training:
            midblock_actions = midblock_dist.sample()  # shape [1,num_proposals]
        else:
            # Deterministic: Choose action 1 if logit > 0 (probability > 0.5), else 0
            midblock_actions = (midblock_logits > 0).float() # shape [1, num_proposals]

        # print(f"\nIntersection logits: {intersection_logits}")
        # print(f"\nMidblock logits: {midblock_logits}")

        # print(f"\nIntersection action: {intersection_action}")
        # print(f"\nMidblock actions: {midblock_actions}")
        
        combined_action = torch.cat([intersection_action, midblock_actions.squeeze(0)], dim=0)
        # print(f"\nCombined action: {combined_action}")

        log_prob = intersection_dist.log_prob(intersection_action) + \
                   midblock_dist.log_prob(midblock_actions).sum()

        # print(f"\nAction Log probability: {log_prob}, shape: {log_prob.shape}")
        return combined_action.int(), log_prob

    def evaluate(self, states, actions, num_proposals_batch, device=None):
        """
        Evaluate a batch of states and pre-sampled actions. 
        number of proposals remains same for all parallel actors that collect experiences. 
        
        TODO: Remove the bias due to number of proposals.
        """
        # print(f"Evaluating... with num_proposals: {num_proposals_batch}, shape: {num_proposals_batch.shape}")
        action_logits = self.actor(states) # [B, 4 + max_proposals]
        intersection_logits = action_logits[:, :4] # [B, 4]
        midblock_logits_all = action_logits[:, 4:] # [B, max_proposals - 4]

        # Intersection
        intersection_dist = Categorical(logits=intersection_logits)
        intersection_action = actions[:, :1].squeeze(1).long() # Categorical expects long
        intersection_log_prob = intersection_dist.log_prob(intersection_action)
        intersection_entropy = intersection_dist.entropy()

        batch_size = states.size(0)
        midblock_log_probs  = torch.zeros(batch_size, device=states.device)
        midblock_entropies = torch.zeros(batch_size, device=states.device)

        # Midblock, per sample
        for i, num_proposals in enumerate(num_proposals_batch.tolist()):
            sample_logits = midblock_logits_all[i, :num_proposals]
            sample_actions = actions[i, 1:1+num_proposals].float() # First action was taken by intersection
            sample_dist = Bernoulli(logits=sample_logits)
            # print(f"Sample log probs: {sample_dist.log_prob(sample_actions)}, shape: {sample_dist.log_prob(sample_actions).shape}")
            midblock_log_probs[i] = sample_dist.log_prob(sample_actions).sum() # sum over all num_proposals actions
            # print(f"After summing: {midblock_log_probs[i]}, shape: {midblock_log_probs[i].shape}")
            # print(f"Sample entropy: {sample_dist.entropy()}, shape: {sample_dist.entropy().shape}")
            midblock_entropies[i] = sample_dist.entropy().sum() # sum over all num_proposals actions
            # print(f"After summing: {midblock_entropies[i]}, shape: {midblock_entropies[i].shape}")

        action_log_probs = intersection_log_prob + midblock_log_probs
        total_entropy = intersection_entropy + midblock_entropies

        state_values = self.critic(states)
        return action_log_probs, state_values, total_entropy

    def param_count(self):
        """
        Return a dict describing the parameter counts, mirroring the CNN version.
        """
        actor_params = sum(p.numel() for p in self.actor_layers.parameters()) + sum(p.numel() for p in self.actor_logits.parameters())
        critic_params = sum(p.numel() for p in self.critic_layers.parameters()) + sum(p.numel() for p in self.critic_value.parameters())

        return {
            "Actor": actor_params,
            "Critic": critic_params,
            "Total": actor_params + critic_params,
        }

class GAT_v2_ActorCritic(nn.Module):
    """
    GATv2 (with edge features).
    Graph attention layers are shared between actor and critic.

    Learnings from Exploration paper:
    * To use either log-std or full covariance matrix:

      - Full Covariance:
      - Requires d(d+1)/2 parameters for d components.
      - Allows for correlation between variables (does not have to be diagonal).

      - log-std:  
      - Implicitly uses diagonal covariance matrix.
      - Requires d parameters for d components.
      - Anisotropic (assumption of independence/ variables are uncorrelated).
      - This option chosen for its simplicity.

    * Estimation of Entropy:
      - Discard monte carlo for its non-differentiability.
      - Use Legendre Quadrature for entropy estimation and use monte carlo to validate.

    * Sampling:
      - Make use of implicit stochasticity in sampling from GMM during training (instead of explicit greedy vs random sampling).
      - Greedy sampling from the GMM at evaluation (Maximum propability point corresponding to mean/ center of each component).
    """

    def __init__(self, in_channels, action_dim, **kwargs):
        """
        in_channels: Number of input features per node (e.g., x and y coordinates)
        hidden_channels: Number of hidden features.
        out_channels: Number of output features.
        initial_heads: Number of attention heads for the first GAT layer.
        second_heads: Number of attention heads for the second GAT layer.
        edge_dim: Number of features per edge
        action_dim is the max number of proposals. 

        TODO: 
        # At every timestep, the actions is a whole bunch of things of max size. Critic has to evaluate all that (insted of just the relevant parts).
        # Thickness and location values that are not in the proposal are set to -1 (which is close to minimum of 0.1)
        # model could potentially interpret these as meaningful values. 

        """
        super(GAT_v2_ActorCritic, self).__init__()
        self.in_channels = in_channels
        self.max_proposals = action_dim
        self.run_dir = kwargs.get('run_dir')
        os.makedirs(os.path.join(self.run_dir, 'gmm_iterations'), exist_ok=True)
        self.num_mixtures = kwargs.get('num_mixtures')
        self.hidden_channels = kwargs.get('hidden_channels')
        self.out_channels = kwargs.get('out_channels')
        self.initial_heads = kwargs.get('initial_heads')
        self.second_heads = kwargs.get('second_heads')
        self.edge_dim = kwargs.get('edge_dim')
        self.readout_k = kwargs.get('readout_k') # Number of nodes to keep for each graph
        self.dropout_rate = 0.0 # kwargs.get('dropout_rate', 0.2)
    
        if kwargs.get('activation') == "elu":
            self.activation = nn.ELU()
        elif kwargs.get('activation') == "relu":
            self.activation = nn.ReLU()
        elif kwargs.get('activation') == "tanh":
            self.activation = nn.Tanh()
        elif kwargs.get('activation') == "leakyrelu":
            self.activation = nn.LeakyReLU()

        model_size = kwargs.get('model_size')
        # Actor has some shared MLP layers
        # Followed by four heads: 
        #     1. Predict GMM means (2 values: location and thickness)
        #     2. Predict log-std (2 values: log-std in location and thickness)
        #     3. Predict mix logits (1 value: weight of each Gaussian component)
        #     4. Predict the number of times to sample from the GMM.
        if model_size == "small":
            actor_shared_hidden_sizes = [256, 128] 
            actor_gmm_hidden_sizes = [128, 64] # For actor heads 1, 2, 3
            actor_sample_hidden_sizes = [128, 64] # For actor head 4
            critic_hidden_sizes = [128, 64]
        elif model_size == "medium":
            actor_shared_hidden_sizes = [512, 256]
            actor_gmm_hidden_sizes = [256, 128, 64]
            actor_sample_hidden_sizes = [256, 128, 64]
            critic_hidden_sizes = [256, 128, 64]

        # First Graph Attention Layer. 
        # conv1 should output [num_nodes, hidden_channels * initial_heads]
        # concat=True by default: the outputs from different attn heads are concatenated to output of size hidden_channels * initial_heads.
        # When concat=False, the outputs from different heads are averaged to output of size: hidden_channels. Reduces the dimensionality of the output
        self.conv1 = GATv2Conv(self.in_channels, 
                               self.hidden_channels, 
                               edge_dim = self.edge_dim, 
                               heads = self.initial_heads, 
                               concat = True, 
                               dropout = self.dropout_rate) 

        # Second Graph Attention Layer 
        # Why heads=1? Often, multi-head attention is used in earlier layers to capture different aspects of the graph and the final layer consolidates this info.
        # conv2 should output [num_nodes, out_channels * second_heads] (when concat = True)
        # conv2 should output [num_nodes, out_channels] (when concat = False). This loses too much information.
        self.conv2 = GATv2Conv(self.hidden_channels * self.initial_heads, 
                               self.out_channels, 
                               edge_dim = self.edge_dim, 
                               heads = self.second_heads, 
                               concat = True, 
                               dropout = self.dropout_rate)

        # actor
        # shared
        actor_shared_layers = []
        # input_size_actor_shared = self.out_channels * self.second_heads # For global mean pooling
        input_size_actor_shared = self.out_channels * self.second_heads * self.readout_k # For custom global sort pooling
        # print(f"Input size actor shared: {input_size_actor_shared}")
        for h in actor_shared_hidden_sizes:
            actor_shared_layers.append(layer_init(nn.Linear(input_size_actor_shared, h)))
            # Add layer norm, batch norm, dropout, etc.
            actor_shared_layers.append(nn.LayerNorm(h))
            actor_shared_layers.append(self.activation)
            # actor_shared_layers.append(nn.Dropout(self.dropout_rate))
            input_size_actor_shared = h

        self.actor_shared_layers = nn.Sequential(*actor_shared_layers)

        # means
        actor_means_layers = []
        input_size_actor_means = input_size_actor_shared
        for h in actor_gmm_hidden_sizes:
            actor_means_layers.append(layer_init(nn.Linear(input_size_actor_means, h)))
            # Add layer norm, batch norm, dropout, etc.
            actor_means_layers.append(nn.LayerNorm(h))
            actor_means_layers.append(self.activation)
            # actor_means_layers.append(nn.Dropout(self.dropout_rate))
            input_size_actor_means = h

        self.actor_means_layers = nn.Sequential(*actor_means_layers)
        self.actor_means = layer_init(nn.Linear(input_size_actor_means, self.num_mixtures * 2))

        # log-std
        # actor_log_std_layers = []
        # input_size_actor_log_std = input_size_actor_shared
        # for h in actor_gmm_hidden_sizes:
        #     actor_log_std_layers.append(layer_init(nn.Linear(input_size_actor_log_std, h)))
        #     # Add layer norm, batch norm, dropout, etc.
        #     actor_log_std_layers.append(nn.LayerNorm(h))
        #     actor_log_std_layers.append(self.activation)
        #     # actor_log_std_layers.append(nn.Dropout(self.dropout_rate))
        #     input_size_actor_log_std = h

        # self.actor_log_std_layers = nn.Sequential(*actor_log_std_layers)
        # self.actor_log_std = layer_init(nn.Linear(input_size_actor_log_std, self.num_mixtures * 2))

        # mix logits
        # actor_mix_logits_layers = []
        # input_size_actor_mix_logits = input_size_actor_shared
        # for h in actor_gmm_hidden_sizes:
        #     actor_mix_logits_layers.append(layer_init(nn.Linear(input_size_actor_mix_logits, h)))
        #     # Add layer norm, batch norm, dropout, etc.
        #     actor_mix_logits_layers.append(nn.LayerNorm(h))
        #     actor_mix_logits_layers.append(self.activation)
        #     # actor_mix_logits_layers.append(nn.Dropout(self.dropout_rate))
        #     input_size_actor_mix_logits = h

        # self.actor_mix_logits_layers = nn.Sequential(*actor_mix_logits_layers)
        # self.actor_mix_logits = layer_init(nn.Linear(input_size_actor_mix_logits, self.num_mixtures))

        # number of proposals (Does not share i.e., attached directly to readout layer output)
        # actor_sample_layers = []
        # input_size_actor_sample = self.out_channels * self.second_heads
        # for h in actor_sample_hidden_sizes:
        #     actor_sample_layers.append(layer_init(nn.Linear(input_size_actor_sample, h)))
        #     # Add layer norm, batch norm, dropout, etc.
        #     actor_sample_layers.append(nn.LayerNorm(h))
        #     actor_sample_layers.append(self.activation)
        #     # actor_gmm_layers.append(nn.Dropout(self.dropout_rate))
        #     input_size_actor_sample = h

        # self.actor_sample_layers = nn.Sequential(*actor_sample_layers)
        # self.actor_num_proposals = layer_init(nn.Linear(input_size_actor_sample, self.max_proposals)) # No activation in last layer.

        # critic
        critic_layers = []
        # input_size_critic = self.out_channels * self.second_heads
        input_size_critic = self.out_channels * self.second_heads * self.readout_k
        for h in critic_hidden_sizes:
            critic_layers.append(layer_init(nn.Linear(input_size_critic, h)))
            # Add layer norm, batch norm, dropout, etc.
            critic_layers.append(nn.LayerNorm(h))
            critic_layers.append(self.activation)
            # critic_layers.append(nn.Dropout(self.dropout_rate))
            input_size_critic = h

        self.critic_layers = nn.Sequential(*critic_layers)
        self.critic_value = layer_init(nn.Linear(input_size_critic, 1))
    
    def readout_layer(self, x, batch):
        """
        Applied at the transition of GAT to MLP layers.
        Without it, the expected shape is (num_nodes * out_channels * second_heads): num_nodes can be different for each graph.
        A number of approaches are possible.
        - global_mean_pool: average across the nodes for each graph in a batch.

        """
        # print(f"\nReadout layer input: {x}, x.shape: {x.shape}")
        # pool = global_mean_pool(x, batch)
        pool = self.custom_global_sort_pool(x, batch, k=self.readout_k) # k = how many nodes to keep?
        # print(f"\nReadout layer output: {pool}, pool.shape: {pool.shape}")
        return pool

    def custom_global_sort_pool(self, x, batch, k):
        """
        The Global sort pooling from the DGCNN paper (https://muhanzhang.github.io/papers/AAAI_2018_DGCNN.pdf) as implemented in default PyG function,
        https://pytorch-geometric.readthedocs.io/en/2.0.4/_modules/torch_geometric/nn/glob/sort.html, 
        first sorts nodes individually based on the value of one specific feature channel (last channel), in descending and then selects top k. 
        Instead, we use the mean activation (mean attention score) across all features, then select top-k nodes.

        What is returned is a concatenated fixed-size representation.

        x (Tensor): Node feature matrix [num_nodes, num_features] from GAT layers
        batch (Tensor): Batch vector [num_nodes] containing integers from 0 to batch_size-1
                        Example: [0,0,0,1,1] means first 3 nodes are from graph 0, last 2 from graph 1
        k (int): Number of nodes to keep for each graph
                Usually set to 2 * max_number_of_branches to capture enough structure

        Returns:
            Tensor: Concatenated top-k node features [batch_size, k * num_features]
                    If a graph has fewer than k nodes, zero padding is used
        """
        # print(f"\n--- Entering custom_global_sort_pool ---")
        # print(f"Input x shape: {x.shape}")
        # print(f"Input batch shape: {batch.shape}")
        # print(f"Input k: {k}")

        # Calculate mean activation for each node
        scores = torch.mean(x, dim=1)  # [num_nodes]
        # print(f"Calculated scores shape: {scores.shape}")
        # print(f"Scores sample: {scores[:10]}") # Optional: print first few scores

        # Convert node features and scores to dense batched format (necessary for batch processing of graphs)
        batch_x, mask = to_dense_batch(x, batch)  # [batch_size, max_nodes, num_features]
        batch_scores, _ = to_dense_batch(scores, batch)  # [batch_size, max_nodes]
        # print(f"Dense batch_x shape: {batch_x.shape}")
        # print(f"Dense mask shape: {mask.shape}")
        # print(f"Dense batch_scores shape: {batch_scores.shape}")

        # Get batch size and dimensions
        B, N, D = batch_x.size()  # B: batch_size, N: max_nodes, D: num_features
        # print(f"Batch size B={B}, Max nodes N={N}, Features D={D}")

        # Sort nodes within each graph based on scores
        _, perm = batch_scores.sort(dim=-1, descending=True)  # [batch_size, max_nodes]
        # print(f"Permutation shape after sort: {perm.shape}")

        # Convert to batch-wise indices
        arange = torch.arange(B, device=x.device) * N # explicitly specify device
        perm = perm + arange.view(-1, 1)
        # print(f"Permutation shape after adding arange: {perm.shape}")

        # Flatten batch_x for gathering
        flat_x = batch_x.view(-1, D)
        # print(f"Flattened flat_x shape: {flat_x.shape}")

        # Select top-k nodes with padding
        out = []
        # print(f"Looping through batch size {B}...")
        for i in range(B): # Need to process each separately
            # Get number of actual nodes in this graph
            n_nodes = mask[i].sum().long()
            # print(f"  Graph {i}: n_nodes = {n_nodes.item()}")

            actual_k_for_graph = min(k, n_nodes.item()) # How many nodes we will actually take before padding
            padding_needed = k - actual_k_for_graph
            # print(f"    Graph {i}: actual_k = {actual_k_for_graph}, padding_needed = {padding_needed}")

            if n_nodes >= k:
                # If we have enough nodes, select top k
                graph_x = flat_x[perm[i, :k]]  # [k, num_features]
                # print(f"    Graph {i}: Selected top {k} nodes. graph_x shape: {graph_x.shape}")
            else:
                # If we have fewer than k nodes, pad with zeros
                graph_x = flat_x[perm[i, :n_nodes]]  # [n_nodes, num_features]
                padding = graph_x.new_zeros(k - n_nodes, D)  # [k-n_nodes, num_features]
                # print(f"    Graph {i}: Selected {n_nodes.item()} nodes. graph_x shape before padding: {graph_x.shape}")
                # print(f"    Graph {i}: Padding shape: {padding.shape}")
                graph_x = torch.cat([graph_x, padding], dim=0)  # [k, num_features]
                # print(f"    Graph {i}: graph_x shape after padding: {graph_x.shape}")

            out.append(graph_x)

        # Stack and flatten
        out = torch.stack(out, dim=0)  # [batch_size, k, num_features]
        # print(f"Stacked output shape (before flatten): {out.shape}")
        out = out.view(B, k * D)  # [batch_size, k * num_features]
        # print(f"Final flattened output shape: {out.shape}")
        # print(f"--- Exiting custom_global_sort_pool ---")
        return out

    def actor(self, states_batch, device):
        """
        - GMM parameters prediction 
        - Number of proposals prediction (# of times to sample from GMM)

        State = Data or Batch object with 4 tensors:
        - node features (x) = (num_nodes, in_channels)
        - edge index (edge_index) = (2, num_edges) connections between nodes. # e.g., edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 1, 1]])
        - edge attributes (edge_attr) = Edge features (num_edges, edge_dim)
        - batch (batch)
        """
        # states_batch = states_batch.to(device)
        y = self.conv1(states_batch.x, states_batch.edge_index, states_batch.edge_attr)
        y = self.activation(y)
        # print(f"\nEdge attributes: {states_batch.edge_attr}")
        # print(f"\nAfter conv1: y.shape: {y.shape}")

        # The same edge_attr is used for both the first and second GAT layers. 
        # Preserving edge information: By passing the edge attributes to each layer, the model can maintain information about the edge features throughout the network depth. 
        # Different learned attention: Each GAT layer learns its own attention mechanism. By providing the edge features to each layer, you allow each layer to learn how to use these features differently
        # Residual-like connections for edges: In a sense, passing edge features to each layer creates a form of residual connection for edge information.         
        y = self.conv2(y, states_batch.edge_index, states_batch.edge_attr)
        y = self.activation(y)
        # print(f"\nAfter conv2: y.shape: {y.shape}")
        y = self.readout_layer(y, states_batch.batch)
        # print(f"\nAfter readout: y.shape: {y.shape}")
        # Shared MLP layers
        shared_y = self.actor_shared_layers(y)
        # print(f"\nAfter shared MLP: shared_y.shape: {shared_y.shape}")
        # GMM parameters. Constrain so that the GMM is within the boundary.
        means = torch.sigmoid(self.actor_means(self.actor_means_layers(shared_y)))  # Constrain means to [0,1]
        # print(f"\nAfter means: means.shape: {means.shape}")
        # Scale and shift tanh to get log_stds in [-2.30, 0] range
        # tanh outputs [-1, 1], so we scale by 0.693 and shift by -2.303 to get [-3.0, -0.693], exp(-3.0) ≈ 0.05, exp(-0.693) ≈ 0.5
        # log_stds = torch.tanh(self.actor_log_std(self.actor_log_std_layers(shared_y))) * 0.693 - 2.303  # Results in std range [0.05, 0.5]

        # # tanh outputs [-1, 1], so we scale by 1.15 and shift by -1.15 to get [-2.30, 0], exp(-2.30) ≈ 0.1, exp(0) = 1.
        # log_stds = torch.tanh(self.actor_log_std(self.actor_log_std_layers(shared_y))) * 1.15 - 1.15  # Results in std range [0.1, 1.0]
        log_stds = torch.ones((means.shape[0], self.num_mixtures * 2), device=device) * -2.5 # exp(-2.5) ≈ 0.08

        # mix_logits = self.actor_mix_logits(self.actor_mix_logits_layers(shared_y))
        mix_logits = torch.ones((means.shape[0], self.num_mixtures), device=device) # give equal probability to all components.
        # num_proposal_logits = self.actor_num_proposals(self.actor_sample_layers(y)) # probabilites obtained later.

        # return means, log_stds, mix_logits, num_proposal_logits
        # print(f"\nMeans: {means.shape}, Log-stds: {log_stds.shape}, Mix logits: {mix_logits.shape}")
        return means, log_stds, mix_logits
    
    def critic(self, states_batch, device):
        """
        """
        states_batch = states_batch.to(device)
        y = self.conv1(states_batch.x, states_batch.edge_index, states_batch.edge_attr)
        y = self.activation(y)
        y = self.conv2(y, states_batch.edge_index, states_batch.edge_attr)
        y = self.activation(y)

        y = self.readout_layer(y, states_batch.batch)   
        critic_features = self.critic_layers(y)
        return self.critic_value(critic_features).squeeze(-1)  # Ensure output is of shape (batch_size,)

    def get_gmm_distribution(self, states_batch, device):
        """
        For a GMM with M components, we need: 
        - M mixture weights (logits -> convert to probabilities that sum to 1)
        - M mean vectors (2D: location and thickness)
        - M log-std vectors (2D: log-std in location and thickness)

        Returns:
        - GMM distribution (MixtureSameFamily)
        - num_proposals_probs (Tensor): Probabilities for the number of proposals.
        """

        # In torch geometric, a batch of graphs is a single large graph. Different from gradient mini-batch.
        # batch_size = 1 looks like [0, 0, 0, 0, ... num_nodes times]
        # batch_size = 4 looks like [0, 0, 1, 1, 2, 2, 3, 3, ... num_nodes times]
        batch_size = states_batch.num_graphs 
        # print(f"\nInside get_gmm_distribution:\nBatch size: {batch_size}")

        # Returns (batch_size, num_mixtures * 5) and (batch_size, max_proposals)
        # means, log_stds, mix_logits, num_proposals_logits = self.actor(states_batch)
        # print(f"\nMeans: {means}, Log-stds: {log_stds}, Mix logits: {mix_logits}, Num proposal logits: {num_proposals_logits}")

        # num_proposals_probs_batch = F.softmax(num_proposals_logits, dim=-1)
        # print(f"\nProposal probabilities: {num_proposals_probs_batch}")
        # print(f"\nInside get_gmm_distribution: Higher states batch: {states_batch}")
        means, log_stds, mix_logits = self.actor(states_batch, device=device)
        # print(f"\nMeans: {means}, Log-stds: {log_stds}, Mix logits: {mix_logits}")

        gmms_batch = []
        for b in range(batch_size):
            # reshape to (num_mixtures, 2)
            means_b = means[b].reshape(self.num_mixtures, 2)
            log_stds_b = log_stds[b].reshape(self.num_mixtures, 2)
            # print(f"\nMeans item {b}: {means_b} \nLog-stds item {b}: {log_stds_b}")
            
            # diagonal covariance matrix
            scale_diag = torch.exp(log_stds_b) # exponentiate the logstdss
            mixture_dist = Categorical(logits = mix_logits[b])

            component_dist = MultivariateNormal(
                loc = means_b, 
                scale_tril = torch.diag_embed(scale_diag) 
            )
            gmm = MixtureSameFamily(mixture_distribution = mixture_dist,
                                    component_distribution = component_dist)
            gmms_batch.append(gmm)
        # return gmms_batch, num_proposals_probs_batch
        return gmms_batch
    
    def _sample_gmm(self, gmm_single, num_proposals, naive_stochastic=False, training=True, device=None):
        """

        Sampling from GMM has several issues of discussion.
        1. During training: Stochastic vs Deterministic sampling:
        * Make use of the implicit stochasticity (set naive_stochastic = True)
          Justification: PG methods rely heavily on action space exploration (to get diverse experiences) within the distribution defined by the policy parameters.
          Algorithms like PPO are foundationally built around optimizing a stochastic policy.

        * Greedily sample at highest probability (modes), deterministic action selection.
          Justification: The policy can do exploration in its parameter space. The sampling does not need to introduce additional stochasticity.

            - Discretize: Divide the GMM into 20 x 20 grid.
            - Evaluate: Calculate the probability at each grid cell
            - Find modes: Find the number of modes in the GMM (cells in which probabilities are lower in all 4 directions)
            - TODO: Calculate num_sampled = Either num_samples that collapse into nearby modes within a threshold or min(num_proposals, number of modes)
            - Add constraints: If there are two samples with same location, instead add their width and make a single sample.
            - Sample: Sample num_sampled cells.

            ---------- DISCARDED ----------
            # - Rank and select: Find the top num_proposals number of cells with highest probability.
            # - Sample: Use the center of the cell to get a sample.

            # The problem here is that: the selected cells may always be near the model with highest probability.
            # When the GMM is multi-modal, we would instead want the selected cells to be spread out across multiple modes.

        2. Sample stochastically at train vs deterministically at test time?
        * Making the design agent stochastic at test time does make sense for this problem.
          Justification: The design actions are more like "one-shot" generation unlike the control actions (which are continouus and episodic). 
          The design policy finishes training and we are supposed to get the best design at the end.
          It is kind of doing one round of expectation minimization.

          Samples returned = First populate with all modes. Then populate with stochastic samples.

        """
        if training:
            if naive_stochastic:
                return gmm_single.sample((num_proposals,))
            
            else:
                # TODO: Implement potentially more sophisticated training sampling if needed
                pass

        else:
            # print(f"\n\nInside _sample_gmm: Test time\n\n")
            # Test time: Hybrid sampling (Modes + Stochastic)
            grid_size = 20  # Resolution for finding modes
            loc_min, loc_max = 0.0, 1.0
            thick_min, thick_max = 0.0, 1.0

            # 1. Discretize
            loc_step = (loc_max - loc_min) / grid_size
            thick_step = (thick_max - thick_min) / grid_size

            loc_centers = torch.linspace(loc_min + loc_step / 2, loc_max - loc_step / 2, grid_size, device=device)
            thick_centers = torch.linspace(thick_min + thick_step / 2, thick_max - thick_step / 2, grid_size, device=device)

            grid_loc, grid_thick = torch.meshgrid(loc_centers, thick_centers, indexing='ij')
            grid_points = torch.stack([grid_loc.ravel(), grid_thick.ravel()], dim=-1) # Shape: (grid_size*grid_size, 2)

            # 2. Evaluate GMM probability on the grid
            with torch.no_grad():
                log_probs = gmm_single.log_prob(grid_points)
                probs = torch.exp(log_probs).reshape(grid_size, grid_size) # Shape: (grid_size, grid_size)

            # 3. Find modes
            padded_probs = F.pad(probs, (1, 1, 1, 1), mode='constant', value=0)
            is_mode = (
                (probs > padded_probs[1:-1, :-2]) &  # Compare with left neighbor
                (probs > padded_probs[1:-1, 2:]) &   # Compare with right neighbor
                (probs > padded_probs[:-2, 1:-1]) &  # Compare with top neighbor
                (probs > padded_probs[2:, 1:-1])     # Compare with bottom neighbor
            )
            mode_indices = torch.nonzero(is_mode) # Shape: (num_modes_found, 2) [row, col] indices
            num_modes_found = mode_indices.shape[0]
            # print(f"    _sample_gmm (Eval): Found {num_modes_found} modes on grid. Requesting {num_proposals} proposals.") # DEBUG

            # 4. Select top k modes
            selected_mode_samples = torch.empty((0, 2), device=device, dtype=torch.float32)
            num_modes_taken = 0
            if num_modes_found > 0:
                mode_probs_values = probs[mode_indices[:, 0], mode_indices[:, 1]]
                sorted_mode_indices_idx = torch.argsort(mode_probs_values, descending=True)

                num_modes_to_take = min(num_modes_found, num_proposals)
                top_mode_indices = mode_indices[sorted_mode_indices_idx[:num_modes_to_take]]

                mode_loc_indices = top_mode_indices[:, 0]
                mode_thick_indices = top_mode_indices[:, 1]
                mode_locations = loc_centers[mode_loc_indices]
                mode_thicknesses = thick_centers[mode_thick_indices]
                selected_mode_samples = torch.stack([mode_locations, mode_thicknesses], dim=-1)
                num_modes_taken = selected_mode_samples.shape[0] # Update based on actual number taken

            # Filter out samples with location < 0.1
            # location_mask = selected_mode_samples[:, 0] >= 0.1
            # selected_mode_samples = selected_mode_samples[location_mask]
            # num_modes_taken = selected_mode_samples.shape[0]
            
            # # 5. Sample remaining points stochastically
            # num_to_sample_stochastically = num_proposals - num_modes_taken
            # stochastic_samples = torch.empty((0, 2), device=device, dtype=torch.float32)
            # if num_to_sample_stochastically > 0:
            #     stochastic_samples = gmm_single.sample((num_to_sample_stochastically,))

            # # 6. Combine samples
            # # torch.cat handles empty tensors gracefully
            # return torch.cat((selected_mode_samples, stochastic_samples), dim=0)
            return torch.cat((selected_mode_samples, torch.full((num_proposals - num_modes_taken, 2), -1.0, device=device)), dim=0)

    def act(self, states, iteration, clamp_min, clamp_max, device, training=True, visualize=False):
        """
        Sample actions from the GMM (propose upto max_proposals number of crosswalks).
        Policy gradient methods require the log probabilities of the actions to be returned as well.
        Should thickness and location be independent? 
        - Yes. Diagonal covarianve implies this. 
        - Modeling thickness and location jointly requires more parameters.
        Utilizing the implicit stochasticity in sampling during training.
        However, at test, sample greedily at each mode.
        """
        states_batch = Batch.from_data_list([states]).to(device)
        # properly batch the data using Batch.from_data_list() before sending here. 
        # gmm_batch, num_proposals_probs_batch = self.get_gmm_distribution(states_batch.to(device))
        gmm_batch = self.get_gmm_distribution(states_batch, device=device)
        # # Sample one proposal for each batch element (add 1 to ensure at least 1 proposal; default starts from index 0)
        # if training:
        #     # Stochastic sampling during training
        #     num_proposals = torch.multinomial(num_proposals_probs_batch, 1).squeeze(-1) + 1
        # else:
        #     # Deterministic selection during evaluation (choose max probability)
        #     num_proposals = torch.argmax(num_proposals_probs_batch, dim=-1) + 1
        

        batch_size = states_batch.num_graphs 
        # Initialize output tensors with -1 so that its easier to infer actual proposals in critic.
        original_proposals = torch.full((batch_size, self.max_proposals, 2), -1.0, dtype=torch.float32, device=device)
        merged_proposals = torch.full((batch_size, self.max_proposals, 2), -1.0, dtype=torch.float32, device=device)
        log_probs = torch.zeros(batch_size, device=device)
        num_proposals = torch.zeros(batch_size, dtype=torch.int32, device=device) # 1D tensor of shape (batch_size,)

        for b in range(batch_size):
            # Sample proposals for this batch element
            samples = self._sample_gmm(gmm_batch[b], 
                                    #    num_proposals[b].item(), 
                                       torch.ones(batch_size, dtype=torch.int32, device=device) * int(self.max_proposals), #sample full proposals
                                       naive_stochastic = True, 
                                       training = training, 
                                       device = device)

            locations, thicknesses = samples.split(1, dim=-1)
            # print(f"\nBefore clamping: Locations: {locations}, Thicknesses: {thicknesses}")
            
            original_proposals[b] = torch.cat([locations, thicknesses], dim=-1)

            # Compute log probabilities for this batch element using the actually returned samples (before things like clamping and merging)
            # SUM operation is correct for joint log prob of thickness and location.
            log_probs[b] = gmm_batch[b].log_prob(original_proposals[b]).sum() # Log probs should be of the real samples (not the merged ones).

            # Apply a noisy clamp individually to prevent exact overlap at same locations.
            # Although means are constrained to [0,1], log_stds are not.
            # We should be less dependent on clamping here and make sure the GMM itself lies in the desired range.
            # below_min_mask_loc = locations < clamp_min
            # above_max_mask_loc = locations > clamp_max
            # locations[below_min_mask_loc] = 0.0 + torch.rand_like(locations[below_min_mask_loc], device=device) * clamp_min # In the range [0, clamp_min]
            # locations[above_max_mask_loc] = clamp_max + torch.rand_like(locations[above_max_mask_loc], device=device) * (1.0 - clamp_max) # In the range [clamp_max, clamp_max + noise_offset]
            
            # Standard clamp
            locations = torch.clamp(locations, clamp_min, clamp_max)  
            
            thicknesses = torch.clamp(thicknesses, clamp_min, clamp_max) 
            # print(f"\nAfter clamping: Locations: {locations}, Thicknesses: {thicknesses}")

            # --- Iterative Merging Until Separation --- Even the merged ones cannot be within 0.08 of each other.
            # Analogy: Deterministic environment physics.
            # current_proposals = torch.cat([locations, thicknesses], dim=-1) # Start with clamped proposals
            # Filter out padding *after* clamping, *before* merging, using the original samples tensor
            current_proposals = torch.cat([locations, thicknesses], dim=-1)[samples[:, 0] != -1.0] # TODO: This was a hacky fix.

            threshold = 0.08 # One standard deviation
            while True:
                num_current = current_proposals.shape[0]
                if num_current <= 1:
                    break # Cannot merge less than 2 proposals

                # Calculate pairwise distances based on location only
                locs = current_proposals[:, :1] # Shape (N, 1)
                dist_matrix = torch.cdist(locs, locs) # Shape (N, N)

                # Set diagonal to infinity to avoid self-merging
                dist_matrix.fill_diagonal_(float('inf'))

                # Find the minimum distance between any pair
                min_dist = torch.min(dist_matrix)

                # If the closest pair is already far enough apart, stop merging
                if min_dist >= threshold:
                    break

                # Find the indices (row, col) of the closest pair
                flat_idx = torch.argmin(dist_matrix.view(-1)) # Index in the flattened matrix
                idx1 = torch.div(flat_idx, num_current, rounding_mode='floor')
                idx2 = flat_idx % num_current
                # Ensure idx1 < idx2 for consistent removal logic
                if idx1 > idx2:
                    idx1, idx2 = idx2, idx1

                # Merge the closest pair by averaging
                merged_proposal = (current_proposals[idx1] + current_proposals[idx2]) / 2.0

                # Create a mask to keep all proposals *except* the merged pair (idx1, idx2)
                keep_mask = torch.ones(num_current, dtype=torch.bool, device=device)
                keep_mask[idx1] = False
                keep_mask[idx2] = False

                # Update current_proposals: keep the ones not merged, add the new merged one
                current_proposals = torch.cat((current_proposals[keep_mask], merged_proposal.unsqueeze(0)), dim=0)
            # --- End Iterative Merging ---

            # current_proposals now holds the final set satisfying the separation condition
            actual_num_proposals = current_proposals.shape[0]

            # Store the final (merged and not consumed in merging) proposals, padded
            merged_proposals[b, :actual_num_proposals] = current_proposals
            merged_proposals[b, actual_num_proposals:] = -1.0 # Pad rest with -1

            num_proposals[b] = actual_num_proposals # Store the actual count after merging
            # print(f"\nAfter Merging: {merged_proposals[b]}")

            # Visualization is only meaningful during act (i.e., not during evaluation)
            if visualize and iteration is not None:
                # original_markers = (locations.squeeze().detach().cpu().numpy(), thicknesses.squeeze().detach().cpu().numpy())
                merged_markers = (merged_proposals[b, :actual_num_proposals, 0].squeeze().detach().cpu().numpy(), merged_proposals[b, :actual_num_proposals, 1].squeeze().detach().cpu().numpy())
                self.visualize_gmm(gmm_batch[b], self.run_dir, markers=merged_markers, batch_index=b, thickness_range=(0, 1), location_range=(0, 1), iteration=iteration)

        # print(f"\noriginal_proposals: {original_proposals}")
        # print(f"\nnum_proposals: {num_proposals}")
        # The higher ppo memory needs to store the proposals and log_probs without considering the merging operations.
        # The number of actual proposals is corresponding to the merged proposals.

        
        print(f"\noriginal_proposals: {original_proposals}")
        print(f"\nnum_proposals: {num_proposals}")
        # The higher ppo memory needs to store the proposals and log_probs without considering the merging operations.
        # The number of actual proposals is corresponding to the merged proposals.

        # For testing
        # # Explicitly keep only selected proposals and invalidate others
        # # [0.1250, 0.9750], [0.4000, 0.4875], [0.7250, 0.5750], [0.8500, 0.0500]
        # desired_proposals = torch.tensor([[0.1250, 0.9750], [0.4000, 0.4875], [0.7250, 0.5750]], device=device)
        # num_desired = desired_proposals.shape[0]

        # # Reset all proposals to -1
        # merged_proposals[:] = -1.0  

        # # Set the desired proposals explicitly
        # merged_proposals[0, :num_desired] = desired_proposals

        # # Update num_proposals accordingly
        # num_proposals[0] = num_desired

        # # Debugging output
        # print(f"\nFinal merged_proposals: {merged_proposals}")
        # print(f"\nFinal num_proposals: {num_proposals}")
                
        return original_proposals, merged_proposals, num_proposals, log_probs
    
    def evaluate(self, states_batch, actions_batch, device=None):
        """
        states_batch (Batch): Batch of states, each state a Data object.
        actions_batch (Tensor): Batch of actions [batch_size, max_proposals, 2]. 
            - Not all the actions contain actual proposals. Each element in the batch can have a different number of proposals.

        Returns:
            action_log_probs (Tensor): Log probabilities of the actions.
            state_values (Tensor): Values of the states.
            entropy (Tensor)
        """
        # print(f"\nInside evaluate: Higher states batch: {states_batch}")
        state_values = self.critic(states_batch, device=device)
        # Get distribution (internally passes through actor)
        gmm_batch = self.get_gmm_distribution(states_batch, device=device)
        
        # Initialize return tensors
        batch_size = states_batch.num_graphs 
        action_log_probs = torch.zeros(batch_size, device=device)
        entropy = torch.zeros(batch_size, device=device)
        
        # print(f"\nActions batch: {actions_batch}\n")
        # Compute num_proposals_batch by checking for -1 in actions
        num_proposals_batch = (actions_batch[:, :, :, 0] != -1).sum(dim=2) # squeeze reduces to two dimensions.
        # print(f"\nNum proposals batch: {num_proposals_batch}\n")

        for b in range(batch_size):
            # Get actual proposals for this batch element
            n_proposals = num_proposals_batch[b].item()
            actual_actions = actions_batch[b, :n_proposals]
            
            # Compute log probabilities and entropy for this batch element
            # The sum operation ensures joint log prob of thickness and location.
            action_log_probs[b] = gmm_batch[b].log_prob(actual_actions).sum()
            _, entropy[b] = self.get_entropy(gmm_batch[b])
        
        return action_log_probs, state_values, entropy
    
    def get_entropy(self, gmm_single):
        """
        """
        entropy_mc = gmm_entropy_monte_carlo(gmm_single)
        # entropy_legendre = gmm_entropy_legendre(gmm_single)
        return None, entropy_mc

    def param_count(self):
        """
        Count the total number of parameters in the model.
        """
        # Shared GATv2 layers
        shared_gat_params = sum(p.numel() for p in self.conv1.parameters()) + \
                        sum(p.numel() for p in self.conv2.parameters())

        # Shared MLP layers in actor
        shared_mlp_params = sum(p.numel() for p in self.actor_shared_layers.parameters())

        # Actor-specific 
        actor_means_params = sum(p.numel() for p in self.actor_means_layers.parameters())
        # actor_log_std_params = sum(p.numel() for p in self.actor_log_std_layers.parameters())
        # actor_mix_logits_params = sum(p.numel() for p in self.actor_mix_logits_layers.parameters())
        # actor_sample_params = sum(p.numel() for p in self.actor_sample_layers.parameters())

        # actor_params = shared_mlp_params + actor_means_params + actor_log_std_params + actor_mix_logits_params + actor_sample_params
        actor_params = shared_mlp_params + actor_means_params

        # Critic-specific 
        critic_params = sum(p.numel() for p in self.critic_layers.parameters())

        total_params = shared_gat_params + shared_mlp_params + actor_params + critic_params 
        return {
            "shared_gat": shared_gat_params,
            "actor_shared_mlp": shared_mlp_params,
            "actor_means": actor_means_params,
            # "actor_log_std": actor_log_std_params,
            # "actor_mix_logits": actor_mix_logits_params,
            # "actor_sample": actor_sample_params,
            "actor_total": actor_params,
            "critic": critic_params,
            "Grand total": total_params}
    
    def visualize_gmm(self, gmm_single, run_dir, num_samples=50000, markers=None, batch_index=None, thickness_range=None, location_range=None, iteration=None):
        """
        Visualize the GMM distribution in 3D.
        If sampling locations provided, they are marked as red crosses in a separate top-down view.

        Args:
            gmm_single (MixtureSameFamily): The GMM distribution for a single batch element.
            num_samples (int): Number of samples to generate for visualization.
            markers (tuple of ndarrays): Markers to plot, shape (N, 2).
        """
        fs = 16
        base_save_path = f"{run_dir}/gmm_iterations/gmm_i_{iteration}_b{batch_index}"
        
        # Sample from the GMM
        samples = gmm_single.sample((num_samples,))  # Shape: (num_samples, 2)
        samples = samples.detach().cpu().numpy()

        xmin, xmax = location_range
        ymin, ymax = thickness_range
        X = np.linspace(xmin, xmax, 100)
        Y = np.linspace(ymin, ymax, 100)
        X, Y = np.meshgrid(X, Y)

        # Create positions array for GMM evaluation
        positions = torch.tensor(np.column_stack([X.ravel(), Y.ravel()]), 
                               dtype=torch.float32,
                               device=gmm_single.component_distribution.loc.device)  # Get device from component distribution

        # Evaluate the GMM on the grid and move to CPU for numpy conversion
        Z = gmm_single.log_prob(positions).detach().cpu()
        Z = np.exp(Z.numpy()).reshape(X.shape)

        # Main 3D plot without markers
        fig = plt.figure(figsize=(10, 8), dpi=100)   
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, linewidth=0, antialiased=False, alpha=0.7)
        ax.set_xlabel('Location', fontweight='bold', fontsize=fs, labelpad=15)
        ax.set_ylabel('Thickness', fontweight='bold', fontsize=fs, labelpad=15)
        ax.set_zlabel('Density', fontweight='bold', fontsize=fs, labelpad=15)
        ax.set_title('GMM Distribution', fontweight='bold', fontsize=fs)
        ax.tick_params(axis='both', which='major', labelsize=fs-2)
        
        # Set fixed z-axis limits
        z_max = 4.0
        ax.set_zlim(0, z_max)
        ax.set_zticks(np.linspace(0, z_max, 6))
        
        plt.tight_layout()
        # plt.show()
        plt.savefig(f"{base_save_path}.png")
        plt.close()

        # Create second plot with markers if provided
        if markers is not None:
            fig = plt.figure(figsize=(10, 8), dpi=100)
            ax = plt.gca()
            
            ax.grid(True, linestyle=(0, (5, 8)), alpha=0.9, zorder=11)

            contour = ax.contourf(X, Y, Z, levels=20, cmap=cm.viridis, alpha=0.8, zorder=7)
            cbar = plt.colorbar(contour)
            cbar.set_label('Density', fontweight='bold', fontsize=fs)
            cbar.ax.tick_params(labelsize=fs-2)

            # Get and plot component means
            means = gmm_single.component_distribution.loc.detach().cpu().numpy()
            ax.scatter(means[:, 0], means[:, 1], c='blue', marker='o', s=120, edgecolors='black', label='Component Means', zorder=11)

            locations, thicknesses = markers
            ax.scatter(locations, thicknesses, c='r', marker='x', s=100, label='Samples Drawn', zorder=10)
            legend = ax.legend(loc='upper right', frameon=True, framealpha=1.0)
            legend.set_zorder(12) # Ensure legend is on top
            
            ax.set_xlabel('Location', fontweight='bold', fontsize=fs)
            ax.set_ylabel('Thickness', fontweight='bold', fontsize=fs)
            ax.set_title('GMM with Samples', fontweight='bold', fontsize=fs)
            ax.tick_params(axis='both', which='major', labelsize=fs-2)
            plt.tight_layout()
            plt.savefig(f"{base_save_path}_markers.png")

            # save data as a file
            data = gmm_single, markers
            with open(f"{base_save_path}_data.pkl", "wb") as f:
                pickle.dump(data, f)
            plt.close()
