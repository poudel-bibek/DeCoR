import torch
import numpy as np
import torch.multiprocessing as mp
from torch_geometric.data import Batch
from scipy.special import legendre, roots_legendre  # Use roots_legendre for Gaussian quadrature

class WelfordNormalizer:
    def __init__(self, shape, eps=1e-8):
        """
        Online normalization (for both state and reward) using Welford's algorithm.
        Each lower level PPO actor uses its own copy of the old policy.
        """
        self.mean = torch.zeros(shape, dtype=torch.float32).share_memory_() # State normalizer needs to be shared between processes.
        self.M2 = torch.zeros(shape, dtype=torch.float32).share_memory_() 
        self.count = mp.Value('i', 0) # A variable i that is shared between processes and is init to 0.
        self.eps = eps
        self.lock = mp.Lock()
        self.training = True # Only update the normalizer when training is True.

    def eval(self,):
        self.training = False

    def manual_load(self, mean, M2, count):
        """
        During evaluation, load the normalizer stats.
        """
        self.mean.copy_(mean)
        self.M2.copy_(M2)
        self.count.value = count

    def update(self, x):
        """
        Update running statistics using a new sample x.
        """
        with self.lock:
            if self.count.value == 0:
                # First sample: initialize mean and zero-out M2.
                self.mean.copy_(x)
                self.M2.zero_()
                self.count.value = 1
            else:
                self.count.value += 1
                delta = x - self.mean
                self.mean.add_(delta / self.count.value)
                delta2 = x - self.mean
                self.M2.add_(delta * delta2)

    def variance(self):
        with self.lock:
            if self.count.value < 2:
                # Not enough samples: return a tensor of ones with the same shape as mean
                return torch.ones_like(self.mean)
            else:
                return self.M2 / (self.count.value - 1)

    def std(self):
        variance = self.variance()
        return torch.where(variance == 0, torch.ones_like(variance), torch.sqrt(variance) + self.eps)

    def normalize(self, x):
        """
        Normalize the sample x using the running stats. Handles both single samples and batches.
        - x (torch.Tensor or array-like): Input data, can be a single sample or a batch (dim 0).
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32) # Ensure x is a tensor

        original_shape = x.shape
        is_batch = len(original_shape) > len(self.mean.shape) # Check if input has more dims than mean

        if is_batch:
            # Reshape batch to (batch_size, *feature_shape)
            batch_size = original_shape[0]
            feature_shape = original_shape[1:]
            x_reshaped = x.view(batch_size, -1) # Flatten features if needed, adjust based on expected shape
            # If self.mean is 1D [features], and x is [batch, features], reshaping might not be needed
            # Check if feature shapes match
            if feature_shape != self.mean.shape:
                 # This case needs careful handling based on actual shapes.
                 # Assuming features are the last dimension matching self.mean.shape
                 if original_shape[-len(self.mean.shape):] == self.mean.shape:
                      x_reshaped = x.reshape(-1, *self.mean.shape) # Reshape to [N, *mean_shape]
                 else:
                      # Raise error or handle differently if shapes are incompatible
                      raise ValueError(f"Input shape {original_shape} incompatible with normalizer mean shape {self.mean.shape}")
            else:
                 x_reshaped = x # Use original if shapes already match [batch, *mean_shape]

            normalized_x = torch.zeros_like(x_reshaped)
            current_mean = self.mean.clone() # Use consistent mean/std for the whole batch
            current_std = self.std().clone()

            for i in range(x_reshaped.shape[0]):
                sample = x_reshaped[i]
                if self.training:
                    self.update(sample) # Update stats one sample at a time
                normalized_x[i] = (sample - current_mean) / current_std
            
            # Reshape back to original batch shape
            return normalized_x.view(original_shape)

        else: # Handle single sample
             # Ensure single sample shape matches mean shape
             if x.shape != self.mean.shape:
                  raise ValueError(f"Input shape {x.shape} incompatible with normalizer mean shape {self.mean.shape}")
             
             if self.training:
                 self.update(x)
             return (x - self.mean) / self.std()
    

class Memory:
    """
    For saving experiences.
    Memories will be made in CPU but loaded in GPU for the policy updates.
    """
    def __init__(self,):
        self.states = []
        self.actions = []
        self.num_proposals = []
        self.values = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        
    def append(self, state, action, num_proposals, value, logprob, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.num_proposals.append(num_proposals)
        self.values.append(value) 
        self.logprobs.append(logprob)
        self.rewards.append(reward) # scalars
        self.is_terminals.append(done) # scalars


class GraphDataset(torch.utils.data.Dataset):
    """
    Graph data loading helper for the higher-level agent
    """
    def __init__(self, states, actions, num_proposals, logprobs, advantages, returns, old_values):
        self.states = states 
        self.actions = actions
        self.num_proposals = num_proposals
        self.logprobs = logprobs
        self.advantages = advantages
        self.returns = returns
        self.old_values = old_values

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return (
            self.states[idx],
            self.actions[idx],
            self.num_proposals[idx],
            self.logprobs[idx],
            self.advantages[idx],
            self.returns[idx],
            self.old_values[idx]
        )

def collate_fn(data):
    """
    Collate function for the higher-level agent
    """
    states_batch, actions_batch, num_proposals_batch, old_logprobs_batch, advantages_batch, returns_batch, old_values_batch = zip(*data)
    # print(f"\nStates batch: {states_batch}")
    # print(f"\nActions batch: {actions_batch}")
    # print(f"\nOld logprobs batch: {old_logprobs_batch}")
    # print(f"\nAdvantages batch: {advantages_batch}")
    # print(f"\nReturns batch: {returns_batch}")
    # print(f"\nOld values batch: {old_values_batch}")
    states_batch = Batch.from_data_list(states_batch)  
    actions_batch = torch.stack(actions_batch, dim=0)
    num_proposals_batch = torch.stack(num_proposals_batch, dim=0)
    old_logprobs_batch = torch.stack(old_logprobs_batch, dim=0)

    # Stack advantages and returns (already scalar tensors)
    advantages_batch = torch.stack(advantages_batch)  # Shape is (batch_size,)
    returns_batch = torch.stack(returns_batch)        # Shape is (batch_size,)
    old_values_batch = torch.stack(old_values_batch)  # Shape is (batch_size,)
    # print(f"\nStates batch: {states_batch}")
    # print(f"\nActions batch: {actions_batch}")
    # print(f"\nNum proposals batch: {num_proposals_batch}")
    # print(f"\nOld logprobs batch: {old_logprobs_batch}")
    # print(f"\nAdvantages batch: {advantages_batch}")
    # print(f"\nReturns batch: {returns_batch}")
    # print(f"\nOld values batch: {old_values_batch}")
    return states_batch, actions_batch, num_proposals_batch, old_logprobs_batch, advantages_batch, returns_batch, old_values_batch


# 1. Legendre Quadrature 
# Helper Functions
_legendre_quadrature_cache = {}  # Cache quadrature points and weights

def _get_legendre_quadrature(order):
    """
    Gets (or computes and caches) Gauss-Legendre quadrature points and weights.
    Args:
        order (int): The quadrature order.
    Returns:
        tuple: (quadrature points, quadrature weights) as numpy arrays.
    """
    if order not in _legendre_quadrature_cache:
        points, weights = roots_legendre(order)
        _legendre_quadrature_cache[order] = (points, weights)
    return _legendre_quadrature_cache[order]


def _shifted_legendre(n, x):
    """
    Evaluates the n-th shifted Legendre polynomial at x.  Shifted to [0, 1].
    Args:
        n (int): The polynomial order.
        x (torch.Tensor): The points at which to evaluate the polynomial.
    Returns:
        torch.Tensor: The polynomial evaluated at x.
    """
    return torch.tensor(legendre(n)(2 * x - 1), dtype=torch.float32, device=x.device)


def gmm_entropy_legendre(gmm_single, order=10, a=None):
    """
    Calculates the entropy of a single GMM using the shifted Legendre polynomial
    approximation, as described in the NeurIPS 2023 paper "On Convergence
    of Polynomial Approximations to the Gaussian Mixture Entropy." This version
    uses Gaussian quadrature for improved accuracy and efficiency.

    Args:
        gmm_single (MixtureSameFamily): A single GMM distribution (not a batch).
        order (int): The order of the Legendre series approximation (N in the paper).
                     Higher order is more accurate but more computationally expensive.
        a (float, optional): The upper bound of the interval [0, a] for the shifted
            Legendre polynomials. If None, it's calculated based on the GMM parameters
            as an upper bound on the maximum density value.

    Returns:
        torch.Tensor: The approximate entropy (scalar).
    """

    # --- 1. Determine 'a' (if not provided) ---
    if a is None:
        # Use the upper bound on the maximum of the GMM (Theorem 4.4)
        with torch.no_grad():  # No gradients needed for calculating 'a'
            mix_probs = torch.softmax(gmm_single.mixture_distribution.logits, dim=-1)
            a = 0.0
            for i in range(gmm_single.mixture_distribution.logits.shape[0]):
                cov_det = torch.det(gmm_single.component_distribution.covariance_matrix[i])
                a += mix_probs[i] * (1 / torch.sqrt((2 * torch.pi)**gmm_single.component_distribution.loc.shape[-1] * cov_det))
            a = a.item()

    # --- 2. Gaussian Quadrature for E[p(x)^k] ---
    # We'll use Gauss-Legendre quadrature to approximate the integral of p(x)^k.
    # This is significantly more efficient and accurate than the closed-form
    # expression for higher orders, and avoids the multinomial coefficient.

    # Get quadrature points and weights (pre-computed and cached for efficiency)
    quad_points, quad_weights = _get_legendre_quadrature(order)
    quad_points = torch.tensor(quad_points, dtype=torch.float32, device=gmm_single.mixture_distribution.logits.device)
    quad_weights = torch.tensor(quad_weights, dtype=torch.float32, device=gmm_single.mixture_distribution.logits.device)


    # --- 3. Calculate the Legendre Series Sum ---
    entropy = 0.0
    for n in range(order + 1):
        inner_sum = 0.0
        for j in range(n + 1):
            term1 = (-1)**(n + j) * np.math.factorial(n + j)
            term2 = (j + 1) * np.log(a) - 1  # Use log(a) for numerical stability
            term3 = np.math.factorial(n - j) * (np.math.factorial(j + 1))**2
            inner_sum += (term1 * term2) / term3

        legendre_coeff_n = (2 * n + 1) * inner_sum

        # --- Gaussian Quadrature for E[L_[0,a],n(p(x))] ---
        # Instead of calculating E[p(x)^k] and then evaluating the Legendre
        # polynomial at those points, we directly compute E[L_[0,a],n(p(x))]
        # using Gaussian quadrature.
        shifted_points = quad_points * a  # Shift points to [0, a]

        # Evaluate p(x) at the shifted quadrature points
        with torch.no_grad(): # No gradient is needed
            log_prob_shifted_points = gmm_single.log_prob(shifted_points)
            p_x_at_points = torch.exp(log_prob_shifted_points)

        # Evaluate the n-th shifted Legendre polynomial at p(x)
        legendre_n = _shifted_legendre(n, p_x_at_points / a)  # Shifted to [0, 1]

        # Quadrature approximation of E[L_[0,a],n(p(x))]
        expectation_legendre_n = torch.sum(quad_weights * legendre_n)

        entropy -= legendre_coeff_n * expectation_legendre_n  # -= because of -log

    return entropy

# 2. Monte Carlo Sampling

def gmm_entropy_monte_carlo(gmm_single, num_samples=20000):
    """
    Calculates the entropy of a single GMM using Monte Carlo sampling.
    1. Sample a large number of points from the GMM
    2. Compute the log probability of each sample
    3. Take the negative mean of these log probabilities
    
    Monte Carlo with a large number of samples solves the variance problem in entropy estimation.
    However, it is still not differentiable.
    We use monte carlo with a large number of samples as "ground truth" to compare with Legendre Quadrature.
    """

    samples = gmm_single.sample((num_samples,))
    log_probs = gmm_single.log_prob(samples)
    entropy = -log_probs.mean() 
    return entropy