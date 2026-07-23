import torch
import torch.nn as nn
import numpy as np
from utils import init_fn, get_device
from torch.distributions import MultivariateNormal, Bernoulli, Normal, Independent, TransformedDistribution
from torch.distributions.transforms import TanhTransform

class CategoricalActorCritic(nn.Module):
    """
    A PyTorch Module that represents the actor-critic network of a categorical agent.
    """
    def __init__(self, actor_network, critic_network, vision_range, drone_count, map_size, time_steps, manual_decay, share_encoder):
        super(CategoricalActorCritic, self).__init__()
        self.device = get_device()
        self.actor = actor_network(vision_range, drone_count, map_size, time_steps, manual_decay).to(self.device)
        self.actor.apply(init_fn)
        inputspace = None if not share_encoder else self.actor.Inputspace
        self.critic = critic_network(vision_range, drone_count, map_size, time_steps, inputspace).to(self.device)
        self.critic.apply(init_fn)

    @staticmethod
    def _get_possible_goals(state):
        """Extract fire/goal positions from dict-based or tuple-based state."""
        if hasattr(state, 'keys') or isinstance(state, dict):
            return state["fire_positions"]
        return state[2]

    def _idx_to_coords(self, state, actions_idx):
        """Map fire indices to (x,y) goal coordinates."""
        possible_goals = self._get_possible_goals(state)
        possible_goals = possible_goals.squeeze(1)
        B, N_D = actions_idx.shape
        batch_idx = torch.arange(B).unsqueeze(1).expand(-1, N_D)
        actions = possible_goals[batch_idx, actions_idx.cpu()]
        return actions.reshape(B, N_D, 2)

    def _coords_to_idx(self, state, actions):
        """Rebuild fire indices from (x,y) coordinates via nearest-neighbor matching."""
        possible_goals = self._get_possible_goals(state)
        possible_goals = possible_goals.squeeze(1)
        dists = torch.cdist(actions, possible_goals)  # [B, N_D, N_G]
        actions_idx = dists.argmin(dim=-1)
        min_d = dists.gather(-1, actions_idx.unsqueeze(-1)).squeeze(-1)
        if torch.any(min_d > 1.0e-8):
            raise RuntimeError("Action doesn't match any possible goal within tolerance.")
        return actions_idx

    def act(self, state):
        """
        Returns an action sampled from the actor's autoregressive distribution and the log probability of that action.
        """
        with torch.no_grad():
            actions_idx, log_probs, _ = self.actor(state)  # autoregressive sampling
            actions = self._idx_to_coords(state, actions_idx)
            return actions, log_probs.detach().cpu().numpy()

    def act_certain(self, state):
        """
        Returns a deterministic action from the actor's autoregressive distribution (argmax).
        """
        with torch.no_grad():
            actions_idx, _, _ = self.actor(state, deterministic=True)
            actions = self._idx_to_coords(state, actions_idx)
            return actions

    def evaluate(self, state, actions, masks=None):
        """
        Returns per-drone log probability, the state value, and per-drone entropy.

        Per-drone tensors are returned (NOT pre-summed over the drone dim) so that
        PPO can apply a per-drone ``locked_mask`` to zero out drones whose actions
        were structurally overridden via PlannerAgent.apply_commitment. PPO is
        responsible for the masked sum: ``(per_drone * (1 - locked)).sum(dim=1)``.
        """
        state_value = self.critic(state, masks)  # (B, 1)

        # Rebuild indices from coordinates
        actions_idx = self._coords_to_idx(state, actions)

        # Independent Categorical per drone (current PointerActor) — actions_idx
        # parameter forces evaluation at the given indices instead of resampling.
        _, action_logprob, dist_entropy = self.actor(state, masks, actions_idx=actions_idx)
        # action_logprob: (B, N), dist_entropy: (B, N) — keep per-drone shape.
        state_value = torch.squeeze(state_value)      # (B,)

        return action_logprob, state_value, dist_entropy

class StochasticActor(nn.Module):
    """
    A PyTorch Module that represents the actor-critic network of a PPO agent.
    """
    def __init__(self, actor_network, vision_range, drone_count, map_size, time_steps, manual_decay, use_tanh_dist, collision, agent_dim=9, neighbor_dim=4):
        super(StochasticActor, self).__init__()
        self.device = get_device()
        self.use_tanh_dist = use_tanh_dist
        self.actor = actor_network(vision_range=vision_range,
                                   drone_count=drone_count,
                                   map_size=map_size,
                                   time_steps=time_steps,
                                   manual_decay=manual_decay,
                                   use_tanh_dist=use_tanh_dist,
                                   collision=collision,
                                   agent_dim=agent_dim,
                                   neighbor_dim=neighbor_dim).to(self.device)
        self.actor.apply(init_fn)

    def get_distribution(self, action_mean, action_std):
        # Create independent normal distributions for each action dimension
        if self.use_tanh_dist:
            dist = TransformedDistribution(Normal(action_mean, action_std),
                                           [TanhTransform(cache_size=1)])  # Tanh-squashed normal distribution
        else:
            dist = Normal(action_mean, action_std)
        dist = Independent(dist, 1)  # Treat the last dimension as the event dimension
        return dist

    def act(self, state):
        """
        Returns an action sampled from the actor's distribution and the log probability of that action.

        :param states: A tuple of the current lidar scan, orientation to goal, distance to goal, and velocity.
        :return: A tuple of the sampled action and the log probability of that action.
        """

        # TODO: check if normalization of states is necessary
        # was suggested in: Implementation_Matters in Deep RL: A Case Study on PPO and TRPO
        with torch.no_grad():
            action_mean, action_var = self.actor(state)

            # Keep computation on GPU for consistency with evaluate()
            # (CPU vs GPU numerical differences cause first batch ratios != 1.0)
            dist = self.get_distribution(action_mean, torch.sqrt(action_var))

            # Sample actions from the distributions
            # action = dist.sample()  # Shape: [batch_size, 3]
            action = dist.rsample()

            # # Clip actions to the valid range
            ### Apparently clipping here destroys the logprobs
            ### Tanh-squashed distribution takes care of that
            if not self.use_tanh_dist:
                action = torch.clamp(action, -1, 1)

            # Compute log probabilities of the sampled actions
            action_logprob = dist.log_prob(action)

            # Only convert to CPU/numpy at the very end
            return action.detach().cpu().numpy(), action_logprob.detach().cpu().numpy()

    def act_certain(self, state):
        """
        Returns an action from the actor's distribution without sampling.

        :param states: A tuple of the current lidar scan, orientation to goal, distance to goal, and velocity.
        :return: The action from the actor's distribution.
        """
        with torch.no_grad():
            action_mean, _ = self.actor(state)

        return torch.tanh(action_mean).detach().cpu().numpy() if self.use_tanh_dist else action_mean.detach().cpu().numpy()

class DeterministicActorCritic(nn.Module):
    """
    A PyTorch Module that represents the actor-critic network of a deterministic agent.
    """
    def __init__(self, actor_network, critic_network, action_dim, exploration_noise, vision_range, drone_count, map_size, time_steps, collision, share_encoder, agent_dim=9, neighbor_dim=4):
        super(DeterministicActorCritic, self).__init__()
        self.device = get_device()
        self.actor = actor_network(vision_range=vision_range,
                                   drone_count=drone_count,
                                   map_size=map_size,
                                   time_steps=time_steps,
                                   collision=collision,
                                   agent_dim=agent_dim,
                                   neighbor_dim=neighbor_dim).to(self.device)
        self.actor.apply(init_fn)
        inputspace = None if not share_encoder else self.actor.Inputspace
        self.critic = critic_network(vision_range=vision_range,
                                     drone_count=drone_count,
                                     map_size=map_size,
                                     time_steps=time_steps,
                                     action_dim=action_dim,
                                     inputspace=inputspace,
                                     collision=collision,
                                     agent_dim=agent_dim,
                                     neighbor_dim=neighbor_dim).to(self.device)
        self.critic.apply(init_fn)
        self.exploration_noise = exploration_noise

    def act(self, state):
        """
        Returns an action from the actor.

        :param state: A tuple of the current state
        :return: The action from the actor
        """
        with torch.no_grad():
            action = self.actor(state).detach().cpu().numpy()
            noise = np.random.normal(0, self.exploration_noise, size=action.shape)
            action += noise
            action = np.clip(action, -1, 1)

        # No log probability for deterministic actions, therefore return None
        return action, None

    def act_certain(self, state):
        """
        Returns an action from the actor without noise.

        :param state: A tuple of the current state
        :return: The action from the actor
        """
        with torch.no_grad():
            action = self.actor(state)

        return action.detach().cpu().numpy()

class ActorCriticPPO(StochasticActor):
    """
    A PyTorch Module that represents the actor-critic network of a PPO agent.
    """
    def __init__(self, actor_network, critic_network, vision_range, drone_count, map_size, time_steps, share_encoder, manual_decay, use_tanh_dist, collision, agent_dim=9, neighbor_dim=4):
        super(ActorCriticPPO, self).__init__(actor_network=actor_network,
                                             vision_range=vision_range,
                                             drone_count=drone_count,
                                             map_size=map_size,
                                             time_steps=time_steps,
                                             manual_decay=manual_decay,
                                             use_tanh_dist=use_tanh_dist,
                                             collision=collision,
                                             agent_dim=agent_dim,
                                             neighbor_dim=neighbor_dim)

        inputspace = None if not share_encoder else self.actor.Inputspace

        self.critic = critic_network(vision_range=vision_range,
                                     drone_count=drone_count,
                                     map_size=map_size,
                                     time_steps=time_steps,
                                     inputspace=inputspace,
                                     collision=collision,
                                     agent_dim=agent_dim,
                                     neighbor_dim=neighbor_dim).to(self.device)
        self.critic.apply(init_fn)

    def evaluate(self, state, action, masks=None):
        """
        Returns the log probability of the given action, the value of the given state, and the entropy of the actor's
        distribution.

        :param state: A tuple of the current lidar scan, orientation to goal, distance to goal, and velocity.
        :param action: The action to evaluate.
        :return: A tuple of the log probability of the given action, the value of the given state, and the entropy of the
        actor's distribution.
        """
        # Evaluate the state value from the critic network
        state_value = self.critic(state)  # Shape: [batch_size, 1]

        # Get action means and variances from the actor network
        action_mean, action_var = self.actor(state)  # action_mean: [batch_size, action_size], action_var: [action_size, ]

        # Create independent normal distributions for each action dimension
        dist = self.get_distribution(action_mean, torch.sqrt(action_var))

        if self.use_tanh_dist:
            # For Tanh-squashed normal distribution, clamp actions to avoid NaNs in log_prob due to intervals
            action = action.clamp(-1 + 1e-6, 1 - 1e-6)
        # Compute log probabilities of the taken actions
        action_logprob = dist.log_prob(action)  # action: [batch_size, action_size], action_logprob: [batch_size]

        # Compute the entropy of the distributions
        if self.use_tanh_dist:
            # The Tanh-squashed normal has no closed-form entropy. The base-normal
            # entropy (0.5*log(2*pi*e) + log_std) omits the tanh Jacobian term
            # E[log(1 - tanh(z)^2)], which becomes increasingly negative as std
            # grows (samples saturate toward +/-1). Without it, the entropy bonus
            # gives log_std a constant, never-saturating upward gradient and the
            # policy parks at an inflated std (bang-bang actions). Estimate the
            # true entropy via a single reparameterized sample (SAC-style):
            #   H = -E[log p(x)],  x ~ dist.
            # dist is the Independent-wrapped TransformedDistribution, so log_prob
            # already sums over the action dims -> shape (B,), matching the
            # joint-entropy scale PPO expects (no entropy_coeff rescale needed).
            sampled_action = dist.rsample()
            dist_entropy = -dist.log_prob(sampled_action)
        else:
            dist_entropy = dist.entropy()

        # Squeeze state_value if necessary
        state_value = torch.squeeze(state_value)  # Shape: [batch_size]

        return action_logprob, state_value, dist_entropy

class ActorCriticIQL(StochasticActor):
    """
    A PyTorch Module that represents the actor-critic network of an IQL agent.
    """
    def __init__(self, actor_network, critic_network, value_network, action_dim, vision_range, drone_count, map_size, time_steps, share_encoder, use_tanh_dist, collision, agent_dim=9, neighbor_dim=4):
        super(ActorCriticIQL, self).__init__(actor_network=actor_network,
                                             vision_range=vision_range,
                                             drone_count=drone_count,
                                             map_size=map_size,
                                             time_steps=time_steps,
                                             use_tanh_dist=use_tanh_dist,
                                             collision=collision,
                                             manual_decay=False,
                                             agent_dim=agent_dim,
                                             neighbor_dim=neighbor_dim)

        self._fixed_log_std = nn.Parameter(
            torch.full((2,), -3.0),  # std ≈ 0.05
            requires_grad=False
        )

        inputspace = None if not share_encoder else self.actor.Inputspace

        self.critic = critic_network(vision_range=vision_range,
                                     drone_count=drone_count,
                                     map_size=map_size,
                                     time_steps=time_steps,
                                     action_dim=action_dim,
                                     inputspace=inputspace,
                                     collision=collision,
                                     agent_dim=agent_dim,
                                     neighbor_dim=neighbor_dim).to(self.device)
        self.critic.apply(init_fn)
        self.value = value_network(vision_range=vision_range,
                                   drone_count=drone_count,
                                   map_size=map_size,
                                   time_steps=time_steps,
                                   inputspace=inputspace,
                                   collision=collision,
                                   agent_dim=agent_dim,
                                   neighbor_dim=neighbor_dim).to(self.device)
        self.value.apply(init_fn)

    def act(self, state):
        """
        Returns an action sampled from the actor's distribution and the log probability of that action.

        :param state: A tuple of the current lidar scan, orientation to goal, distance to goal, and velocity.
        :return: A tuple of the sampled action and the log probability of that action.
        """

        with torch.no_grad():
            action_mean, _ = self.actor(state)

            # Move Tensor to CPU
            action_mean = action_mean.cpu()

            # Generate the Raw and Noised Action; the noise is used for exploration during data collection
            # Clean action without noise is also returned for computing log probabilities during training
            raw_action = torch.tanh(action_mean) if self.use_tanh_dist else action_mean  # [-1,1] if tanh
            # TODO: Make this a parameter
            action_sigma = 0.05
            if action_sigma and action_sigma > 0:
                eps = torch.randn_like(raw_action) * action_sigma
                noised_action = (raw_action + eps).clamp(-1 + 1e-6, 1 - 1e-6)
            else:
                noised_action = raw_action.clamp(-1 + 1e-6, 1 - 1e-6)

            return noised_action.detach().numpy(), raw_action.detach().numpy()

    def get_logprobs(self, states, actions):
        # Get action means and variances from the actor network
        action_mean, _ = self.actor(states)  # action_mean: [batch_size, action_size], action_var: irrelevant

        # fixed small std
        action_std = self._fixed_log_std.exp().expand_as(action_mean).to(self.device)

        # Expand action_var to match action_mean
        # batch_size = action_mean.size(0)
        # action_size = action_mean.size(1)
        # action_var = action_var.unsqueeze(0).expand(batch_size, action_size)  # Shape: [batch_size, action_size]
        # action_std = torch.sqrt(action_var)  # Shape: [batch_size, action_size]

        # Create independent normal distributions for each action dimension
        dist = self.get_distribution(action_mean, action_std)

        if self.use_tanh_dist:
            # For Tanh-squashed normal distribution, clamp actions to avoid NaNs in log_prob due to intervals
            actions = actions.clamp(-1 + 1e-6, 1 - 1e-6)
        # Compute log probabilities of the taken actions
        action_logprob = dist.log_prob(actions)  # action: [batch_size, action_size], action_logprob: [batch_size]

        return action_logprob