import torch
import torch.nn as nn
import numpy as np
from utils import init_fn
from torch.distributions import MultivariateNormal, Bernoulli, Normal, Independent

class CategoricalActorCritic(nn.Module):
    """
    A PyTorch Module that represents the actor-critic network of a categorical agent.
    """
    def __init__(self, Actor, Critic, vision_range, drone_count, map_size, time_steps):
        super(CategoricalActorCritic, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(vision_range, drone_count, map_size, time_steps).to(self.device)
        self.actor.apply(init_fn)
        self.critic = Critic(vision_range, drone_count, map_size, time_steps).to(self.device)
        self.critic.apply(init_fn)

    def act(self, state):
        """
        Returns an action sampled from the actor's distribution and the log probability of that action.

        :param states: States of the Agents
        :return: A tuple of the sampled action and the log probability of that action.
        """
        with torch.no_grad():
            probs = self.actor(state)  # logits: [batch_size, num_drones, num_fires]
            cat_dist = torch.distributions.Categorical(probs=probs)

            actions_idx = cat_dist.sample()  # [batch_size, num_drones]
            action_logprob = cat_dist.log_prob(actions_idx)  # [batch_size, num_drones]

            _, _, possible_goals = state
            possible_goals = possible_goals.squeeze(1)

            B, N_D = actions_idx.shape
            batch_idx = torch.arange(B).unsqueeze(1).expand(-1, N_D)
            actions = possible_goals[batch_idx, actions_idx.cpu()]

            actions = actions.reshape(B, N_D, 2)

            return actions, action_logprob.detach().cpu().numpy()

    def act_certain(self, state):
        """
        Returns an action from the actor's distribution without sampling.

        :param states: States of the Agents
        :return: The action from the actor's distribution.
        """
        with torch.no_grad():
            action_logits = self.actor(state)
            # Shape: [batch_size, action_size]
            actions_idx = torch.argmax(action_logits, dim=-1)
            _, _, possible_goals = state# [batch_size, num_drones]
            possible_goals = possible_goals.squeeze(1)

            B, N_D = actions_idx.shape
            batch_idx = torch.arange(B).unsqueeze(1).expand(-1, N_D)
            actions = possible_goals[batch_idx, actions_idx.cpu()]

            actions = actions.reshape(B, N_D, 2)
            return actions

    def evaluate(self, state, actions, masks=None):
        """
        Returns the log probability of the given action, the value of the given state, and the entropy of the actor's
        distribution.

        :param state: A tuple of the current lidar scan, orientation to goal, distance to goal, and velocity.
        :param action: The action to evaluate.
        :return: A tuple of the log probability of the given action, the value of the given state, and the entropy of the
        actor's distribution.
        """

        state_value = self.critic(state, masks)  # Shape: [batch_size, 1]
        # state_value = torch.squeeze(state_value)

        probs = self.actor(state, masks)  # logits: [batch_size, num_drones, num_fires]
        cat_dist = torch.distributions.Categorical(probs)
        # Sample action for each drone
        actions = cat_dist.sample()  # [batch_size, num_drones]
        # Log probability for each action (per drone)
        action_logprob = cat_dist.log_prob(actions)  # [batch_size, num_drones]
        dist_entropy = cat_dist.entropy()  # [B, D]

        action_logprob = action_logprob.mean(dim=1)  # [B]
        dist_entropy = dist_entropy.mean(dim=1)  # [B]
        state_value = torch.squeeze(state_value.mean(dim=1))  # [B]

        return action_logprob, state_value, dist_entropy

class StochasticActor(nn.Module):
    """
    A PyTorch Module that represents the actor-critic network of a PPO agent.
    """
    def __init__(self, Actor, Critic, vision_range, drone_count, map_size, time_steps):
        super(StochasticActor, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(vision_range, drone_count, map_size, time_steps).to(self.device)
        self.actor.apply(init_fn)

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

            # Move Tensors to CPU
            action_mean = action_mean.cpu()
            action_var = action_var.cpu()

            # Create independent normal distributions for each action dimension
            # dist = Normal(action_mean, action_std)  # Shape: [batch_size, action_size]
            dist = Normal(action_mean, torch.sqrt(action_var))  # Shape: [batch_size, action_size] Use PyTorch's built-in broadcasting
            dist = Independent(dist, 1)  # Treat the last dimension as the event dimension

            # Sample actions from the distributions
            action = dist.sample()  # Shape: [batch_size, 3]

            # Clip actions to the valid range
            action = torch.clamp(action, -1, 1)

            # Compute log probabilities of the sampled actions
            action_logprob = dist.log_prob(action)

            return action.detach().numpy(), action_logprob.detach().numpy()

    def act_certain(self, state):
        """
        Returns an action from the actor's distribution without sampling.

        :param states: A tuple of the current lidar scan, orientation to goal, distance to goal, and velocity.
        :return: The action from the actor's distribution.
        """
        with torch.no_grad():
            action_mean, _ = self.actor(state)

        return action_mean.detach().cpu().numpy()

class DeterministicActorCritic(nn.Module):
    """
    A PyTorch Module that represents the actor-critic network of a deterministic agent.
    """
    def __init__(self, Actor, Critic, action_dim, exploration_noise, vision_range, drone_count, map_size, time_steps):
        super(DeterministicActorCritic, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(vision_range, drone_count, map_size, time_steps).to(self.device)
        self.actor.apply(init_fn)
        self.critic = Critic(vision_range, drone_count, map_size, time_steps, action_dim).to(self.device)
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
    def __init__(self, Actor, Critic, vision_range, drone_count, map_size, time_steps, share_encoder=False):
        super(ActorCriticPPO, self).__init__(Actor, Critic, vision_range, drone_count, map_size, time_steps)

        inputspace = None if not share_encoder else self.actor.Inputspace

        self.critic = Critic(vision_range, drone_count, map_size, time_steps, inputspace).to(self.device)
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
        dist = Normal(action_mean, torch.sqrt(action_var))  # Shape: [batch_size, action_size] Use PyTorch's built-in broadcasting
        dist = Independent(dist, 1)  # Treat the last dimension as the event dimension

        # Compute log probabilities of the taken actions
        action_logprob = dist.log_prob(action)  # action: [batch_size, action_size], action_logprob: [batch_size]

        # Compute the entropy of the distributions
        dist_entropy = dist.entropy()  # Shape: [batch_size]

        # Squeeze state_value if necessary
        state_value = torch.squeeze(state_value)  # Shape: [batch_size]

        return action_logprob, state_value, dist_entropy

class ActorCriticIQL(StochasticActor):
    """
    A PyTorch Module that represents the actor-critic network of an IQL agent.
    """
    def __init__(self, Actor, Critic, Value, action_dim, vision_range, drone_count, map_size, time_steps):
        super(ActorCriticIQL, self).__init__(Actor, Critic, vision_range, drone_count, map_size, time_steps)

        self.critic = Critic(vision_range, drone_count, map_size, time_steps, action_dim).to(self.device)
        self.critic.apply(init_fn)
        self.value = Value(vision_range, drone_count, map_size, time_steps).to(self.device)
        self.value.apply(init_fn)

    def get_logprobs(self, states, actions):
        # Get action means and variances from the actor network
        action_mean, action_var = self.actor(states)  # action_mean: [batch_size, action_size], action_var: [action_size, ]

        # Expand action_var to match action_mean
        batch_size = action_mean.size(0)
        action_size = action_mean.size(1)
        action_var = action_var.unsqueeze(0).expand(batch_size, action_size)  # Shape: [batch_size, action_size]
        action_std = torch.sqrt(action_var)  # Shape: [batch_size, action_size]

        # Create independent normal distributions for each action dimension
        dist = Normal(action_mean, action_std)  # Shape: [batch_size, action_size]
        dist = Independent(dist, 1)  # Treat the last dimension as the event dimension

        # Compute log probabilities of the taken actions
        action_logprob = dist.log_prob(actions)  # action: [batch_size, action_size], action_logprob: [batch_size]

        return action_logprob