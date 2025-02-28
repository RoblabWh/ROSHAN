import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Bernoulli, Normal, Independent


class ActorCritic(nn.Module):
    """
    A PyTorch Module that represents the actor-critic network of a PPO agent.
    """
    def __init__(self, Actor, Critic, vision_range, map_size, time_steps):
        super(ActorCritic, self).__init__()
        self.actor_cnt = 0
        self.critic_cnt = 0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(vision_range, map_size, time_steps).to(self.device)
        self.critic = Critic(vision_range, map_size, time_steps).to(self.device)

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

            # Expand action_var to match batch size
            batch_size = action_mean.size(0)
            action_size = action_mean.size(1)
            action_var = action_var.unsqueeze(0).expand(batch_size, action_size)
            action_std = torch.sqrt(action_var)

            # Create independent normal distributions for each action dimension
            dist = Normal(action_mean, action_std)  # Shape: [batch_size, action_size]
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

    def evaluate(self, state, action):
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

        # Expand action_var to match action_mean
        batch_size = action_mean.size(0)
        action_size = action_mean.size(1)
        action_var = action_var.unsqueeze(0).expand(batch_size, action_size)  # Shape: [batch_size, action_size]
        action_std = torch.sqrt(action_var)  # Shape: [batch_size, action_size]

        # Create independent normal distributions for each action dimension
        dist = Normal(action_mean, action_std)  # Shape: [batch_size, action_size]
        dist = Independent(dist, 1)  # Treat the last dimension as the event dimension

        # Compute log probabilities of the taken actions
        action_logprob = dist.log_prob(action)  # action: [batch_size, action_size], action_logprob: [batch_size]

        # Compute the entropy of the distributions
        dist_entropy = dist.entropy()  # Shape: [batch_size]

        # Squeeze state_value if necessary
        state_value = torch.squeeze(state_value)  # Shape: [batch_size]

        return action_logprob, state_value, dist_entropy