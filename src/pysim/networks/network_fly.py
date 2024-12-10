import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Bernoulli, Normal, Independent
import torch
from utils import initialize_hidden_weights, initialize_output_weights, get_in_features_2d, get_in_features_3d

torch.autograd.set_detect_anomaly(True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Inputspace(nn.Module):

    def __init__(self, vision_range, time_steps):
        """
        A PyTorch Module that represents the input space of a neural network.
        """
        super(Inputspace, self).__init__()

        self.vision_range = vision_range
        self.time_steps = time_steps

        vel_out_features = 16
        self.vel_dense1 = nn.Linear(in_features=2, out_features=vel_out_features)
        initialize_hidden_weights(self.vel_dense1)

        goal_out_features = 8
        self.goal_dense1 = nn.Linear(in_features=2, out_features=goal_out_features)
        initialize_hidden_weights(self.goal_dense1)

        self.flatten = nn.Flatten()

        input_features = (vel_out_features + goal_out_features) * self.time_steps
        self.out_features = 32
        mid_features = 64

        self.pos_dense1 = nn.Linear(in_features=input_features, out_features=mid_features)
        initialize_hidden_weights(self.pos_dense1)
        self.pos_dense2 = nn.Linear(in_features=mid_features, out_features=self.out_features)
        initialize_hidden_weights(self.pos_dense2)

    @staticmethod
    def prepare_tensor(states):
        velocity, position = states

        if isinstance(velocity, np.ndarray):
            velocity = torch.tensor(velocity, dtype=torch.float32).to(device)

        if isinstance(position, np.ndarray):
            position = torch.tensor(position, dtype=torch.float32).to(device)

        return velocity, position

    def forward(self, states):
        velocity, goal = self.prepare_tensor(states)

        velocity = F.relu(self.vel_dense1(velocity))
        velocity = torch.flatten(velocity, start_dim=1)

        goal = F.relu(self.goal_dense1(goal))
        goal = torch.flatten(goal, start_dim=1)

        concat_tensor = torch.cat((velocity, goal), dim=1)

        concat_tensor = F.relu(self.pos_dense1(concat_tensor))
        concat_tensor = F.relu(self.pos_dense2(concat_tensor))
        output_vision = torch.flatten(concat_tensor, start_dim=1)

        return output_vision


class Actor(nn.Module):
    """
    A PyTorch Module that represents the actor network of a PPO agent.
    """
    def __init__(self, vision_range, time_steps):
        super(Actor, self).__init__()
        self.Inputspace = Inputspace(vision_range, time_steps=time_steps)
        self.in_features = self.Inputspace.out_features
        # Mu
        self.mu_move = nn.Linear(in_features=self.in_features, out_features=2)
        initialize_output_weights(self.mu_move, 'actor')

        # Logstd
        self.log_std = nn.Parameter(torch.zeros(2, ))

    def forward(self, states):
        x = self.Inputspace(states)
        mu_move = torch.tanh(self.mu_move(x))
        std = torch.exp(self.log_std)
        var = torch.pow(std, 2)

        return mu_move, var


class Critic(nn.Module):
    """
    A PyTorch Module that represents the critic network of a PPO agent.
    """
    def __init__(self, vision_range, time_steps):
        super(Critic, self).__init__()
        self.Inputspace = self.Inputspace = Inputspace(vision_range, time_steps=time_steps)
        self.in_features = self.Inputspace.out_features

        # Value
        self.value = nn.Linear(in_features=self.in_features, out_features=1)
        initialize_output_weights(self.value, 'critic')

    def forward(self, states):
        x = self.Inputspace(states)
        value = self.value(x)
        return value


class ActorCritic(nn.Module):
    """
    A PyTorch Module that represents the actor-critic network of a PPO agent.
    """
    def __init__(self, vision_range, time_steps):
        super(ActorCritic, self).__init__()
        self.actor_cnt = 0
        self.critic_cnt = 0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(vision_range, time_steps).to(device)
        self.critic = Critic(vision_range,time_steps).to(device)

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