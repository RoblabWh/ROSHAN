import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Bernoulli, Normal, Independent
import torch

torch.autograd.set_detect_anomaly(True)

from utils import initialize_hidden_weights, initialize_output_weights

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
        # initialize_hidden_weights(self.vel_dense1)

        position_out_features = 16
        self.position_dense1 = nn.Linear(in_features=2, out_features=position_out_features)
        # initialize_hidden_weights(self.position_dense1)

        goal_out_features = 16
        self.goal_dense1 = nn.Linear(in_features=2, out_features=goal_out_features)

        self.flatten = nn.Flatten()

        input_features = (position_out_features + vel_out_features + goal_out_features) * self.time_steps
        self.out_features = 128
        #self.pos_dense1 = nn.Linear(in_features=input_features, out_features=16)
        self.pos_dense1 = nn.Linear(in_features=input_features, out_features=self.out_features)
        # self.vic_dense1 = nn.Linear(in_features=vicinity_features, out_features=32)
        # self.vic_dense2 = nn.Linear(in_features=32, out_features=16)

    def get_in_features_2d(self, h_in, w_in, layers_dict):
        for layer in layers_dict:
            padding = layer['padding']
            dilation = layer['dilation']
            kernel_size = layer['kernel_size']
            stride = layer['stride']

            h_in = ((h_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0]) + 1
            w_in = ((w_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1]) + 1

        return h_in * w_in

    def get_in_features_3d(self, h_in, w_in, d_in, layers_dict):
        for layer in layers_dict:
            padding = layer['padding']
            dilation = layer['dilation']
            kernel_size = layer['kernel_size']
            stride = layer['stride']

            d_in = ((d_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0]) + 1
            h_in = ((h_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1]) + 1
            w_in = ((w_in + 2 * padding[2] - dilation[2] * (kernel_size[2] - 1) - 1) // stride[2]) + 1

        return d_in * h_in * w_in

    def prepare_tensor(self, states):
        velocity, position, goal = states

        if isinstance(velocity, np.ndarray):
            velocity = torch.tensor(velocity, dtype=torch.float32).to(device)

        if isinstance(position, np.ndarray):
            position = torch.tensor(position, dtype=torch.float32).to(device)

        if isinstance(goal, np.ndarray):
            goal = torch.tensor(goal, dtype=torch.float32).to(device)

        return velocity, position, goal

    def forward(self, states):
        velocity, position, goal = self.prepare_tensor(states)

        position = F.relu(self.position_dense1(position))
        position = torch.flatten(position, start_dim=1)

        velocity = F.relu(self.vel_dense1(velocity))
        velocity = torch.flatten(velocity, start_dim=1)

        goal = F.relu(self.goal_dense1(goal))
        goal = torch.flatten(goal, start_dim=1)

        concat_pos = torch.cat((position, velocity, goal), dim=1)
        # concat_vic = torch.cat((view, water), dim=1)
        pos_feature = F.relu(self.pos_dense1(concat_pos))
        output_vision = torch.flatten(pos_feature, start_dim=1)

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
        self.mu_water = nn.Linear(in_features=self.in_features, out_features=1)

        # initialize_output_weights(self.mu_move, 'actor')
        # initialize_output_weights(self.mu_water, 'actor')

        # Logstd
        self.log_std = nn.Parameter(torch.zeros(3, ))

    def forward(self, states):
        x = self.Inputspace(states)
        mu_move = torch.tanh(self.mu_move(x))
        mu_water = torch.tanh(self.mu_water(x))
        actions = torch.cat((mu_move, mu_water), dim=1)
        std = torch.exp(self.log_std)
        var = torch.pow(std, 2)

        return actions, var


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
        # initialize_output_weights(self.value, 'critic')

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

        # TODO statische var testen
        #self.logstds_param = nn.Parameter(torch.full((n_actions,), 0.1))
        #self.action_var = torch.full((action_dim, ), action_std * action_std).to(device)

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
            action_var = action_var.unsqueeze(0).expand(batch_size, 3)
            action_std = torch.sqrt(action_var)

            # Create independent normal distributions for each action dimension
            dist = Normal(action_mean, action_std)  # Shape: [batch_size, 3]
            dist = Independent(dist, 1)  # Treat the last dimension as the event dimension

            # Sample actions from the distributions
            action = dist.sample()  # Shape: [batch_size, 3]

            # Clip actions to the valid range
            action = torch.clamp(action, -1, 1)

            # Compute log probabilities of the sampled actions
            action_logprob = dist.log_prob(action)

            # action_mean_velocity = action_mean[:, :2].to('cpu')
            # action_var_velocity = action_var[:2].to('cpu')
            # action_mean_water_emit = action_mean[:, 2].to('cpu')
            # action_var_water_emit = action_var[2].to('cpu')
            #
            # cov_mat = torch.diag(action_var_velocity)
            # dist_velocity = MultivariateNormal(action_mean_velocity, cov_mat)
            # dist_water = Normal(action_mean_water_emit, action_var_water_emit)
            # ## logging of actions
            # #self.logger.add_actor_output(action_mean.mean(0)[0].item(), action_mean.mean(0)[1].item(), action_var[0].item(), action_var[1].item())
            #
            # action_velocity = dist_velocity.sample()
            # action_velocity = torch.clip(action_velocity, -1, 1)
            # action_water = dist_water.sample().clip(-1, 1)
            #
            # action_logprob_velocity = dist_velocity.log_prob(action_velocity)
            # action_logprob_water = dist_water.log_prob(action_water)
            # action = torch.cat([action_velocity, action_water.unsqueeze(dim=1)], dim=1)
            # combined_logprob = action_logprob_velocity + action_logprob_water

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
        action_mean, action_var = self.actor(state)  # action_mean: [batch_size, 3], action_var: [3, ]

        # Expand action_var to match action_mean
        batch_size = action_mean.size(0)
        action_var = action_var.unsqueeze(0).expand(batch_size, 3)  # Shape: [batch_size, 3]
        action_std = torch.sqrt(action_var)  # Shape: [batch_size, 3]

        # Create independent normal distributions for each action dimension
        dist = Normal(action_mean, action_std)  # Shape: [batch_size, 3]
        dist = Independent(dist, 1)  # Treat the last dimension as the event dimension

        # Compute log probabilities of the taken actions
        action_logprob = dist.log_prob(action)  # action: [batch_size, 3], action_logprob: [batch_size]

        # Compute the entropy of the distributions
        dist_entropy = dist.entropy()  # Shape: [batch_size]

        # Squeeze state_value if necessary
        state_value = torch.squeeze(state_value)  # Shape: [batch_size]

        return action_logprob, state_value, dist_entropy

        # state_value = self.critic(state)
        #
        # action_mean, action_var = self.actor(state)
        #
        # action_mean_velocity = action_mean[:, :2]
        # action_var_velocity = action_var[:2]
        # action_mean_water_emit = action_mean[:, 2]
        # action_var_water_emit = action_var[2]
        #
        # cov_mat = torch.diag(action_var_velocity)
        # dist_velocity = MultivariateNormal(action_mean_velocity, cov_mat)
        # dist_water = Normal(action_mean_water_emit, action_var_water_emit)
        #
        # action_logprob_velocity = dist_velocity.log_prob(action[:, :2])
        # action_logprob_water = dist_water.log_prob(action[:, 2])
        # # action_logprob_water = dist_water.log_prob(action[:, 2].view(1, -1))
        # combined_logprob = action_logprob_velocity + action_logprob_water
        #
        # dist_entropy_velocity = dist_velocity.entropy()
        # dist_entropy_water = dist_water.entropy()
        # combined_entropy = dist_entropy_velocity + dist_entropy_water
        #
        # return combined_logprob.to(device), torch.squeeze(state_value), combined_entropy.to(device)
