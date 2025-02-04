import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from utils import initialize_output_weights, get_in_features_2d, get_in_features_3d

torch.autograd.set_detect_anomaly(True)

class Inputspace(nn.Module):

    def __init__(self, vision_range, time_steps):
        """
        A PyTorch Module that represents the input space of a neural network.
        """
        super(Inputspace, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.vision_range = vision_range
        self.time_steps = time_steps

        mid_layer_out_features = 64
        self.hidden_layer1 = nn.Linear(in_features=4, out_features=mid_layer_out_features)
        initialize_output_weights(self.hidden_layer1, 'hidden')
        self.hidden_layer2 = nn.Linear(in_features=mid_layer_out_features, out_features=mid_layer_out_features)
        initialize_output_weights(self.hidden_layer2, 'hidden')
        self.out_features = mid_layer_out_features * self.time_steps

        self.flatten = nn.Flatten()

    def prepare_tensor(self, states):
        velocity, position = states

        if isinstance(velocity, np.ndarray):
            velocity = torch.tensor(velocity, dtype=torch.float32).to(self.device)

        if isinstance(position, np.ndarray):
            position = torch.tensor(position, dtype=torch.float32).to(self.device)

        return velocity, position

    def forward(self, states):
        velocity, goal = self.prepare_tensor(states)

        delta_position = goal - velocity
        x = torch.cat((velocity, delta_position), dim=-1)
        x = F.relu(self.hidden_layer1(x))
        x = F.relu(self.hidden_layer2(x))
        output_vision = torch.flatten(x, start_dim=1)

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