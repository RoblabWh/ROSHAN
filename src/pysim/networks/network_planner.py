import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from utils import initialize_output_weights, get_in_features_2d, get_in_features_3d

torch.autograd.set_detect_anomaly(True)

class Inputspace(nn.Module):

    def __init__(self, drone_dim, time_steps):
        """
        A PyTorch Module that represents the input space of a neural network.
        """
        super(Inputspace, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.drone_dim = drone_dim
        hidden_dim = 64
        self.drone_embed = nn.Linear(2, hidden_dim)
        self.fire_embed = nn.Linear(2, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)

        self.out_features = hidden_dim


    def prepare_tensor(self, states, mask=None):
        drone_states, fire_states = states

        if not torch.is_tensor(drone_states):
            drone_states = torch.as_tensor(drone_states, dtype=torch.float32)

        if not torch.is_tensor(fire_states):
            fire_states = torch.as_tensor(fire_states, dtype=torch.float32)

        # If input is 4D (batch, time, n, feat), select last time step
        if drone_states.dim() == 4:
            drone_states = drone_states[:, -1, :, :]
        if fire_states.dim() == 4:
            fire_states = fire_states[:, -1, :, :]
        if drone_states.dim() == 5:
            drone_states = drone_states[:, -1, -1, :, :]
        if fire_states.dim() == 5:
            fire_states = fire_states[:, -1, -1, :, :]

        # Only move if needed
        if drone_states.device != self.device:
            drone_states = drone_states.to(self.device)
        if fire_states.device != self.device:
            fire_states = fire_states.to(self.device)

        return drone_states, fire_states

    def forward(self, states, mask=None):
        # drone_states (B,N,F), fire_states (B,2,F), mask (B,2)
        drone_states, fire_states = self.prepare_tensor(states, mask)
        drone_embed = self.drone_embed(drone_states) # (B,N,F)
        fire_embed = self.fire_embed(fire_states) # (B,2,F)
        # Cross-attention between drone and fire states
        attn_output, attention_weights = self.cross_attn(query=drone_embed,
                                         key=fire_embed,
                                         value=fire_embed,
                                         key_padding_mask=mask)

        return attn_output, attention_weights


class Actor(nn.Module):
    """
    A PyTorch Module that represents the actor network of a PPO agent.
    """
    def __init__(self, vision_range, drone_count, map_size, time_steps):
        super(Actor, self).__init__()
        self.Inputspace = Inputspace(drone_count, time_steps=time_steps)
        self.in_features = self.Inputspace.out_features
        # Mu
        self.mu_move = nn.Linear(in_features=self.in_features, out_features=2)
        initialize_output_weights(self.mu_move, 'actor')

        # Logstd
        self.log_std = nn.Parameter(torch.zeros(2, ))

    def forward(self, states, masks=None):
        attn_out, attention_weight = self.Inputspace(states, masks)
        # The attention output is run through another network to get the logits
        # x = self.mu_move(x)
        # TODO We can use the attention weights directly as logits if the attention matrix is the correct assignment

        return attention_weight

class Critic(nn.Module):
    """
    A PyTorch Module that represents the critic network of a PPO agent.
    """
    def __init__(self, vision_range, drone_count, map_size, time_steps):
        super(Critic, self).__init__()
        self.Inputspace = Inputspace(vision_range, time_steps=time_steps)
        self.in_features = self.Inputspace.out_features

        self.value_head = nn.Linear(in_features=self.in_features, out_features=1)

    def forward(self, states, masks=None):
        x, _ = self.Inputspace(states, masks)
        value = self.value_head(x)
        return value