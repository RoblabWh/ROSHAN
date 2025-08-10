import os
import torch.nn as nn
import torch
from utils import initialize_output_weights

if os.getenv("PYTORCH_DETECT_ANOMALY", "").lower() in ("1", "true"):
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
        self.pos_emb = nn.Linear(2, hidden_dim)
        self.goal_emb = nn.Linear(2, hidden_dim)
        self.fire_emb = nn.Linear(2, hidden_dim)
        self.id_emb = nn.Embedding(drone_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)

        self.out_features = hidden_dim


    def prepare_tensor(self, states):
        drone_states, goal_positions, fire_states = states

        if not torch.is_tensor(drone_states):
            drone_states = torch.as_tensor(drone_states, dtype=torch.float32)

        if not torch.is_tensor(fire_states):
            fire_states = torch.as_tensor(fire_states, dtype=torch.float32)

        if not torch.is_tensor(goal_positions):
            goal_positions = torch.as_tensor(goal_positions, dtype=torch.float32)

        # If input is 4D (batch, time, n, feat), select last time step
        if drone_states.dim() == 4:
            drone_states = drone_states[:, -1, :, :]
        if fire_states.dim() == 4:
            fire_states = fire_states[:, -1, :, :]
        if goal_positions.dim() == 4:
            goal_positions = goal_positions[:, -1, :, :]
        if drone_states.dim() == 5:
            drone_states = drone_states[:, -1, -1, :, :]
        if fire_states.dim() == 5:
            fire_states = fire_states[:, -1, -1, :, :]
        if goal_positions.dim() == 5:
            goal_positions = goal_positions[:, -1, -1, :, :]

        # Only move if needed
        if drone_states.device != self.device:
            drone_states = drone_states.to(self.device)
        if fire_states.device != self.device:
            fire_states = fire_states.to(self.device)
        if goal_positions.device != self.device:
            goal_positions = goal_positions.to(self.device)

        batch_size, n_drones, _ = drone_states.shape
        agent_ids = torch.arange(n_drones, device=drone_states.device).unsqueeze(0).expand(batch_size, -1)
        return drone_states, goal_positions, fire_states, agent_ids

    def forward(self, states, mask=None):
        # drone_states (B,N,F), fire_states (B,2,F), mask (B,2)
        drone_pos, goal_positions, fire_states, agent_ids = self.prepare_tensor(states)
        if mask is None:
            mask = torch.zeros(fire_states.shape[0], fire_states.shape[1], device=drone_pos.device, dtype=torch.bool)
            mask[:, 0] = True  # Mask for fire state 0, which is the groundstation
        pos_emb = self.pos_emb(drone_pos) # (B,N,F)
        goal_emb = self.goal_emb(goal_positions) # (B,N,F)
        fire_emb = self.fire_emb(fire_states) # (B,2,F)
        id_emb = self.id_emb(agent_ids) # (B,N,F)
        drone_emb = pos_emb + goal_emb + id_emb # (B,N,F)
        # Cross-attention between drone and fire states
        attn_output, attention_weights = self.cross_attn(query=drone_emb,
                                         key=fire_emb,
                                         value=fire_emb,
                                         key_padding_mask=mask)

        return attn_output, attention_weights


class Actor(nn.Module):
    """
    A PyTorch Module that represents the actor network of a PPO agent.
    """
    def __init__(self, vision_range, drone_count, map_size, time_steps):
        super(Actor, self).__init__()
        self.Inputspace = Inputspace(drone_dim=drone_count, time_steps=time_steps)
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

class CriticPPO(nn.Module):
    """
    A PyTorch Module that represents the critic network of a PPO agent.
    """
    def __init__(self, vision_range, drone_count, map_size, time_steps):
        super(CriticPPO, self).__init__()
        self.Inputspace = Inputspace(drone_dim=drone_count, time_steps=time_steps)
        self.in_features = self.Inputspace.out_features

        self.value_head = nn.Linear(in_features=self.in_features, out_features=1)

    def forward(self, states, masks=None):
        x, _ = self.Inputspace(states, masks)
        value = self.value_head(x)
        return value