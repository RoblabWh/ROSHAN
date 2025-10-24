import os
import torch.nn as nn
import torch

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

class AttentionActor(nn.Module):
    """
    A PyTorch Module that represents the actor network of a PPO agent.
    """
    def __init__(self, vision_range, drone_count, map_size, time_steps, manual_decay):
        super(AttentionActor, self).__init__()
        self.Inputspace = Inputspace(drone_dim=drone_count, time_steps=time_steps)
        self.in_features = self.Inputspace.out_features
        # Mu
        self.mu_move = nn.Linear(in_features=self.in_features, out_features=2)
        self.mu_move._init_gain = 0.1

        # Logstd
        self.log_std = nn.Parameter(torch.zeros(2, ), requires_grad=not manual_decay)

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
    def __init__(self, vision_range, drone_count, map_size, time_steps, inputspace=None):
        super(Critic, self).__init__()
        self.Inputspace_1 = Inputspace(vision_range, time_steps=time_steps) if not inputspace else inputspace
        self.Inputspace_2 = Inputspace(vision_range, time_steps=time_steps) if not inputspace else inputspace
        self.in_features = self.Inputspace_1.out_features

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement forward.")

class CriticPPO(Critic):
    """
    A PyTorch Module that represents the critic network of a PPO agent.
    """
    def __init__(self, vision_range, drone_count, map_size, time_steps, inputspace=None):
        super(CriticPPO, self).__init__(vision_range, drone_count, map_size, time_steps, inputspace)
        self.Inputspace_2 = None
        # Value
        self.value = nn.Linear(in_features=self.in_features, out_features=1)
        self.value._init_gain = 1.0

    def forward(self, states, masks=None):
        x, _ = self.Inputspace_1(states, masks)
        value = self.value(x)
        return value

class OffPolicyCritic(Critic):
    """
    A PyTorch Module that represents the critic network of an IQL agent.
    """
    def __init__(self, vision_range, drone_count, map_size, time_steps, action_dim, inputspace=None):
        super(OffPolicyCritic, self).__init__(vision_range, drone_count, map_size, time_steps, inputspace)

        # Q1 architecture
        self.l1 = nn.Linear(self.in_features + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        self.l3._init_gain = 1.0

        # Q2 architecture
        self.l4 = nn.Linear(self.in_features + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)
        self.l6._init_gain = 1.0

    def forward(self, state, action):
        q1 = self.Q1(state, action)
        q2 = self.Q2(state, action)

        return q1, q2

    def Q1(self, state, action):
        x = self.Inputspace_1(state)
        x = torch.cat([x, action], dim=1)

        q1 = F.relu(self.l1(x))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

    def Q2(self, state, action):
        x = self.Inputspace_2(state)
        x = torch.cat([x, action], dim=1)

        q2 = F.relu(self.l4(x))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q2

class Value(nn.Module):
    """
    A PyTorch Module that represents the value network of an IQL agent.
    It estimates V(s), the state value.
    """
    def __init__(self, vision_range, drone_count, map_size, time_steps, inputspace):
        super(Value, self).__init__()
        self.Inputspace = Inputspace(vision_range, time_steps=time_steps) if not inputspace else inputspace
        self.in_features = self.Inputspace.out_features

        # Simple MLP head for value estimation
        self.fc1 = nn.Linear(self.in_features, 256)
        self.fc2 = nn.Linear(256, 256)
        self.v_value = nn.Linear(256, 1)
        self.v_value._init_gain = 1.0

    def forward(self, state):
        x = self.Inputspace(state)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        v = self.v_value(x)
        return v