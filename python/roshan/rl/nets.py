"""Policy networks."""

import torch
from torch import nn


def mlp(*sizes):
    layers = []
    for a, b in zip(sizes, sizes[1:]):
        layers += [nn.Linear(a, b), nn.ReLU()]
    return nn.Sequential(*layers)


class FlyNet(nn.Module):
    """Actor-critic for the navigate task.

    The drone's own features and an attention-pooled summary of its neighbour slots (map edge
    and other drones) feed one shared trunk with a policy head and a value head. Empty slots
    are masked out, so the same weights work for any number of drones.
    """

    def __init__(self, agent_dim=8, neighbor_dim=4, hidden=128):
        super().__init__()
        self.own = mlp(agent_dim, hidden, hidden)
        self.neighbor = mlp(neighbor_dim, hidden, hidden)
        self.query = nn.Linear(hidden, hidden)
        self.trunk = mlp(2 * hidden + 1, hidden, hidden)
        self.mean = nn.Linear(hidden, 2)
        self.value = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.full((2,), -0.5))
        nn.init.orthogonal_(self.mean.weight, 0.01)  # start close to "hover"
        nn.init.zeros_(self.mean.bias)

    def forward(self, agent, neighbors, mask):
        """agent [B, A], neighbors [B, K, F], mask [B, K] -> (mean [B, 2], log_std [2], value [B])."""
        own = self.own(agent)
        slots = self.neighbor(neighbors)
        valid = mask.bool()
        scores = torch.einsum("bkh,bh->bk", slots, self.query(own)) / own.shape[-1] ** 0.5
        # A row with no valid slot would be NaN after softmax; multiplying by `valid` makes
        # its weights exactly zero instead.
        weights = torch.softmax(scores.masked_fill(~valid, -1e9), dim=-1) * valid
        pooled = torch.einsum("bk,bkh->bh", weights, slots)
        anyone = valid.any(-1, keepdim=True).float()
        z = self.trunk(torch.cat([own, pooled, anyone], dim=-1))
        return self.mean(z), self.log_std, self.value(z).squeeze(-1)


class PlannerNet(nn.Module):
    """Actor-critic for the allocate task.

    Every (drone, target) pair is scored by one small network from the drone's state, the
    target's state and the vector between them, so the weights do not depend on how many
    drones or targets there are. Drones are decided one after the other; each sees which
    targets other drones are already flying to and which earlier drones just claimed, which
    lets the policy spread the team instead of sending everyone to the same fire. A separate
    score stands for "keep doing what you are doing". The value is one number per environment.
    """

    def __init__(self, drone_dim=6, target_dim=4, global_dim=5, hidden=128):
        super().__init__()
        self.drone = mlp(drone_dim + global_dim, hidden, hidden)
        self.target = mlp(target_dim + 1 + global_dim, hidden, hidden)
        # pair input: drone encoding, target features, taken, claimed, relative vector, distance
        self.pair = nn.Sequential(mlp(hidden + target_dim + 2 + 3, hidden, hidden), nn.Linear(hidden, 1))
        self.keep = nn.Linear(hidden, 1)
        self.value = nn.Sequential(mlp(2 * hidden + global_dim, hidden), nn.Linear(hidden, 1))

    def forward(self, drones, targets, target_mask, action_mask, globals_, decide, actions=None, sample=True):
        """Chooses (or, with `actions`, rates) a slot per drone.

        drones [B, D, 6], targets [B, S, 4], target_mask [B, S], action_mask [B, D, S],
        globals_ [B, 5], decide [B, D]. Slot S means "keep".
        Returns actions [B, D], log_prob [B], entropy [B], value [B]; drones that were not
        asked contribute nothing to log_prob and entropy.
        """
        b, d, _ = drones.shape
        s = targets.shape[1]
        decide = decide.bool()
        valid = target_mask.bool()
        ctx = globals_[:, None, :]
        enc = self.drone(torch.cat([drones, ctx.expand(b, d, -1)], -1))

        # A target is taken if a drone that is not deciding now is already flying to it.
        flying = (drones[..., 5] == 1.0) & ~decide
        same = (drones[:, :, None, 3:5] - targets[:, None, :, 0:2]).abs().amax(-1) < 1e-4
        taken = (same & flying[:, :, None]).any(1).float()
        rel = targets[:, None, :, 0:2] - drones[:, :, None, 0:2]
        geometry = torch.cat([rel, rel.norm(dim=-1, keepdim=True)], -1)

        claimed = torch.zeros(b, s, device=drones.device)
        chosen, log_prob, entropy = [], 0.0, 0.0
        for i in range(d):
            features = torch.cat([targets, taken[..., None], claimed[..., None], geometry[:, i]], -1)
            pair = torch.cat([enc[:, i, None, :].expand(b, s, -1), features], -1)
            scores = self.pair(pair).squeeze(-1).masked_fill(action_mask[:, i] == 0, -1e9)
            policy = torch.distributions.Categorical(logits=torch.cat([scores, self.keep(enc[:, i])], -1))
            if actions is not None:
                a = actions[:, i]
            else:
                a = policy.sample() if sample else policy.logits.argmax(-1)
            asked = decide[:, i].float()
            log_prob = log_prob + policy.log_prob(a) * asked
            entropy = entropy + policy.entropy() * asked
            claimed = claimed + nn.functional.one_hot(a, s + 1)[:, :s] * asked[:, None]
            chosen.append(a)

        weights = valid.float()[..., None]
        seen = self.target(torch.cat([targets, taken[..., None], ctx.expand(b, s, -1)], -1))
        pooled = (seen * weights).sum(1) / weights.sum(1).clamp(min=1.0)
        value = self.value(torch.cat([enc.mean(1), pooled, globals_], -1)).squeeze(-1)
        return torch.stack(chosen, 1), log_prob, entropy, value
