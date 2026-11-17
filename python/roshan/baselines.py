"""Planners without learning for the allocate task: idle, greedy and Hungarian assignment.

Each takes an AllocateEnv and returns actions [E, D] (int32) for its current observation.
"""

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

TASK_FIRE = 1.0


def assign(drone_xy, water, task_xy, task_kind, decide, can_drop, target_xy, target_mask, stations, method):
    """Targets for the drones of one environment. Coordinates in metres.

    drone_xy [D, 2], water [D] (tank fraction), task_xy [D, 2], task_kind [D], decide [D],
    can_drop [D] (enough water for a drop), target_xy [S, 2], target_mask [S]; the first
    `stations` slots are ground stations. Returns [D] slots, -1 meaning "keep".

    Rules shared by both methods:
    - only drones asked to decide get a new target;
    - a fire another drone is already flying to is left alone;
    - a drone that cannot drop goes to the nearest station; a drone left without a fire
      refills if its tank is not full and otherwise waits.
    `greedy` repeatedly pairs the closest free drone and free fire; `hungarian` minimises the
    total distance of the pairing.
    """
    n = len(drone_xy)
    actions = np.full(n, -1, np.int32)
    fires = [s for s in range(stations, len(target_xy)) if target_mask[s]]
    busy = [task_xy[j] for j in range(n) if task_kind[j] == TASK_FIRE and not decide[j]]
    free = [s for s in fires if not any(np.allclose(target_xy[s], b, atol=1.0) for b in busy)]
    ready = [i for i in range(n) if decide[i] and can_drop[i]]

    pairs = []
    if ready and free:
        cost = np.linalg.norm(drone_xy[ready][:, None] - target_xy[free][None], axis=-1)
        if method == "hungarian":
            pairs = list(zip(*linear_sum_assignment(cost)))
        elif method == "greedy":
            cost = cost.copy()
            for _ in range(min(len(ready), len(free))):
                r, c = np.unravel_index(np.argmin(cost), cost.shape)
                pairs.append((r, c))
                cost[r, :], cost[:, c] = np.inf, np.inf
        else:
            raise ValueError(f"unknown method {method!r}")
    for r, c in pairs:
        actions[ready[r]] = free[c]

    for i in range(n):
        if decide[i] and actions[i] < 0 and water[i] < 1.0 and stations:
            actions[i] = int(np.argmin(np.linalg.norm(target_xy[:stations] - drone_xy[i], axis=-1)))
    return actions


def _planner(method):
    def plan(env):
        size = np.array(env.spec["size_m"], np.float32)
        stations = env.spec["stations"]
        drones, targets = env.drones.numpy(), env.targets.numpy()
        can_drop = env.action_mask.numpy()[:, :, stations:].any(-1) | (
            drones[:, :, 2] >= env.spec["config"]["drop_l"] / env.spec["config"]["tank_l"]
        )
        out = np.empty(env.actions.shape, np.int32)
        for e in range(out.shape[0]):
            out[e] = assign(
                drones[e, :, 0:2] * size, drones[e, :, 2], drones[e, :, 3:5] * size, drones[e, :, 5],
                env.decide.numpy()[e], can_drop[e], targets[e, :, 0:2] * size, env.target_mask.numpy()[e],
                stations, method,
            )
        return torch.from_numpy(out)

    return plan


def idle(env):
    """No drone ever does anything: the reference for how much a planner saves."""
    return torch.full(env.actions.shape, -1, dtype=torch.int32)


greedy = _planner("greedy")
hungarian = _planner("hungarian")


def rollout(env, horizon_s=120.0, candidates=4):
    """Greedy improved by lookahead: for each deciding drone, a few alternatives are simulated
    ahead on copies of the world and the best is taken (implemented in the Rust core)."""
    return env.plan("rollout", horizon_s, candidates)


POLICIES = {"none": idle, "greedy": greedy, "hungarian": hungarian, "rollout": rollout}
