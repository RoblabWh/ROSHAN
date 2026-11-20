"""Planners without learning for the allocate task. They run in the Rust core
(`AllocEnv::assign` and `rollout` in crates/roshan-core/src/allocate.rs).

Each takes an AllocateEnv and returns actions [E, D] (int32) for its current observation.
"""

import torch


def idle(env):
    """No drone is ever sent: the reference for how much a planner saves."""
    return torch.full(env.actions.shape, -1, dtype=torch.int32)


def greedy(env):
    """Every deciding drone to the nearest block that still needs water after what is on its
    way; drones left over wait at their station."""
    return env.plan("greedy")


def head(env):
    """Like greedy, but the most intense block first: attack the head of the fire."""
    return env.plan("head")


def flank(env):
    """Like greedy, but the least intense block first: work along the flanks."""
    return env.plan("flank")


def rollout(env, horizon_s=300.0, candidates=4):
    """Greedy improved by lookahead: for each deciding drone, a few alternatives are simulated
    ahead on copies of the world and the best is taken."""
    return env.plan("rollout", horizon_s, candidates)


def triage(env):
    """Blocks whose strongest front is beyond the cap get no drone; the rest by the lowest score of
    distance, intensity, drops still lacking and open neighbours (the `triage` settings of the task)."""
    return env.plan("triage")


def triage_rollout(env, horizon_s=300.0, candidates=4):
    """Triage improved by lookahead; the alternatives are the blocks the rule ranks best."""
    return env.plan("triage_rollout", horizon_s, candidates)


POLICIES = {"none": idle, "greedy": greedy, "head": head, "flank": flank, "rollout": rollout, "triage": triage,
            "triage_rollout": triage_rollout}
