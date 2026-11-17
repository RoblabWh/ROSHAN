"""The learned planner (network, imitation) and flight by an outside pilot."""

import torch

from roshan import AllocateEnv
from roshan.eval import evaluate
from roshan.rl.nets import FlyNet, PlannerNet
from roshan.rl.train_planner import imitate, observe, planner_policy
from test_allocate import CONFIG, grassland


def test_planner_net_respects_the_mask_and_rates_its_own_choice():
    torch.manual_seed(0)
    env = AllocateEnv(8, grassland(), seed=3, config=CONFIG)
    obs = observe(env, "cpu")
    net = PlannerNet()
    actions, log_prob, entropy, value = net(*obs)
    keep = env.spec["slots"]
    allowed = torch.cat([obs[3] > 0, torch.ones(8, 4, 1, dtype=torch.bool)], -1)  # "keep" is always allowed
    assert allowed.gather(2, actions[..., None]).all()
    assert actions.max() <= keep and value.shape == (8,) and (entropy >= 0).all()
    again = net(*obs, actions=actions)[1]
    assert torch.allclose(again, log_prob, atol=1e-5)
    # Drones that were not asked add nothing.
    nobody = (*obs[:5], torch.zeros_like(obs[5]))
    assert net(*nobody, actions=actions)[1].abs().max() == 0


def test_imitation_learns_the_greedy_planner_and_runs_in_the_evaluation():
    torch.manual_seed(0)
    net = PlannerNet()
    before = imitate(net, torch.optim.Adam(net.parameters(), 0.0), grassland(), "cpu", "greedy", steps=1, num_envs=64, config=CONFIG)
    after = imitate(net, torch.optim.Adam(net.parameters(), 1e-3), grassland(), "cpu", "greedy", steps=150, num_envs=64, config=CONFIG)
    assert after > before + 0.2, (before, after)
    rows = evaluate(grassland(), planner_policy(net, "cpu"), [1, 2, 3, 4], CONFIG)
    assert len(rows) == 4 and sum(r["Drops"] for r in rows) > 0


def test_an_outside_pilot_moves_the_drones():
    env = AllocateEnv(2, grassland(), seed=1, config=CONFIG)
    env.autoreset = False
    start = env.drones[..., :2].clone()
    env.fly_tick(torch.zeros(2, 4, 2))
    assert torch.equal(env.drones[..., :2], start)  # hovering
    agent, neighbors, mask = env.pilot
    assert agent.shape == (2, 4, 8) and neighbors.shape == (2, 4, 4, 4) and mask.shape == (2, 4, 4)
    velocity = torch.zeros(2, 4, 2)
    velocity[..., 0] = 1.0
    env.fly_tick(velocity)
    moved = env.drones[..., :2] - start
    assert (moved[..., 0] > 0).all() and (moved[..., 1] == 0).all()


def test_evaluation_with_a_fly_network_finishes():
    torch.manual_seed(0)
    rows = evaluate(grassland(), "greedy", [1, 2], {**CONFIG, "time_limit_s": 60.0}, fly=FlyNet())
    assert len(rows) == 2 and all("Min_Separation" in r for r in rows)
