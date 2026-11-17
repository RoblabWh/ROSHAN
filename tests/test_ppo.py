"""PPO building blocks: advantage estimation, the squashed policy, and one real update."""

import math

import torch

from roshan.rl.nets import FlyNet
from roshan.rl.ppo import PPOConfig, gae, squash, update
from roshan.rl.train import train


def column(*values):
    return torch.tensor(values, dtype=torch.float32).unsqueeze(1)


def test_gae_matches_the_textbook_recursion():
    gamma, lam = 0.9, 0.8
    rewards, values = column(1.0, 2.0, 3.0), column(0.5, 0.4, 0.3, 0.2)
    zeros = torch.zeros(3, 1)
    adv, ret = gae(rewards, values, zeros, zeros, torch.full((3, 1), gamma), lam)
    d = [1.0 + gamma * 0.4 - 0.5, 2.0 + gamma * 0.3 - 0.4, 3.0 + gamma * 0.2 - 0.3]
    want = [d[0] + gamma * lam * (d[1] + gamma * lam * d[2]), d[1] + gamma * lam * d[2], d[2]]
    assert torch.allclose(adv.squeeze(1), torch.tensor(want), atol=1e-6)
    assert torch.allclose(ret, adv + values[:-1])


def test_a_true_ending_drops_the_bootstrap_and_a_cut_off_keeps_it():
    gamma, lam = 0.9, 0.95
    rewards, values = column(1.0, 1.0), column(10.0, 20.0, 30.0)
    ended = column(1.0, 0.0)
    discount = torch.full((2, 1), gamma)
    # Terminated at step 0: nothing follows, and step 1 belongs to another episode.
    adv, _ = gae(rewards, values, ended, ended, discount, lam)
    assert adv[0].item() == 1.0 - 10.0
    # Cut off at step 0: the value of the final observation (20) still counts.
    adv, _ = gae(rewards, values, torch.zeros(2, 1), ended, discount, lam)
    assert math.isclose(adv[0].item(), 1.0 + gamma * 20.0 - 10.0, rel_tol=1e-6)
    # Either way the advantage of step 1 must not leak back across the episode boundary.
    assert math.isclose(adv[1].item(), 1.0 + gamma * 30.0 - 20.0, rel_tol=1e-6)


def test_steps_of_different_length_are_discounted_by_their_duration():
    # One 3-second step must be worth the same as three 1-second steps with nothing in between.
    gamma_s, lam = 0.9, 1.0
    zeros = lambda n: torch.zeros(n, 1)  # noqa: E731
    long_adv, _ = gae(column(0.0), column(0.0, 5.0), zeros(1), zeros(1), column(gamma_s**3), lam)
    short_adv, _ = gae(zeros(3), column(0.0, 0.0, 0.0, 5.0), zeros(3), zeros(3), torch.full((3, 1), gamma_s), lam)
    assert math.isclose(long_adv[0].item(), gamma_s**3 * 5.0, rel_tol=1e-6)
    assert math.isclose(short_adv[0].item(), long_adv[0].item(), rel_tol=1e-6)


def test_squashed_policy_scores_its_own_samples_consistently():
    torch.manual_seed(0)
    mean, log_std = torch.randn(1000, 2), torch.tensor([-0.5, 0.2])
    raw, action, log_prob = squash(mean, log_std)
    assert action.abs().max() < 1.0
    _, again, rescored = squash(mean, log_std, raw)
    assert torch.equal(action, again) and torch.allclose(log_prob, rescored)
    # Density check in 1-D: integrate exp(log_prob) over the action range.
    grid = torch.linspace(-0.999, 0.999, 20001).unsqueeze(1)
    raw_grid = torch.atanh(grid).expand(-1, 2).clone()
    raw_grid[:, 1] = 0.0
    _, _, lp = squash(torch.zeros(20001, 2), torch.zeros(2), raw_grid)
    lp_other = torch.distributions.Normal(0.0, 1.0).log_prob(torch.tensor(0.0))  # the fixed 2nd axis
    density = (lp - lp_other).exp()
    assert math.isclose(torch.trapezoid(density, grid.squeeze(1)).item(), 1.0, abs_tol=0.01)


def test_empty_neighbour_slots_do_not_poison_the_network():
    net = FlyNet()
    agent, neighbors = torch.randn(5, 8), torch.randn(5, 4, 4)
    mask = torch.tensor([[1, 1, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [1, 1, 1, 1]], dtype=torch.uint8)
    mean, _, value = net(agent, neighbors, mask)
    assert torch.isfinite(mean).all() and torch.isfinite(value).all()
    # What sits in a masked slot must not matter.
    scrambled = neighbors.clone()
    scrambled[mask == 0] = 1e3
    mean2, _, value2 = net(agent, scrambled, mask)
    assert torch.allclose(mean, mean2, atol=1e-5) and torch.allclose(value, value2, atol=1e-5)


def test_update_moves_the_policy_and_skips_invalid_samples():
    torch.manual_seed(0)
    net = FlyNet()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    n = 256
    obs = (torch.randn(n, 8), torch.randn(n, 4, 4), torch.ones(n, 4, dtype=torch.uint8))
    with torch.no_grad():
        mean, log_std, value = net(*obs)
        raw, _, log_prob = squash(mean, log_std)
    advantages, valid = torch.randn(n), torch.ones(n, dtype=torch.bool)
    valid[: n // 2] = False
    advantages[~valid] = float("nan")  # would poison the loss if it were ever used
    stats = update(net, optimizer, PPOConfig(), obs, raw, log_prob, advantages, value + 1.0, valid)
    assert all(math.isfinite(v) for v in stats.values()), stats
    assert not torch.allclose(net(*obs)[0], mean), "the policy must have changed"


def test_a_few_training_updates_run_end_to_end(tmp_path):
    cfg = PPOConfig(rollout=16)
    net, result = train(num_envs=32, updates=3, out=tmp_path, cfg=cfg, log_every=1)
    assert (tmp_path / "fly.pt").exists() and (tmp_path / "evaluation.json").exists()
    assert math.isclose(sum(result.values()), 1.0, abs_tol=1e-5)
    checkpoint = torch.load(tmp_path / "fly.pt", weights_only=True)
    assert checkpoint["spec"]["terms"][0] == "goal" and checkpoint["schema_version"] == 1
    FlyNet().load_state_dict(checkpoint["model"])
