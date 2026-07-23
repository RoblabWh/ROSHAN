"""Regression tests for the SMDP GAE alignment fix.

Under the deferred transition collection in agent_handler.train_loop, each stored planner
transition t carries reward[t], duration[t], and next_value[t] all describing the SAME
forward window [t, t+1] governed by action a_t. PPO.get_advantages must therefore discount
transition t by gamma**durations[t] directly — NOT gamma**durations[t+1] (the old "+1 shift"
that compensated for the pre-fix off-by-one collection).
"""
import pytest

torch = pytest.importorskip("torch")

# conftest.py puts src/pysim and build/ on sys.path so the real PPO (and its firesim /
# networks imports) resolve exactly as they do when main.py runs.
from algorithms.ppo import PPO


class _Stub:
    """Minimal carrier of the attributes get_advantages reads."""
    gamma = 0.99
    _lambda = 0.95
    smdp_normalize_k = False
    max_low_level_steps = 20
    device = torch.device("cpu")


def _reference_gae(gamma, lam, values, masks, rewards, k):
    """Independent SMDP-GAE reference: delta_t uses gamma**k_t with NO index shift."""
    T = len(rewards)
    next_values = values[1:T + 1]
    gamma_k = gamma ** k
    deltas = rewards + gamma_k * masks * next_values - values[:T]
    coeffs = gamma_k * lam * masks
    adv = torch.zeros(T)
    gae = 0.0
    for t in reversed(range(T)):
        gae = deltas[t] + coeffs[t] * gae
        adv[t] = gae
    returns = adv + values[:T]
    return adv, returns


def test_get_advantages_uses_durations_without_shift():
    s = _Stub()
    rewards = torch.tensor([1.0, -0.5, 2.0])
    masks = torch.tensor([1.0, 1.0, 1.0])          # no episode boundary
    values = torch.tensor([0.3, 0.1, 0.4, 0.2])    # len T+1 -> bootstrapped tail
    durations = torch.tensor([2.0, 5.0, 3.0])      # distinct so a shift would be detectable

    adv, ret = PPO.get_advantages(
        s, values.clone(), masks.clone(), rewards.clone(),
        durations=durations.clone(), bootstrap_duration=999.0,  # must be ignored now
    )

    exp_adv, exp_ret = _reference_gae(s.gamma, s._lambda, values, masks, rewards, durations)
    assert torch.allclose(adv, exp_adv, atol=1e-6), (adv, exp_adv)
    assert torch.allclose(ret, exp_ret, atol=1e-6)


def test_bootstrap_duration_is_ignored():
    """The tail transition must use durations[-1], not bootstrap_duration."""
    s = _Stub()
    rewards = torch.tensor([1.0, 2.0])
    masks = torch.tensor([1.0, 1.0])
    values = torch.tensor([0.5, 0.2, 0.3])
    durations = torch.tensor([4.0, 7.0])

    adv_a, _ = PPO.get_advantages(s, values.clone(), masks.clone(), rewards.clone(),
                                  durations=durations.clone(), bootstrap_duration=1.0)
    adv_b, _ = PPO.get_advantages(s, values.clone(), masks.clone(), rewards.clone(),
                                  durations=durations.clone(), bootstrap_duration=500.0)
    # Changing bootstrap_duration must not change the result.
    assert torch.allclose(adv_a, adv_b, atol=1e-7)
    # And the tail delta must reflect durations[-1]=7, not any shifted/fallback value.
    exp_adv, _ = _reference_gae(s.gamma, s._lambda, values, masks, rewards, durations)
    assert torch.allclose(adv_a, exp_adv, atol=1e-6)


def test_terminal_mask_zeroes_bootstrap():
    """A done transition (mask=0) must drop the bootstrap term regardless of duration."""
    s = _Stub()
    rewards = torch.tensor([1.0, 3.0])
    masks = torch.tensor([1.0, 0.0])               # second transition terminal
    values = torch.tensor([0.5, 0.2, 9.9])         # tail value should be ignored for t=1
    durations = torch.tensor([2.0, 6.0])

    adv, _ = PPO.get_advantages(s, values.clone(), masks.clone(), rewards.clone(),
                                durations=durations.clone())
    exp_adv, _ = _reference_gae(s.gamma, s._lambda, values, masks, rewards, durations)
    assert torch.allclose(adv, exp_adv, atol=1e-6)
    # delta[1] = reward[1] - value[1] (bootstrap masked out)
    assert abs(adv[1].item() - (3.0 - 0.2)) < 1e-6


def test_none_durations_is_standard_gae():
    """Non-planner path (durations=None) stays the classic gamma^1 GAE."""
    s = _Stub()
    rewards = torch.tensor([1.0, -1.0, 0.5])
    masks = torch.tensor([1.0, 1.0, 1.0])
    values = torch.tensor([0.2, 0.1, 0.3, 0.4])
    ones = torch.ones(3)
    adv_none, _ = PPO.get_advantages(s, values.clone(), masks.clone(), rewards.clone(),
                                     durations=None)
    # gamma^1 everywhere == durations all-ones through the SMDP path
    exp_adv, _ = _reference_gae(s.gamma, s._lambda, values, masks, rewards, ones)
    assert torch.allclose(adv_none, exp_adv, atol=1e-6)
