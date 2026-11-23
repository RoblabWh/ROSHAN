"""Proximal Policy Optimization for batched ROSHAN environments. One file on purpose."""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class PPOConfig:
    lr: float = 3e-4
    # Discount per simulated second. A step that takes `s` seconds is discounted by
    # gamma_per_s ** s, so changing the step length does not change the objective.
    gamma_per_s: float = 0.90
    lam: float = 0.95
    clip: float = 0.2
    epochs: int = 4
    minibatches: int = 8
    vf_coef: float = 0.5
    ent_coef: float = 0.003
    max_grad_norm: float = 0.5
    rollout: int = 64


def gae(rewards, values, terminated, done, discount, lam):
    """Generalised advantage estimation over one rollout.

    rewards, terminated, done, discount: [T, B].  values: [T + 1, B]; row t + 1 is the value
    of the observation that followed step t, which for a finished episode is its final
    observation.

    `discount[t]` is gamma ** (duration of step t). `terminated` marks true endings, where
    nothing follows and the bootstrap is dropped. `done` (terminated or cut off by the time
    limit) only stops advantages from leaking across episodes: a cut-off episode still
    bootstraps from the value of its final observation.

    Returns (advantages, returns), both [T, B].
    """
    advantages = torch.zeros_like(rewards)
    carry = torch.zeros_like(rewards[0])
    for t in reversed(range(rewards.shape[0])):
        delta = rewards[t] + discount[t] * values[t + 1] * (1 - terminated[t]) - values[t]
        carry = delta + discount[t] * lam * (1 - done[t]) * carry
        advantages[t] = carry
    return advantages, advantages + values[:-1]


def squash(mean, log_std, raw=None):
    """Tanh-squashed Gaussian policy. Returns (raw sample, action in (-1, 1), log-probability).

    Pass `raw` to score a stored sample instead of drawing a new one. The log-probability is
    that of the squashed action, so the entropy bonus sees the distribution that actually
    acts (scoring the unsquashed Gaussian lets its std grow without bound).
    """
    std = log_std.exp()
    if raw is None:
        raw = mean + std * torch.randn_like(mean)
    action = torch.tanh(raw)
    log_prob = torch.distributions.Normal(mean, std).log_prob(raw) - torch.log1p(-action.pow(2) + 1e-6)
    return raw, action, log_prob.sum(-1)


def score_squashed(net, obs, raw_actions):
    """Log-probability, entropy estimate and value for the squashed Gaussian fly policy."""
    mean, log_std, value = net(*obs)
    _, _, log_prob = squash(mean, log_std, raw_actions)
    return log_prob, -log_prob, value  # -log_prob: sample estimate of the entropy


def update(net, optimizer, cfg, obs, actions, old_log_prob, advantages, returns, valid, score=score_squashed,
           value_only=False):
    """Clipped PPO update over the samples where `valid` is set.

    obs: tuple of tensors [N, ...]. `score(net, obs, actions) -> (log_prob, entropy, value)`,
    each [N], says how the policy rates stored actions; the default is the fly policy's.
    With `value_only` the policy is left alone and only the value function learns, which lets
    the critic catch up with a policy that was trained some other way.
    Returns a dict of diagnostics.
    """
    index = valid.nonzero().squeeze(-1)
    stats = {}
    for _ in range(cfg.epochs):
        for batch in index[torch.randperm(len(index), device=index.device)].chunk(cfg.minibatches):
            log_prob, entropy, value = score(net, tuple(o[batch] for o in obs), actions[batch])
            adv = advantages[batch]
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            ratio = (log_prob - old_log_prob[batch]).exp()
            policy_loss = -torch.min(ratio * adv, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * adv).mean()
            value_loss = 0.5 * (value - returns[batch]).pow(2).mean()
            entropy = entropy.mean()
            loss = cfg.vf_coef * value_loss
            if not value_only:
                loss = loss + policy_loss - cfg.ent_coef * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
            optimizer.step()
            stats = {
                "policy_loss": policy_loss.item(),
                "value_loss": value_loss.item(),
                "entropy": entropy.item(),
                "clip_fraction": ((ratio - 1).abs() > cfg.clip).float().mean().item(),
                "approx_kl": ((ratio - 1) - ratio.log()).mean().item(),
            }
    with torch.no_grad():
        _, _, value = score(net, tuple(o[index] for o in obs), actions[index])
        explained = 1 - (returns[index] - value).var() / (returns[index].var() + 1e-8)
    stats["explained_variance"] = explained.item()
    return stats
