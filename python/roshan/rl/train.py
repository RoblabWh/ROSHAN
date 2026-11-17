"""Train the low-level fly policy on the navigate task.

    uv run python -m roshan.rl.train --updates 300 --out runs/navigate
"""

import argparse
import dataclasses
import json
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from roshan import NavigateEnv
from roshan.rl.nets import FlyNet
from roshan.rl.ppo import PPOConfig, gae, squash, update

# Reward = sum over terms of weight * term (names from env.spec["terms"]).
# `progress` must outweigh `approach_drone`: with the opposite balance the policy learns to
# hover, which avoids every penalty and never reaches a waypoint.
REWARD = {
    "goal": 1.0,
    "success": 2.0,
    "collision": -3.0,
    "out_of_bounds": -3.0,
    "timeout": -1.0,
    "progress": 1.0,
    "approach_drone": -0.5,
    "approach_edge": -0.5,
}
REASONS = ["success", "collision", "out_of_bounds", "timeout"]


def observe(env, device):
    """The current observation, one row per drone, on `device`."""
    e, d = env.agent.shape[:2]
    return (
        env.agent.to(device).view(e * d, -1),
        env.neighbors.to(device).view(e * d, d, -1),
        env.mask.to(device).view(e * d, d),
    )


@torch.no_grad()
def evaluate(net, device, episodes=1000, seed=10_000_019, config=None, threads=0):
    """Deterministic policy on `episodes` fresh episodes, each run exactly once.

    The seed is far from the training seeds, so these episodes were never trained on.
    Returns the share of episodes per outcome.
    """
    env = NavigateEnv(episodes, seed=seed, threads=threads, config=config)
    env.autoreset = False
    e, d = env.agent.shape[:2]
    running = torch.ones(e, dtype=torch.bool)
    outcome = torch.zeros(e, dtype=torch.long)
    while running.any():
        mean, _, _ = net(*observe(env, device))
        env.step(torch.tanh(mean).view(e, d, 2))
        ended = running & ((env.terminated | env.truncated) > 0)
        outcome[ended] = env.info[ended, 1].long()
        running &= ~ended
    return {name: (outcome == i).float().mean().item() for i, name in enumerate(REASONS)}


def train(num_envs=1024, updates=300, seed=0, out="runs/navigate", cfg=PPOConfig(), reward=None, config=None, log_every=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    env = NavigateEnv(num_envs, seed=seed, config=config)
    e, d = env.agent.shape[:2]
    b, t_len = e * d, cfg.rollout
    weights = torch.tensor([(reward or REWARD)[name] for name in env.spec["terms"]], device=device)
    step_s = env.spec["sim"]["dt_s"]
    discount = cfg.gamma_per_s**step_s

    net = FlyNet(len(env.spec["agent"]), len(env.spec["neighbor"])).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr, eps=1e-5)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(out / "tensorboard")

    first = observe(env, device)
    obs = [torch.zeros((t_len + 1, *o.shape), dtype=o.dtype, device=device) for o in first]
    raw = torch.zeros((t_len, b, 2), device=device)
    log_prob, rewards, terminated, done = (torch.zeros((t_len, b), device=device) for _ in range(4))
    valid = torch.zeros((t_len, b), dtype=torch.bool, device=device)
    values = torch.zeros((t_len + 1, b), device=device)
    # True where the previous step ended an episode: this step only resets, it is not data.
    resetting = torch.zeros(b, dtype=torch.bool, device=device)

    outcomes = torch.zeros(len(REASONS))
    start, steps_done = time.perf_counter(), 0
    for u in range(1, updates + 1):
        # Linear decay to zero: late updates fine-tune instead of overshooting.
        optimizer.param_groups[0]["lr"] = cfg.lr * (1 - (u - 1) / updates)
        with torch.no_grad():
            for t in range(t_len):
                current = observe(env, device)
                for store, o in zip(obs, current):
                    store[t] = o
                mean, log_std, values[t] = net(*current)
                raw[t], action, log_prob[t] = squash(mean, log_std)
                env.step(action.view(e, d, 2))

                rewards[t] = (env.terms.to(device) @ weights).view(b)
                ended_env = (env.terminated | env.truncated) > 0
                terminated[t] = env.terminated.to(device).repeat_interleave(d).float()
                done[t] = ended_env.to(device).repeat_interleave(d).float()
                valid[t] = ~resetting
                resetting = done[t].bool()
                if ended_env.any():
                    outcomes += torch.bincount(env.info[ended_env, 1].long(), minlength=len(REASONS))
            last = observe(env, device)
            for store, o in zip(obs, last):
                store[t_len] = o
            values[t_len] = net(*last)[2]
            advantages, returns = gae(rewards, values, terminated, done, torch.full_like(rewards, discount), cfg.lam)

        flat = lambda x: x.reshape(t_len * b, *x.shape[2:])  # noqa: E731
        stats = update(
            net, optimizer, cfg, tuple(flat(o[:-1]) for o in obs), flat(raw), flat(log_prob),
            flat(advantages), flat(returns), flat(valid),
        )
        steps_done += t_len * b

        if u % log_every == 0 or u == updates:
            rate = steps_done / (time.perf_counter() - start)
            share = outcomes / outcomes.sum().clamp(min=1)
            print(
                f"update {u:4d}  {rate / 1e3:7.0f}k drone-steps/s  episodes {int(outcomes.sum()):6d}  "
                + "  ".join(f"{name} {s:.3f}" for name, s in zip(REASONS, share))
                + f"  value_loss {stats['value_loss']:.3f}  kl {stats['approx_kl']:.4f}  std {net.log_std.exp().mean().item():.2f}"
            )
            for name, s in zip(REASONS, share):
                writer.add_scalar(f"episodes/{name}", s, steps_done)
            for name, value in stats.items():
                writer.add_scalar(f"ppo/{name}", value, steps_done)
            writer.add_scalar("charts/drone_steps_per_s", rate, steps_done)
            outcomes.zero_()

    result = evaluate(net, device, config=config)
    torch.save(
        {
            "model": net.state_dict(),
            "net": "roshan.rl.nets.FlyNet",
            "spec": env.spec,
            "ppo": dataclasses.asdict(cfg),
            "reward": reward or REWARD,
            "evaluation": result,
            "schema_version": 1,
        },
        out / "fly.pt",
    )
    (out / "evaluation.json").write_text(json.dumps(result, indent=2))
    print("evaluation on unseen episodes:", "  ".join(f"{k} {v:.3f}" for k, v in result.items()))
    writer.close()
    return net, result


if __name__ == "__main__":
    # ponytail: plain flags. Layered YAML configs (OmegaConf) come with the experiment
    # harness in M4, when there is more than one thing to configure.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", type=int, default=1024)
    parser.add_argument("--updates", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="runs/navigate")
    args = parser.parse_args()
    train(args.envs, args.updates, args.seed, args.out)
