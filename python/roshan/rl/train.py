"""Train the low-level fly policy on the navigate task.

    uv run python -m roshan.rl.train experiments/fly.toml          # writes runs/fly/
    uv run python -m roshan.rl.train --updates 300 --out runs/navigate
    uv run python -m roshan.rl.train --evaluate decentral          # or straight, scripted, a fly.pt

The experiment file (see roshan.experiment) has [run], [net], [sim], [navigate], [reward] and [ppo];
flags override it. Without a file the defaults below run.
"""

import argparse
import dataclasses
import json
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from roshan import NavigateEnv, experiment
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
# [run] of an experiment file. The map is empty: nx x ny cells of cell_m metres.
RUN = {"kind": "fly", "envs": 1024, "updates": 300, "seed": 0, "nx": 35, "ny": 36, "cell_m": 10.0}
# [net] of an experiment file: the arguments of FlyNet besides the observation sizes.
NET = {"hidden": 128}


def load(checkpoint, device):
    """The fly network of a checkpoint written by `train`."""
    saved = torch.load(checkpoint, weights_only=True)
    net = FlyNet(hidden=saved.get("hidden", 128)).to(device)  # checkpoints from before [net] have 128
    net.load_state_dict(saved["model"])
    return net


def observe(env, device):
    """The current observation, one row per drone, on `device`."""
    e, d = env.agent.shape[:2]
    return (
        env.agent.to(device).view(e * d, -1),
        env.neighbors.to(device).view(e * d, d, -1),
        env.mask.to(device).view(e * d, d),
    )


@torch.no_grad()
def evaluate(net, device, episodes=1000, seed=10_000_019, config=None, threads=0, sim=None, nx=35, ny=36, cell_m=10.0):
    """Deterministic policy on `episodes` fresh episodes, each run exactly once.

    `net` is a fly network or the name of an untrained pilot (see NavigateEnv.pilot); all of
    them fly the same episodes. The seed is far from the training seeds, so these episodes
    were never trained on. Returns the share of episodes per outcome.
    """
    env = NavigateEnv(episodes, seed=seed, threads=threads, nx=nx, ny=ny, cell_m=cell_m, config=config, sim=sim)
    env.autoreset = False
    e, d = env.agent.shape[:2]
    running = torch.ones(e, dtype=torch.bool)
    outcome = torch.zeros(e, dtype=torch.long)
    while running.any():
        if isinstance(net, str):
            env.step(env.pilot(net))
        else:
            mean, _, _ = net(*observe(env, device))
            env.step(torch.tanh(mean).view(e, d, 2))
        ended = running & ((env.terminated | env.truncated) > 0)
        outcome[ended] = env.info[ended, 1].long()
        running &= ~ended
    return {name: (outcome == i).float().mean().item() for i, name in enumerate(REASONS)}


def train(num_envs=1024, updates=300, seed=0, out="runs/navigate", cfg=PPOConfig(), reward=None, config=None, log_every=10,
          sim=None, hidden=128, nx=35, ny=36, cell_m=10.0):
    """Trains the fly policy with PPO. What ran is written to `out`/experiment.toml before training starts."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    began = time.perf_counter()
    reward = reward or REWARD
    env = NavigateEnv(num_envs, seed=seed, nx=nx, ny=ny, cell_m=cell_m, config=config, sim=sim)
    e, d = env.agent.shape[:2]
    b, t_len = e * d, cfg.rollout
    weights = torch.tensor([reward[name] for name in env.spec["terms"]], device=device)
    step_s = env.spec["sim"]["dt_s"]
    discount = cfg.gamma_per_s**step_s

    net = FlyNet(len(env.spec["agent"]), len(env.spec["neighbor"]), hidden).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr, eps=1e-5)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tables = {
        "run": {"kind": "fly", "envs": num_envs, "updates": updates, "seed": seed, "nx": nx, "ny": ny, "cell_m": cell_m},
        "net": {"hidden": hidden},
        "sim": env.spec["sim"],
        "navigate": env.spec["config"],
        "reward": reward,
        "ppo": dataclasses.asdict(cfg),
        "record": experiment.record(out),
    }
    experiment.write(out / "experiment.toml", tables)
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

    result = evaluate(net, device, config=config, sim=sim, nx=nx, ny=ny, cell_m=cell_m)
    torch.save(
        {
            "model": net.state_dict(),
            "net": "roshan.rl.nets.FlyNet",
            "hidden": hidden,
            "spec": env.spec,
            "ppo": dataclasses.asdict(cfg),
            "reward": reward,
            "evaluation": result,
            "schema_version": 1,
        },
        out / "fly.pt",
    )
    (out / "evaluation.json").write_text(json.dumps(result, indent=2))
    experiment.finish(out / "experiment.toml", tables, time.perf_counter() - began, result)
    print("evaluation on unseen episodes:", "  ".join(f"{k} {v:.3f}" for k, v in result.items()))
    writer.close()
    return net, result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", nargs="?", help="an experiment file (.toml); flags override it")
    parser.add_argument("--envs", type=int)
    parser.add_argument("--updates", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--out", help="default: runs/<name of the experiment file>, or runs/navigate")
    parser.add_argument("--force", action="store_true", help="write over a run that is already in --out")
    parser.add_argument("--evaluate", metavar="PILOT", help="no training: fly 1000 unseen episodes with "
                        "'straight', 'scripted', 'decentral' or a fly checkpoint (.pt)")
    parser.add_argument("--drones", type=int, help="with --evaluate: number of drones (default 4)")
    args = parser.parse_args()
    if args.evaluate:
        pilot, device = args.evaluate, "cuda" if torch.cuda.is_available() else "cpu"
        if pilot.endswith(".pt"):
            pilot = load(pilot, device)
        result = evaluate(pilot, device, config={"drones": args.drones} if args.drones else None)
        print(f"{args.evaluate}:", "  ".join(f"{k} {v:.3f}" for k, v in result.items()))
    else:
        exp = experiment.training(args.experiment, "fly", "navigate") if args.experiment else {}
        given = {k: v for k, v in vars(args).items() if v is not None}
        run = experiment.merged(RUN, exp.get("run", {}), "run") | {k: given[k] for k in RUN if k in given}
        train(
            run["envs"], run["updates"], run["seed"],
            out=experiment.claim(args.out or f"runs/{Path(args.experiment).stem if args.experiment else 'navigate'}", args.force),
            cfg=PPOConfig(**experiment.merged(dataclasses.asdict(PPOConfig()), exp.get("ppo", {}), "ppo")),
            reward=experiment.merged(REWARD, exp.get("reward", {}), "reward"), config=exp.get("navigate"), sim=exp.get("sim"),
            hidden=experiment.merged(NET, exp.get("net", {}), "net")["hidden"], nx=run["nx"], ny=run["ny"], cell_m=run["cell_m"],
        )
