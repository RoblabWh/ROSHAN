"""Train the planner on the allocate task.

    uv run python -m roshan.rl.train_planner experiments/planner.toml      # writes runs/planner/
    uv run python -m roshan.rl.train_planner experiments/planner.toml --updates 5 --out runs/try

The experiment file (see roshan.experiment) has [run], [net], [sim], [allocate], [reward] and [ppo];
flags override it. Without a file the defaults below run.
"""

import argparse
import dataclasses
import json
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from roshan import AllocateEnv, experiment
from roshan.rl.nets import PlannerNet
from roshan.rl.ppo import PPOConfig, gae, update
from roshan.scenario import Scenario

# Reward per environment and step = sum over terms of weight * term (names from
# env.spec["terms"]). `burned` is the share of the burnable area that caught fire during the
# step, so -100 charges one point per percent burned: the reward is the objective itself.
REWARD = {"drop": 0.0, "refill": 0.0, "burned": -100.0, "fire_out": 1.0, "timeout": 0.0}
# The task the planner is trained and compared on (see the results in docs/systems.md).
TASK = {"random_ignitions": 2, "random_wind": True, "preburn_s": 1800.0, "preburn_cells": 20}
PPO = PPOConfig(lr=3e-4, gamma_per_s=0.999, lam=0.95, epochs=4, minibatches=4, ent_coef=0.01, rollout=64)
# [run] of an experiment file. `teacher`: "greedy", "rollout" or "" for none; `init`: a
# checkpoint to start from, or "".
RUN = {"kind": "planner", "scenario": "scenarios/demo.npz", "envs": 512, "updates": 300, "seed": 0, "teacher": "",
       "imitation_steps": 800, "imitation_envs": 128, "init": "", "critic_updates": 0}
# [net] of an experiment file: the arguments of PlannerNet.
NET = {"hidden": 128}
# Of a checkpoint: 2 since the observation has seven target columns.
SCHEMA_VERSION = 2


def observe(env, device):
    """The current observation, one row per environment, on `device`."""
    return tuple(
        getattr(env, name).to(device)
        for name in ("drones", "targets", "target_mask", "action_mask", "globals", "decide")
    )


def score(net, obs, actions):
    _, log_prob, entropy, value = net(*obs, actions=actions)
    return log_prob, entropy, value


def planner_policy(net, device, sample=False):
    """The network as a planner for the evaluation runner and the viewer's batch interface:
    env -> actions [E, D], with -1 for "keep" and for drones that were not asked."""

    @torch.no_grad()
    def plan(env):
        actions, _, _, _ = net(*observe(env, device), sample=sample)
        keep = env.spec["slots"]
        actions = torch.where((actions == keep) | (env.decide.to(device) == 0), -1, actions)
        return actions.int().cpu()

    return plan


def imitate(net, optimizer, scenario, device, teacher="rollout", steps=800, num_envs=128, seed=0, config=None, log_every=100,
            sim=None):
    """Teaches the network to choose what a built-in planner (`teacher`) would choose.

    The environments are first driven by the teacher and, from the halfway point on, by the
    network itself, while the teacher keeps labelling every state. That way the network also
    sees, and learns to recover from, the situations its own mistakes lead to (DAgger).
    Returns the share of decisions on which the network agreed with the teacher at the end.
    """
    env = AllocateEnv(num_envs, scenario, seed=seed + 1_000, config=config if config is not None else TASK, sim=sim)
    keep = env.spec["slots"]
    agreement = 0.0
    for step in range(1, steps + 1):
        obs = observe(env, device)
        wanted = env.plan(teacher).to(device).long()
        wanted = torch.where(wanted < 0, keep, wanted)
        _, log_prob, _, _ = net(*obs, actions=wanted)
        asked = obs[5].bool()
        loss = -log_prob.sum() / asked.sum().clamp(min=1)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            own, _, _, _ = net(*obs, sample=False)
            agreement = ((own == wanted) & asked).sum().item() / max(asked.sum().item(), 1)
            chosen = wanted if step <= steps // 2 else own
            env.step(torch.where((chosen == keep) | ~asked, -1, chosen))
        if step % log_every == 0:
            print(f"imitation step {step:4d}  loss {loss.item():.3f}  agrees with {teacher} on {agreement:.1%} of decisions")
    return agreement


def load(checkpoint, device=None):
    """The network of a checkpoint written by `train`."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(checkpoint, weights_only=True)
    if saved.get("schema_version", 1) < SCHEMA_VERSION:
        raise ValueError(f"{checkpoint} was trained on the task as it was before 2026-11-19 (drones with their own "
                         "start cells, four target columns); train a new one")
    net = PlannerNet(hidden=saved.get("hidden", 128)).to(device)  # checkpoints from before [net] have 128
    net.load_state_dict(saved["model"])
    return net.eval(), device


def train(scenario, num_envs=512, updates=300, seed=0, out="runs/planner", cfg=PPO, reward=None, config=None, log_every=10,
          teacher=None, imitation_steps=800, init=None, critic_updates=0, sim=None, hidden=128, imitation_envs=128):
    """Trains a planner with PPO on `scenario` (a Scenario or the path of one). With `teacher` ("greedy" or "rollout") the network first
    imitates that planner for `imitation_steps` environment steps; `init` starts from saved weights
    instead. During the first `critic_updates` updates only the value function learns: a
    policy that was imitated comes with an untrained critic, whose random advantages would
    otherwise undo it. What ran is written to `out`/experiment.toml before training starts."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    began, result = time.perf_counter(), {}
    path = None
    if isinstance(scenario, (str, Path)):
        path, scenario = str(scenario), Scenario.load(scenario)
    reward = reward or REWARD
    env = AllocateEnv(num_envs, scenario, seed=seed, config=config if config is not None else TASK, sim=sim)
    e, d, keep = num_envs, env.spec["drones"], env.spec["slots"]
    t_len = cfg.rollout
    weights = torch.tensor([reward[name] for name in env.spec["terms"]], device=device)
    # Team terms (the last three) are the same for every drone; drone terms are summed.
    per_drone = torch.tensor([1.0, 1.0, 1.0 / d, 1.0 / d, 1.0 / d], device=device)

    net = PlannerNet(hidden=hidden).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr, eps=1e-5)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tables = {
        "run": {"kind": "planner", "scenario": path, "envs": num_envs, "updates": updates, "seed": seed, "teacher": teacher,
                "imitation_steps": imitation_steps, "imitation_envs": imitation_envs, "init": init,
                "critic_updates": critic_updates},
        "net": {"hidden": hidden},
        "sim": env.spec["sim"],
        "allocate": env.spec["config"],
        "reward": reward,
        "ppo": dataclasses.asdict(cfg),
        "record": experiment.record(out, scenario_fingerprint=scenario.fingerprint(),
                                    init_digest=experiment.digest(init) if init else None),
    }
    experiment.write(out / "experiment.toml", tables)
    writer = SummaryWriter(out / "tensorboard")
    if init:
        net.load_state_dict(torch.load(init, weights_only=True)["model"])
    if teacher:
        result["teacher_agreement"] = imitate(net, optimizer, scenario, device, teacher, imitation_steps, imitation_envs,
                                              seed=seed, config=config, sim=sim)
        torch.save({"model": net.state_dict(), "spec": env.spec, "hidden": hidden, "schema_version": SCHEMA_VERSION}, out / "imitation.pt")

    first = observe(env, device)
    obs = [torch.zeros((t_len + 1, *o.shape), dtype=o.dtype, device=device) for o in first]
    actions = torch.zeros((t_len, e, d), dtype=torch.long, device=device)
    log_prob, rewards, terminated, done, discount = (torch.zeros((t_len, e), device=device) for _ in range(5))
    valid = torch.zeros((t_len, e), dtype=torch.bool, device=device)
    values = torch.zeros((t_len + 1, e), device=device)
    # True where the previous step ended an episode: this step only resets, it is not data.
    resetting = torch.zeros(e, dtype=torch.bool, device=device)

    episodes, fire_out, burned, drops = 0, 0.0, 0.0, 0.0
    start, decisions = time.perf_counter(), 0
    for u in range(1, updates + 1):  # updates == 0: imitation only
        optimizer.param_groups[0]["lr"] = cfg.lr * (1 - (u - 1) / updates)
        with torch.no_grad():
            for t in range(t_len):
                current = observe(env, device)
                for store, o in zip(obs, current):
                    store[t] = o
                actions[t], log_prob[t], _, values[t] = net(*current)
                decisions += int(env.decide.sum())
                asked = current[5].bool()
                env.step(torch.where((actions[t] == keep) | ~asked, -1, actions[t]))

                rewards[t] = ((env.terms.to(device) * per_drone).sum(1) * weights).sum(-1)
                ended = (env.terminated | env.truncated) > 0
                terminated[t] = env.terminated.to(device).float()
                done[t] = ended.to(device).float()
                discount[t] = cfg.gamma_per_s ** env.elapsed_s.to(device)
                valid[t] = ~resetting
                resetting = done[t].bool()
                if ended.any():
                    info = env.info[ended]
                    episodes += len(info)
                    fire_out += info[:, 0].sum().item()
                    burned += info[:, 3].sum().item()
                    drops += info[:, 5].sum().item()
            last = observe(env, device)
            for store, o in zip(obs, last):
                store[t_len] = o
            values[t_len] = net(*last)[3]
            advantages, returns = gae(rewards, values, terminated, done, discount, cfg.lam)

        flat = lambda x: x.reshape(t_len * e, *x.shape[2:])  # noqa: E731
        stats = update(
            net, optimizer, cfg, tuple(flat(o[:-1]) for o in obs), flat(actions), flat(log_prob),
            flat(advantages), flat(returns), flat(valid), score=score, value_only=u <= critic_updates,
        )

        if u % log_every == 0 or u == updates:
            n = max(episodes, 1)
            rate = decisions / (time.perf_counter() - start)
            print(
                f"update {u:4d}  {rate / 1e3:5.1f}k decisions/s  episodes {episodes:5d}  burned {burned / n:.4f}  "
                f"fire out {fire_out / n:.3f}  drops {drops / n:5.1f}  value_loss {stats['value_loss']:.3f}  "
                f"kl {stats['approx_kl']:.4f}  entropy {stats['entropy']:.2f}"
            )
            step = u * t_len * e
            writer.add_scalar("episodes/burned_fraction", burned / n, step)
            writer.add_scalar("episodes/fire_out", fire_out / n, step)
            writer.add_scalar("episodes/drops", drops / n, step)
            for name, value in stats.items():
                writer.add_scalar(f"ppo/{name}", value, step)
            writer.add_scalar("charts/decisions_per_s", rate, step)
            # Training episodes with the sampled policy since the last line; not an evaluation.
            if episodes:
                result |= {"train_episodes": episodes, "train_burned_fraction": burned / n,
                           "train_fire_out": fire_out / n, "train_drops": drops / n}
            episodes, fire_out, burned, drops = 0, 0.0, 0.0, 0.0

    torch.save(
        {
            "model": net.state_dict(),
            "net": "roshan.rl.nets.PlannerNet",
            "hidden": hidden,
            "spec": env.spec,
            "ppo": dataclasses.asdict(cfg),
            "reward": reward,
            "schema_version": SCHEMA_VERSION,
        },
        out / "planner.pt",
    )
    experiment.finish(out / "experiment.toml", tables, time.perf_counter() - began, result)
    writer.close()
    return net


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", nargs="?", help="an experiment file (.toml); flags override it")
    parser.add_argument("--scenario")
    parser.add_argument("--envs", type=int)
    parser.add_argument("--updates", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--config", help="JSON overrides of the allocate config, replaces [allocate]")
    parser.add_argument("--out", help="default: runs/<name of the experiment file>, or runs/planner")
    parser.add_argument("--force", action="store_true", help="write over a run that is already in --out")
    parser.add_argument("--teacher", choices=["greedy", "rollout"], help="imitate this planner before PPO")
    parser.add_argument("--imitation-steps", type=int)
    parser.add_argument("--imitation-envs", type=int)
    parser.add_argument("--init", help="start from the weights of this checkpoint")
    parser.add_argument("--critic-updates", type=int, help="updates that train only the value function")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--entropy", type=float)
    args = parser.parse_args()
    # A file states its task in full ([allocate] on top of the simulator's defaults); without a
    # file the task is TASK.
    exp = experiment.training(args.experiment, "planner", "allocate") if args.experiment else {"allocate": TASK}
    given = {k: v for k, v in vars(args).items() if v is not None}
    run = experiment.merged(RUN, exp.get("run", {}), "run") | {k: given[k] for k in RUN if k in given}
    net = experiment.merged(NET, exp.get("net", {}), "net")
    ppo = experiment.merged(dataclasses.asdict(PPO), exp.get("ppo", {}), "ppo")
    ppo |= {k: given[flag] for k, flag in (("lr", "lr"), ("ent_coef", "entropy")) if flag in given}
    train(
        run["scenario"], run["envs"], run["updates"], run["seed"],
        out=experiment.claim(args.out or f"runs/{Path(args.experiment).stem if args.experiment else 'planner'}", args.force),
        cfg=PPOConfig(**ppo), reward=experiment.merged(REWARD, exp.get("reward", {}), "reward"),
        config=json.loads(args.config) if args.config else exp.get("allocate", {}), sim=exp.get("sim"),
        teacher=run["teacher"], imitation_steps=run["imitation_steps"], init=run["init"], critic_updates=run["critic_updates"],
        hidden=net["hidden"], imitation_envs=run["imitation_envs"],
    )
