"""Train the planner on the allocate task.

    uv run python -m roshan.rl.train_planner --scenario scenarios/demo.npz --updates 300 --out runs/planner
"""

import argparse
import dataclasses
import json
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from roshan import AllocateEnv
from roshan.rl.nets import PlannerNet
from roshan.rl.ppo import PPOConfig, gae, update
from roshan.scenario import Scenario

# Reward per environment and step = sum over terms of weight * term (names from
# env.spec["terms"]). `burned` is the share of the burnable area that caught fire during the
# step, so -100 charges one point per percent burned: the reward is the objective itself.
REWARD = {"drop": 0.0, "refill": 0.0, "burned": -100.0, "fire_out": 1.0, "timeout": 0.0}
# The task the planner is trained and compared on (see the results in docs/systems.md).
TASK = {"random_ignitions": 6, "random_wind": True}
PPO = PPOConfig(lr=3e-4, gamma_per_s=0.999, lam=0.95, epochs=4, minibatches=4, ent_coef=0.01, rollout=64)


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


def imitate(net, optimizer, scenario, device, teacher="rollout", steps=800, num_envs=128, seed=0, config=None, log_every=100):
    """Teaches the network to choose what a built-in planner (`teacher`) would choose.

    The environments are first driven by the teacher and, from the halfway point on, by the
    network itself, while the teacher keeps labelling every state. That way the network also
    sees, and learns to recover from, the situations its own mistakes lead to (DAgger).
    Returns the share of decisions on which the network agreed with the teacher at the end.
    """
    env = AllocateEnv(num_envs, scenario, seed=seed + 1_000, config=config if config is not None else TASK)
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
    net = PlannerNet().to(device)
    net.load_state_dict(saved["model"])
    return net.eval(), device


def train(scenario, num_envs=512, updates=300, seed=0, out="runs/planner", cfg=PPO, reward=None, config=None, log_every=10,
          teacher=None, imitation_steps=800, init=None, critic_updates=0):
    """Trains a planner with PPO. With `teacher` ("greedy" or "rollout") the network first
    imitates that planner for `imitation_steps` environment steps; `init` starts from saved weights
    instead. During the first `critic_updates` updates only the value function learns: a
    policy that was imitated comes with an untrained critic, whose random advantages would
    otherwise undo it."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    env = AllocateEnv(num_envs, scenario, seed=seed, config=config if config is not None else TASK)
    e, d, keep = num_envs, env.spec["drones"], env.spec["slots"]
    t_len = cfg.rollout
    weights = torch.tensor([(reward or REWARD)[name] for name in env.spec["terms"]], device=device)
    # Team terms (the last three) are the same for every drone; drone terms are summed.
    per_drone = torch.tensor([1.0, 1.0, 1.0 / d, 1.0 / d, 1.0 / d], device=device)

    net = PlannerNet().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr, eps=1e-5)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(out / "tensorboard")
    if init:
        net.load_state_dict(torch.load(init, weights_only=True)["model"])
    if teacher:
        imitate(net, optimizer, scenario, device, teacher, imitation_steps, seed=seed, config=config)
        torch.save({"model": net.state_dict(), "spec": env.spec, "schema_version": 1}, out / "imitation.pt")

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
            episodes, fire_out, burned, drops = 0, 0.0, 0.0, 0.0

    torch.save(
        {
            "model": net.state_dict(),
            "net": "roshan.rl.nets.PlannerNet",
            "spec": env.spec,
            "ppo": dataclasses.asdict(cfg),
            "reward": reward or REWARD,
            "schema_version": 1,
        },
        out / "planner.pt",
    )
    writer.close()
    return net


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="scenarios/demo.npz")
    parser.add_argument("--envs", type=int, default=512)
    parser.add_argument("--updates", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", default=json.dumps(TASK), help="JSON overrides of the allocate config")
    parser.add_argument("--out", default="runs/planner")
    parser.add_argument("--teacher", choices=["greedy", "rollout"], help="imitate this planner before PPO")
    parser.add_argument("--imitation-steps", type=int, default=800)
    parser.add_argument("--init", help="start from the weights of this checkpoint")
    parser.add_argument("--critic-updates", type=int, default=0, help="updates that train only the value function")
    parser.add_argument("--lr", type=float, default=PPO.lr)
    parser.add_argument("--entropy", type=float, default=PPO.ent_coef)
    args = parser.parse_args()
    train(
        Scenario.load(args.scenario), args.envs, args.updates, args.seed, args.out,
        cfg=dataclasses.replace(PPO, lr=args.lr, ent_coef=args.entropy), config=json.loads(args.config),
        teacher=args.teacher, imitation_steps=args.imitation_steps, init=args.init, critic_updates=args.critic_updates,
    )
