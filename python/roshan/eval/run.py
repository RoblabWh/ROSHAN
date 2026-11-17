"""Run a planner on a scenario over a fixed list of seeds and write one CSV row per episode.

    uv run python -m roshan.eval --scenario scenarios/x.npz --policy greedy --split val --out greedy.csv
    uv run python -m roshan.eval --scenario scenarios/x.npz --policy runs/planner/planner.pt --out learned.csv

Every seed runs exactly once. The same seeds are also run with idle drones, which gives
`Burned_NoAction` and `Area_Saved` = 1 - burned / burned without action. The episode of a
seed is the same fire for every policy (checked through `Fire_Fingerprint`).
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from roshan import AllocateEnv
from roshan.baselines import POLICIES
from roshan.scenario import Scenario

SPLITS = Path(__file__).resolve().parents[3] / "splits"
FIELDS = ["Episode_Index", "Episode_Seed", "Fire_Fingerprint", "Success", "Failure_Reason", "Time",
          "Percent_Burned", "Burned_Cells", "Burned_NoAction", "Area_Saved", "Water_L", "Drops", "Min_Separation"]


def seeds_of(split):
    """The frozen seed list of a split (`val` or `test`); training never uses these seeds."""
    return [int(line) for line in (SPLITS / f"{split}.txt").read_text().split()]


def _run(scenario, policy, seeds, config, sim, threads, batch):
    rows = {}
    for start in range(0, len(seeds), batch):
        chunk = seeds[start : start + batch]
        env = AllocateEnv(len(chunk), scenario, threads=threads, config=config, sim=sim)
        env.autoreset = False
        env.reset(chunk)
        fingerprint = env.fingerprint.clone()
        running = torch.ones(len(chunk), dtype=torch.bool)
        while running.any():
            env.step(policy(env))
            ended = running & ((env.terminated | env.truncated) > 0)
            for e in ended.nonzero().flatten().tolist():
                info = dict(zip(env.spec["info"], env.info[e].tolist()))
                rows[chunk[e]] = {**info, "fingerprint": int(fingerprint[e])}
            running &= ~ended
    return rows


def _run_flown(scenario, policy, fly, seeds, config, sim, threads, batch, device):
    """Like `_run`, but the drones are flown by the fly network `fly` instead of the scripted
    controller: every tick the network gets what it would see in the navigate task and its
    best-guess velocities are applied."""
    rows = {}
    view_m = 50.0  # the view the fly policy was trained with
    for start in range(0, len(seeds), batch):
        chunk = seeds[start : start + batch]
        env = AllocateEnv(len(chunk), scenario, threads=threads, config=config, sim=sim)
        env.autoreset = False
        env.reset(chunk)
        fingerprint = env.fingerprint.clone()
        e, d = env.actions.shape
        env.fly_tick(torch.zeros(e, d, 2), view_m=view_m)  # one hovering tick fills the pilot's view
        running = torch.ones(e, dtype=torch.bool)
        while running.any():
            agent, neighbors, mask = (t.to(device).view(e * d, *t.shape[2:]) for t in env.pilot)
            with torch.no_grad():
                velocity = torch.tanh(fly(agent, neighbors, mask)[0]).view(e, d, 2)
            env.fly_tick(velocity, policy(env) if env.decide.any() else None, view_m)
            ended = running & ((env.terminated | env.truncated) > 0)
            for i in ended.nonzero().flatten().tolist():
                info = dict(zip(env.spec["info"], env.info[i].tolist()))
                rows[chunk[i]] = {**info, "fingerprint": int(fingerprint[i])}
            running &= ~ended
    return rows


def evaluate(scenario, policy, seeds, config=None, sim=None, threads=0, batch=256, fly=None, device="cpu"):
    """Episode rows (dicts with the CSV fields) for `policy` on `seeds`, in the order given.

    `policy` is a name from roshan.baselines.POLICIES or a function env -> actions. With
    `fly` (a fly network on `device`) the drones are flown by it instead of the scripted
    controller; `Min_Separation` below the collision distance then means drones collided.
    """
    plan = POLICIES[policy] if isinstance(policy, str) else policy
    if fly is None:
        acted = _run(scenario, plan, seeds, config, sim, threads, batch)
    else:
        acted = _run_flown(scenario, plan, fly, seeds, config, sim, threads, batch, device)
    idle = acted if plan is POLICIES["none"] else _run(scenario, POLICIES["none"], seeds, config, sim, threads, batch)
    rows = []
    for index, seed in enumerate(seeds, start=1):
        a, i = acted[seed], idle[seed]
        if a["fingerprint"] != i["fingerprint"]:
            raise RuntimeError(f"seed {seed}: the fire differed between the policy and the idle run")
        out = bool(a["fire_out"])
        rows.append({
            "Episode_Index": index,
            "Episode_Seed": seed,
            "Fire_Fingerprint": a["fingerprint"],
            "Success": float(out),
            "Failure_Reason": "None" if out else "Timeout",
            "Time": a["seconds"],
            "Percent_Burned": a["burned_fraction"],
            "Burned_Cells": a["burned_cells"],
            "Burned_NoAction": i["burned_cells"],
            "Area_Saved": 1.0 - a["burned_cells"] / i["burned_cells"] if i["burned_cells"] else 0.0,
            "Water_L": a["water_l"],
            "Drops": a["drops"],
            "Min_Separation": a["min_separation_m"],
        })
    return rows


def write_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--policy", required=True, help=f"one of {sorted(POLICIES)}, or a planner checkpoint (.pt)")
    parser.add_argument("--sample", action="store_true", help="sample from a learned planner instead of taking its best guess")
    parser.add_argument("--horizon", type=float, default=120.0, help="lookahead of the rollout planner, seconds")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--n", type=int, help="use only the first N seeds of the split")
    parser.add_argument("--config", default="{}", help="JSON overrides of the allocate config")
    parser.add_argument("--sim", default="{}", help="JSON overrides of the simulation config")
    parser.add_argument("--fly", metavar="CHECKPOINT", help="fly the drones with this trained fly policy instead of the scripted controller")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    seeds = seeds_of(args.split)[: args.n]
    policy = args.policy
    if policy == "rollout":
        policy = lambda env: POLICIES["rollout"](env, args.horizon)  # noqa: E731
    elif policy.endswith(".pt"):
        from roshan.rl.train_planner import load, planner_policy

        policy = planner_policy(*load(policy), sample=args.sample)
    elif policy not in POLICIES:
        parser.error(f"unknown policy {policy!r}")
    fly, device = None, "cpu"
    if args.fly:
        from roshan.rl.nets import FlyNet

        device = "cuda" if torch.cuda.is_available() else "cpu"
        fly = FlyNet().to(device)
        fly.load_state_dict(torch.load(args.fly, weights_only=True)["model"])
    rows = evaluate(
        Scenario.load(args.scenario), policy, seeds, json.loads(args.config), json.loads(args.sim), fly=fly, device=device
    )
    path = write_csv(rows, args.out)
    mean = lambda key: float(np.mean([r[key] for r in rows]))  # noqa: E731
    print(f"{path}: {len(rows)} episodes, fire out in {mean('Success'):.1%}, burned {mean('Percent_Burned'):.2%}, "
          f"area saved {mean('Area_Saved'):.1%}, {mean('Drops'):.1f} drops per episode, "
          f"drones closer than 2 m in {np.mean([r['Min_Separation'] < 2.0 for r in rows]):.1%} of episodes")


if __name__ == "__main__":
    main()
