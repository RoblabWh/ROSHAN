"""Run planners on a scenario over a fixed list of seeds and write one CSV row per episode.

    uv run python -m roshan.eval experiments/eval_demo.toml           # every arm of the file, into runs/eval_demo/
    uv run python -m roshan.eval --scenario scenarios/x.npz --policy greedy --split val --out greedy.csv
    uv run python -m roshan.eval --scenario scenarios/x.npz --policy runs/planner/planner.pt --out learned.csv

An evaluation file (see roshan.experiment) compares several planners on the same episodes:

    [eval]
    scenario = "scenarios/demo.npz"
    split = "val"                    # or "test"; n = 100 uses only the first 100 seeds
    pair = ["greedy", "learned"]     # optional: paired statistics of two arms

    [allocate]                       # the task, and [sim], on top of the simulator's defaults
    random_ignitions = 6

    [arms.greedy]
    policy = "greedy"                # a name, or a planner checkpoint (.pt)

    [arms.learned]
    policy = "runs/planner/planner.pt"
    sample = false                   # also: horizon, candidates (rollout), fly (checkpoint or "decentral")

It writes one CSV per arm, `summary.md` and `evaluation.toml`: the file with every value filled
in, the digest of each checkpoint, when and from which code it ran, and the mean results. A
directory that already holds an evaluation is refused without `--force`. A checkpoint trained
on a simulator that behaved differently, or with other settings, gets a WARNING line. With flags, the same
record is written next to the CSV (`greedy.evaluation.toml`).

Every seed runs exactly once. The same seeds are also run with idle drones, which gives
`Burned_NoAction`, `Area_Saved` = 1 - burned / burned without action, and `Peak_kW_NoAction`,
the highest fireline intensity of the fire left alone. The summary shows the results by
`Peak_kW_Start`, the intensity of the fire as the drones find it, because small drones can
only stop weak fronts. The episode of a seed is the same
fire for every policy (checked through `Fire_Fingerprint`).

With a file, `--config`, `--sim`, `--split` and `--n` change it for this run (the record
holds what ran), so a sweep is a loop:

    for n in 10 20 40 80; do uv run python -m roshan.eval experiments/eval_demo.toml \\
        --config "{\"drones_per_station\": $n}" --out runs/team_$n; done
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from roshan import AllocateEnv, _core, experiment
from roshan.baselines import POLICIES
from roshan.scenario import Scenario

SPLITS = Path(__file__).resolve().parents[3] / "splits"
FIELDS = ["Episode_Index", "Episode_Seed", "Fire_Fingerprint", "Success", "Failure_Reason", "Time",
          "Percent_Burned", "Burned_Cells", "Burned_NoAction", "Area_Saved", "Water_L", "Drops", "Min_Separation",
          "Burned_Start", "Stopped_Cells", "Stopped_kW_Mean", "Stopped_kW_Max", "Peak_kW", "Peak_kW_NoAction", "Peak_kW_Start",
          "First_Drop_s", "Queue_s", "Max_Sortie_s", "Detect_s", "Seen_Share", "Belief_Miss", "Belief_Phantom"]
# Classes of the fire's peak fireline intensity when the drones start, kW/m: what drones can
# plausibly stop, what aircraft are credited with, and beyond (docs/systems.md, "Calibration").
CLASSES = [("up to 500 kW/m", 0.0, 500.0), ("500 - 2000 kW/m", 500.0, 2000.0), ("above 2000 kW/m", 2000.0, float("inf"))]


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
    best-guess velocities are applied. `fly="decentral"` is the hand-written pilot that
    decides from the same observation."""
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
            if fly == "decentral":
                c = env.spec["config"]
                seen = (t.numpy() for t in env.pilot)
                velocity = torch.from_numpy(_core.decentral_pilot(*seen, view_m, c["max_speed_ms"], c["safe_m"]))
            else:
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


def idle_run(scenario, seeds, config=None, sim=None, threads=0, batch=256):
    """The episodes with drones that do nothing: what burns without action."""
    return _run(scenario, POLICIES["none"], seeds, config, sim, threads, batch)


def evaluate(scenario, policy, seeds, config=None, sim=None, threads=0, batch=256, fly=None, device="cpu", idle=None):
    """Episode rows (dicts with the CSV fields) for `policy` on `seeds`, in the order given.

    `policy` is a name from roshan.baselines.POLICIES or a function env -> actions. With
    `fly` (a fly network on `device`) the drones are flown by it instead of the scripted
    controller; `Min_Separation` below the collision distance then means drones collided.
    `idle` is the result of `idle_run` on the same seeds, to compute it once for several policies.
    """
    plan = POLICIES[policy] if isinstance(policy, str) else policy
    if fly is None:
        acted = _run(scenario, plan, seeds, config, sim, threads, batch)
    else:
        acted = _run_flown(scenario, plan, fly, seeds, config, sim, threads, batch, device)
    if idle is None:
        idle = acted if plan is POLICIES["none"] else idle_run(scenario, seeds, config, sim, threads, batch)
    rows = []
    for index, seed in enumerate(seeds, start=1):
        a, i = acted[seed], idle[seed]
        if a["fingerprint"] != i["fingerprint"]:
            raise RuntimeError(f"seed {seed}: the fire differed between the policy and the idle run")
        out = bool(a["fire_out"])
        # Only what caught fire after the episode started could have been saved.
        start = a["start_cells"]
        # A starting cell that is put out before its fuel is gone counts as unburned again, so
        # fewer cells than at the start can be burned at the end: everything was saved.
        new, new_idle = max(a["burned_cells"] - start, 0.0), i["burned_cells"] - start
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
            "Area_Saved": 1.0 - new / new_idle if new_idle > 0 else 0.0,
            "Water_L": a["water_l"],
            "Drops": a["drops"],
            "Min_Separation": a["min_separation_m"],
            "Burned_Start": start,
            "Stopped_Cells": a["stopped_cells"],
            "Stopped_kW_Mean": a["stopped_kw_mean"],
            "Stopped_kW_Max": a["stopped_kw_max"],
            "Peak_kW": a["peak_kw"],
            "Peak_kW_NoAction": i["peak_kw"],
            "Peak_kW_Start": a["start_kw"],
            "First_Drop_s": a["first_drop_s"],
            "Queue_s": a["queue_s"],
            "Max_Sortie_s": a["max_sortie_s"],
            "Detect_s": a["detect_s"],
            "Seen_Share": a["seen_share"],
            "Belief_Miss": a["belief_miss"],
            "Belief_Phantom": a["belief_phantom"],
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


# An arm: `policy` is a name from POLICIES or a planner checkpoint; `sample` draws from a learned
# planner instead of taking its best guess; `horizon` (seconds) and `candidates` belong to the
# rollout planner; `fly` is a fly checkpoint, "decentral", or "" for the scripted controller.
ARM = {"policy": "", "sample": False, "horizon": 300.0, "candidates": 4, "fly": ""}
# `n`: only the first n seeds, 0 for all. `pair`: two arm names for paired statistics, or [].
EVAL = {"scenario": "", "split": "val", "n": 0, "pair": []}


def _players(arm):
    """(planner, pilot, device) of an arm."""
    policy, fly, device = arm["policy"], None, "cpu"
    if policy == "rollout":
        policy = lambda env: POLICIES["rollout"](env, arm["horizon"], arm["candidates"])  # noqa: E731
    elif str(policy).endswith(".pt"):
        from roshan.rl.train_planner import load, planner_policy

        policy = planner_policy(*load(policy), sample=arm["sample"])
    elif policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}: one of {sorted(POLICIES)}, or a planner checkpoint (.pt)")
    if arm["fly"] == "decentral":
        fly = "decentral"
    elif arm["fly"]:
        from roshan.rl.train import load

        device = "cuda" if torch.cuda.is_available() else "cpu"
        fly = load(arm["fly"], device)
    return policy, fly, device


def _stale(name, arm, spec):
    """Reasons why the checkpoints of an arm may not fit this evaluation, as lines to warn with:
    trained on a simulator that behaved differently, or with other settings."""
    lines, now = [], experiment.simulator_digest()
    for key in ("policy", "fly"):
        path = str(arm[key])
        if not path.endswith(".pt"):
            continue
        trained = experiment.recorded(Path(path).parent / "experiment.toml").get("simulator")
        if trained is None:
            lines.append(f"{name}: {path} has no record of the simulator it was trained on")
        elif trained != now:
            lines.append(f"{name}: {path} was trained on a simulator that behaved differently (digest {trained}, now {now})")
        if key == "policy":
            saved = torch.load(path, weights_only=True)["spec"]
            shown = lambda v: "other values" if isinstance(v, dict) else "not known then" if v is None else v  # noqa: E731
            differ = [f"{section}.{k}: {shown(saved[was].get(k))} -> {shown(v) if not isinstance(v, dict) else 'now'}"
                      for section, was in (("sim", "sim"), ("allocate", "config"))
                      for k, v in spec[was].items() if saved[was].get(k) != v]
            if differ:
                lines.append(f"{name}: trained with other settings than this evaluation ({', '.join(differ)})")
    return lines


def run_experiment(exp, out, record="evaluation.toml"):
    """Runs every arm of an evaluation (the sections of an evaluation file) into the directory
    `out`: `<arm>.csv` each and the record of what ran. Returns the summary as markdown."""
    out, began = Path(out), time.perf_counter()
    ev = experiment.merged(EVAL, exp.get("eval", {}), "eval")
    arms = {name: experiment.merged(ARM, arm, f"arms.{name}") for name, arm in exp.get("arms", {}).items()}
    if not arms or not ev["scenario"]:
        raise ValueError("an evaluation needs [eval] scenario and at least one [arms.<name>]")
    scenario = Scenario.load(ev["scenario"])
    seeds = seeds_of(ev["split"])[: ev["n"] or None]
    config, sim = exp.get("allocate", {}), exp.get("sim", {})
    players = {name: _players(arm) for name, arm in arms.items()}  # every arm loads before the first one runs
    spec = AllocateEnv(1, scenario, config=config, sim=sim).spec
    digests = {f"{name}_{key}": experiment.digest(arm[key]) for name, arm in arms.items() for key in ("policy", "fly")
               if str(arm[key]).endswith(".pt")}
    warnings = [line for name, arm in arms.items() for line in _stale(name, arm, spec)]
    for line in warnings:
        print(f"WARNING {line}", flush=True)
    tables = {
        "eval": ev,
        "sim": spec["sim"],
        "allocate": spec["config"],
        "arms": arms,
        # The code patch only goes into a directory of its own, not next to a single CSV.
        "record": experiment.record(out if record == "evaluation.toml" else None, scenario_fingerprint=scenario.fingerprint(),
                                    episodes=len(seeds), warnings=warnings, **digests),
    }
    experiment.write(out / record, tables)

    idle, result = idle_run(scenario, seeds, config, sim), {}
    lines = [f"WARNING {line}" for line in warnings] + [""] * bool(warnings)
    c = spec["config"]
    lines += [f"{ev['scenario']}, {len(seeds)} {ev['split']} episodes, {spec['drones']} drones of {c['tank_l']:g} L", "",
             "| arm | fire out | burned | area saved | drops | fronts stopped | at kW/m (mean, highest) | "
             "drones closer than 2 m |", "|---|---|---|---|---|---|---|---|"]
    paths, by_class = {}, []
    for name, (policy, fly, device) in players.items():
        rows = evaluate(scenario, policy, seeds, config, sim, fly=fly, device=device, idle=idle)
        paths[name] = write_csv(rows, out / f"{name}.csv")
        mean = lambda key, rows=rows: float(np.mean([r[key] for r in rows])) if rows else float("nan")  # noqa: E731
        stopped = sum(r["Stopped_Cells"] for r in rows)
        # Per stopped front, not per episode: an episode that stopped nothing has no intensity.
        stopped_kw = sum(r["Stopped_Cells"] * r["Stopped_kW_Mean"] for r in rows) / max(stopped, 1)
        near = [r["Min_Separation"] < 2.0 for r in rows if not np.isnan(r["Min_Separation"])]
        lines.append(f"| {name} | {mean('Success'):.1%} | {mean('Percent_Burned'):.2%} | {mean('Area_Saved'):.1%} | "
                     f"{mean('Drops'):.1f} | {mean('Stopped_Cells'):.1f} | {stopped_kw:.0f}, "
                     f"{max(r['Stopped_kW_Max'] for r in rows):.0f} | "
                     + (f"{np.mean(near):.1%} of episodes |" if near else "not watched |"))
        print(lines[-1], flush=True)
        result[name] = {"fire_out": mean("Success"), "burned": mean("Percent_Burned"), "area_saved": mean("Area_Saved"),
                        "drops": mean("Drops"), "stopped_cells": mean("Stopped_Cells"), "stopped_kw_mean": stopped_kw}
        for label, low, high in CLASSES:
            part = [r for r in rows if low <= r["Peak_kW_Start"] < high]
            by_class.append(f"| {name} | {label} | {len(part)} | {mean('Success', part):.1%} | "
                            f"{mean('Area_Saved', part):.1%} | {mean('Stopped_Cells', part):.1f} |" if part else
                            f"| {name} | {label} | 0 | - | - | - |")
    lines += ["", "By the peak fireline intensity of the fire when the drones start:", "",
              "| arm | fire at the start | episodes | fire out | area saved | fronts stopped |", "|---|---|---|---|---|---|",
              *by_class]
    experiment.finish(out / record, tables, time.perf_counter() - began, result)
    text = "\n".join(lines) + "\n"
    if ev["pair"]:
        from roshan.eval import paired

        a, b = ev["pair"]
        pair = paired.pair_arms([paired.load_csv(paths[a])], [paired.load_csv(paths[b])])
        text += "\n" + paired.render_pair(a, b, paired.paired_stats(*pair, 0.95, 2000, np.random.default_rng(0)), 0.95)
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", nargs="?", help="an evaluation file (.toml); the flags below are for a single run without one")
    parser.add_argument("--scenario")
    parser.add_argument("--policy", help=f"one of {sorted(POLICIES)}, or a planner checkpoint (.pt)")
    parser.add_argument("--sample", action="store_true", help="sample from a learned planner instead of taking its best guess")
    parser.add_argument("--horizon", type=float, default=300.0, help="lookahead of the rollout planner, seconds")
    parser.add_argument("--split", choices=["val", "test"], help="default: val, or the file's")
    parser.add_argument("--n", type=int, help="use only the first N seeds of the split")
    parser.add_argument("--config", default="{}", help="JSON overrides of the allocate config (with a file: of its [allocate])")
    parser.add_argument("--sim", default="{}", help="JSON overrides of the simulation config (with a file: of its [sim])")
    parser.add_argument("--fly", metavar="CHECKPOINT", help="fly the drones with this trained fly policy, or with 'decentral' "
                        "(the hand-written pilot), instead of the scripted controller")
    parser.add_argument("--out", help="with a file: the directory (default runs/<name of the file>); otherwise the CSV")
    parser.add_argument("--force", action="store_true", help="with a file: write over an evaluation that is already in --out")
    args = parser.parse_args()
    if args.experiment:
        out = experiment.claim(args.out or f"runs/{Path(args.experiment).stem}", args.force)
        exp = experiment.load(args.experiment, ["eval", "sim", "allocate", "arms"])
        exp["allocate"] = exp.get("allocate", {}) | json.loads(args.config)
        exp["sim"] = exp.get("sim", {}) | json.loads(args.sim)
        exp["eval"] = exp.get("eval", {}) | {k: v for k, v in (("split", args.split), ("n", args.n)) if v}
        text = run_experiment(exp, out)
        (out / "summary.md").write_text(text)
        print(f"\n{text}\nwritten to {out}/")
        return
    if not (args.scenario and args.policy and args.out):
        parser.error("give an evaluation file, or --scenario, --policy and --out")
    csv_path = Path(args.out).with_suffix(".csv")
    exp = {
        "eval": {"scenario": args.scenario, "split": args.split or "val", "n": args.n or 0},
        "allocate": json.loads(args.config),
        "sim": json.loads(args.sim),
        "arms": {csv_path.stem: {"policy": args.policy, "sample": args.sample, "horizon": args.horizon, "fly": args.fly or ""}},
    }
    print(run_experiment(exp, csv_path.parent, record=f"{csv_path.stem}.evaluation.toml"))


if __name__ == "__main__":
    main()
