"""Tunes the numbers of the triage planner on training seeds, by the cross-entropy method.

    uv run python -m roshan.tune experiments/eval_demo.toml
    uv run python -m roshan.tune experiments/eval_demo.toml --config '{"drones_per_station": 80, "pads": 16}'
    uv run python -m roshan.tune experiments/eval_demo.toml --fix per_neighbour_m   # leave a number as the file has it

Every candidate runs the same seeds (0, 1, 2, ...: never the validation or test seeds), so
candidates differ by what they decide and not by the fires they met. Prints the best numbers
as the `[allocate.triage]` table of an experiment file and checks them on the validation seeds.
"""

import argparse
import json

import numpy as np

from roshan.eval.run import evaluate, idle_run, seeds_of
from roshan.experiment import ROOT, load
from roshan.scenario import Scenario

NAMES = ["cap_kw_m", "per_mw_m", "per_drop_m", "per_neighbour_m"]
# How far the first candidates are spread around the starting values, and the lowest value allowed.
SPREAD = np.array([1000.0, 100.0, 15.0, 40.0])
LOWEST = np.array([0.0, -np.inf, -np.inf, -np.inf])


def saved(scenario, config, sim, seeds, idle, weights=None, policy="triage"):
    """Mean share of the area saved on `seeds`, with these triage numbers (a dict) if given."""
    if weights is not None:
        config = config | {"triage": weights}
    return float(np.mean([r["Area_Saved"] for r in evaluate(scenario, policy, seeds, config, sim, idle=idle)]))


def tune(scenario, config, sim, seeds, generations=12, population=24, elite=6, fix=(), rng_seed=0, log=print):
    """The best triage numbers found (a dict) and their area saved on `seeds`."""
    rng = np.random.default_rng(rng_seed)
    idle = idle_run(scenario, seeds, config, sim)
    free = np.array([name not in fix for name in NAMES])
    mean = np.array([config["triage"][name] for name in NAMES], dtype=float)
    std = SPREAD * free
    best = (saved(scenario, config, sim, seeds, idle), mean.copy())
    log(f"start      {100 * best[0]:5.1f} %  {dict(zip(NAMES, mean.round(1)))}")
    for generation in range(generations):
        tries = np.maximum(mean + std * rng.standard_normal((population, len(NAMES))), LOWEST)
        scores = np.array([saved(scenario, config, sim, seeds, idle, dict(zip(NAMES, map(float, t)))) for t in tries])
        top = tries[np.argsort(-scores)[:elite]]
        mean, std = top.mean(0), top.std(0) * free
        if scores.max() > best[0]:
            best = (float(scores.max()), tries[scores.argmax()].copy())
        log(f"generation {generation + 1:2d}: best {100 * scores.max():5.1f} %, mean of the best {elite} {dict(zip(NAMES, mean.round(1)))}")
    return dict(zip(NAMES, (float(round(v, 1)) for v in best[1]))), best[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("experiment", help="an evaluation file: its scenario, [sim] and [allocate] are the task")
    parser.add_argument("--config", default="{}", help="JSON overrides of [allocate]")
    parser.add_argument("--seeds", type=int, default=1000, help="training seeds per candidate")
    parser.add_argument("--generations", type=int, default=12)
    parser.add_argument("--fix", nargs="*", default=[], choices=NAMES, help="numbers to leave as they are")
    args = parser.parse_args()
    exp = load(args.experiment, ["eval", "sim", "allocate", "arms"])
    config, sim = exp["allocate"] | json.loads(args.config), exp.get("sim", {})
    scenario = Scenario.load(ROOT / exp["eval"]["scenario"])
    weights, score = tune(scenario, config, sim, list(range(args.seeds)), args.generations, fix=args.fix)
    print(f"\nbest on {args.seeds} training seeds: {100 * score:.1f} % of the area saved\n\n[allocate.triage]")
    print("".join(f"{name} = {value}\n" for name, value in weights.items()))
    val = seeds_of("val")
    idle = idle_run(scenario, val, config, sim)
    for label, policy, w in (("greedy", "greedy", None), ("triage as the file has it", "triage", None), ("triage, tuned", "triage", weights)):
        print(f"validation, {label}: {100 * saved(scenario, config, sim, val, idle, w, policy):.1f} %")


if __name__ == "__main__":
    main()
