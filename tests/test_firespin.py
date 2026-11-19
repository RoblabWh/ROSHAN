"""The FireSPIN fire model under both tasks, chosen through the simulation config."""

import json

import numpy as np
import torch

from roshan import AllocateEnv, NavigateEnv, _core
from roshan.eval import evaluate
from roshan.scenario import Scenario

SPIN = {"fire_model": "firespin"}
CONFIG = {"time_limit_s": 900.0}


def scenario():
    fuel = np.full((40, 60), _core.fuel_models().index("GR2") + 1, np.uint8)
    block = [(x, y) for x in (29, 30, 31) for y in (19, 20, 21)]  # a single cell often dies out
    return Scenario(fuel=fuel, stations=[(3, 5)], ignitions=block)


def test_allocate_runs_on_firespin_and_drones_save_area():
    env = AllocateEnv(2, scenario(), sim=SPIN, config=CONFIG)
    assert env.spec["sim"]["fire_model"] == "firespin" and env.spec["sim"]["spin"]["dt_s"] == 0.5
    seeds = [1, 2, 3, 4]
    greedy = evaluate(scenario(), "greedy", seeds, CONFIG, SPIN)
    again = evaluate(scenario(), "greedy", seeds, CONFIG, SPIN, threads=1, batch=1)
    assert json.dumps(greedy) == json.dumps(again), "same seeds, same episodes for any batch size and thread count"
    assert all(r["Burned_NoAction"] > 20 for r in greedy), "the fire spreads when left alone"
    assert sum(r["Drops"] for r in greedy) > 0
    assert sum(r["Burned_Cells"] for r in greedy) < sum(r["Burned_NoAction"] for r in greedy)
    rothermel = evaluate(scenario(), "none", seeds, CONFIG)
    assert [r["Burned_Cells"] for r in rothermel] != [r["Burned_NoAction"] for r in greedy]
    # A pre-burn works on the particle model too: the episode starts with more cells alight.
    burned = evaluate(scenario(), "none", seeds, {**CONFIG, "preburn_s": 600.0}, SPIN)
    assert all(r["Burned_Start"] > 9 and r["Time"] == 900.0 for r in burned)


def test_navigate_runs_on_firespin():
    env = NavigateEnv(4, sim=SPIN)
    assert env.spec["sim"]["fire_model"] == "firespin"
    for _ in range(50):
        env.step(torch.zeros_like(env.actions))
    assert torch.isfinite(env.agent).all()
