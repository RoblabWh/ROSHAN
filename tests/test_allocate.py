"""The allocate task from Python: the environment, the built-in planners and the evaluation runner."""

import json

import numpy as np
import pytest
import torch

from roshan import AllocateEnv, _core
from roshan.baselines import POLICIES
from roshan.eval import evaluate, write_csv
from roshan.eval import paired as pe
from roshan.eval.run import seeds_of
from roshan.scenario import Scenario

# 16 drones (8 at each of two stations) on four random fires for ten minutes.
CONFIG = {"time_limit_s": 600.0, "random_ignitions": 4, "random_wind": True, "drones_per_station": 8}
DOCKED = 4.0


def grassland(stations=((3, 5), (56, 36))):
    fuel = np.full((40, 60), _core.fuel_models().index("GR2") + 1, np.uint8)
    return Scenario(fuel=fuel, stations=list(stations), ignitions=[(30, 20)])


def same(a, b):
    """Equal rows; a row holds NaN where nothing was measured."""
    return json.dumps(a) == json.dumps(b)


def test_environment_buffers_and_seeds():
    env = AllocateEnv(3, grassland(), config=CONFIG)
    spec = env.spec
    assert (spec["drones"], spec["stations"], spec["slots"]) == (16, 2, 2 + spec["config"]["max_targets"])
    assert spec["size_m"] == (600.0, 400.0)
    assert env.targets.shape[-1] == len(spec["target"]) == 7 and env.info.shape[-1] == len(spec["info"]) == 15
    assert env.decide.all(), "every drone decides at the start"
    assert (env.drones[..., 5] == DOCKED).all() and (env.drones[..., 2] == 1).all(), "docked with a full tank"
    assert env.target_mask[:, :2].all() and env.targets[:, :2, 2].all(), "stations fill the first slots"
    assert not env.action_mask[:, :, :2].any(), "a docked drone cannot be sent to a station"
    assert (env.action_mask[:, :, 2:].sum(-1) == 4).all(), "the four fires are open to everyone"

    env.reset([5, 5, 6])
    assert env.seed.tolist() == [5, 5, 6]
    assert torch.equal(env.targets[0], env.targets[1]) and not torch.equal(env.targets[0], env.targets[2])
    assert env.fingerprint[0] == env.fingerprint[1] != env.fingerprint[2]
    with pytest.raises(ValueError, match="one seed per environment"):
        env.reset([1, 2])
    with pytest.raises(ValueError, match="unknown field"):
        AllocateEnv(1, grassland(), config={"tank": 1})
    with pytest.raises(ValueError, match="no ground station"):
        AllocateEnv(1, grassland(stations=()))
    with pytest.raises(ValueError, match="more than the tank holds"):
        AllocateEnv(1, grassland(), config={"drop_l": 50.0})


def test_the_planners_send_what_the_fires_need():
    env = AllocateEnv(2, grassland(), config=CONFIG)
    need = env.targets[:, 2:6, 4]  # drops each of the four fires needs
    assert torch.allclose(need, torch.full_like(need, 0.4 * np.exp(0.1) / 0.28), atol=1e-4)
    for name in ("greedy", "head", "flank", "rollout"):
        actions = POLICIES[name](env)
        assert actions.shape == (2, 16) and actions.dtype == torch.int32
        sent = actions[actions >= 0]
        # Two drops per fire: 8 of the 16 drones go, each fire gets two, the rest wait.
        assert len(sent) == 16 and all((sent == slot).sum() == 4 for slot in range(2, 6)), name
    assert (POLICIES["none"](env) == -1).all()
    with pytest.raises(ValueError, match="expected one of"):
        env.plan("hungarian")
    with pytest.raises(ValueError, match="expected one of"):
        env.plan("rollout", 0.0)

    # The waiting drones are asked again every 10 s; meanwhile the others fly and drop.
    drops, inbound = torch.zeros(2), 0.0
    for _ in range(8):
        env.step(POLICIES["greedy"](env))
        assert (env.elapsed_s <= 10.0).all() and env.decide.any()
        drops += env.terms[..., 0].sum(-1)
        inbound = max(inbound, float(env.targets[:, 2:, 5].sum()))
    assert (drops == 8).all(), "eight drops per environment, and no more are sent"
    assert inbound == 16.0, "drops on their way show in the target table"


def test_the_fire_of_a_seed_is_what_it_was_before_the_drones_changed():
    # Recorded on 2026-11-19 with the task as it was before (4 drones with their own start
    # cells): the first validation seeds on the demo map with six random fires.
    env = AllocateEnv(8, Scenario.load("scenarios/demo.npz"), config={"random_ignitions": 6, "random_wind": True})
    env.autoreset = False
    env.reset(seeds_of("val")[:8])
    assert env.fingerprint.tolist() == [290909206, 951932052, 1373252995, 482423397, 1966920333, 327978195,
                                        1881370801, 291257301]


def test_a_scenario_of_the_first_file_version_loads_without_its_drones(tmp_path):
    s = grassland()
    plain = {**s._plain(), "schema_version": 1, "drones": [[2, 2], [3, 2]]}
    np.savez_compressed(tmp_path / "old.npz", plain=json.dumps(plain), fuel=s.fuel)
    assert Scenario.load(tmp_path / "old.npz").fingerprint() == s.fingerprint()
    plain["schema_version"] = 3
    np.savez_compressed(tmp_path / "new.npz", plain=json.dumps(plain), fuel=s.fuel)
    with pytest.raises(ValueError, match="schema version 3"):
        Scenario.load(tmp_path / "new.npz")


def test_evaluation_is_the_same_for_any_batch_size_and_thread_count():
    seeds = seeds_of("val")[:12]
    rows = evaluate(grassland(), "greedy", seeds, CONFIG)
    assert same(rows, evaluate(grassland(), "greedy", seeds, CONFIG, batch=1, threads=1))
    assert [r["Episode_Seed"] for r in rows] == seeds and len({r["Fire_Fingerprint"] for r in rows}) == 12
    assert set(seeds_of("val")).isdisjoint(seeds_of("test"))


def test_a_preburn_starts_episodes_with_a_spread_fire_that_does_not_count_against_the_planner():
    seeds = seeds_of("val")[:12]
    config = {**CONFIG, "preburn_s": 300.0}
    idle, greedy = (evaluate(grassland(), name, seeds, config) for name in ("none", "greedy"))
    assert same(greedy, evaluate(grassland(), "greedy", seeds, config, batch=1, threads=1))
    plain = evaluate(grassland(), "none", seeds, CONFIG)
    for i, g, p in zip(idle, greedy, plain):
        assert i["Burned_Start"] == g["Burned_Start"] > p["Burned_Start"] == 4
        assert i["Time"] == 600.0 and i["Area_Saved"] == 0.0, "an idle episode saves nothing and runs full length"
        # Saved area is measured on what caught fire after the start, so it can reach 1.
        new, new_idle = g["Burned_Cells"] - g["Burned_Start"], g["Burned_NoAction"] - g["Burned_Start"]
        assert g["Area_Saved"] == 1.0 - new / new_idle
        assert g["Peak_kW_NoAction"] == i["Peak_kW"] >= g["Stopped_kW_Max"] > 0
    assert all(0.0 < g["Area_Saved"] < 1.0 for g in greedy), "a head start makes the fire harder, not hopeless"


def test_planners_save_area_and_the_paired_table_accepts_the_files(tmp_path):
    seeds = seeds_of("val")[:24]
    config = {**CONFIG, "preburn_s": 300.0}
    arms = {name: evaluate(grassland(), name, seeds, config) for name in ("none", "greedy", "head", "flank")}
    saved = {name: np.mean([r["Area_Saved"] for r in rows]) for name, rows in arms.items()}
    assert saved["none"] == 0.0 and all(r["Drops"] == 0 and r["Stopped_Cells"] == 0 for r in arms["none"])
    assert np.isnan(arms["none"][0]["First_Drop_s"])
    # Every rule saves area; taking the nearest fire that needs water saves the most here.
    assert saved["greedy"] > 0.3 and saved["greedy"] > max(saved["head"], saved["flank"]) > 0.1, saved
    for rows in (arms["greedy"], arms["head"], arms["flank"]):
        for r in rows:
            assert r["Water_L"] == 40.0 * r["Drops"] and r["Stopped_Cells"] > 0
            assert r["Max_Sortie_s"] <= 900.0 and 0 < r["First_Drop_s"] < 60.0
            assert np.isnan(r["Min_Separation"]), "not watched while the drones fly straight lines"
    head, flank = (np.mean([r["Stopped_kW_Mean"] for r in arms[name]]) for name in ("head", "flank"))
    assert head > flank, "the head rule stops the more intense fronts"

    files = {name: str(write_csv(rows, tmp_path / f"{name}.csv")) for name, rows in arms.items()}
    a, b = pe.pair_arms([pe.load_csv(files["greedy"])], [pe.load_csv(files["head"])])
    stats = pe.paired_stats(a, b, 0.95, 200, np.random.default_rng(0))
    assert stats["n"] == 24 and "burned_diff_mean" in stats
    assert pe.main(["--arm", "greedy", files["greedy"], "--arm", "head", files["head"], "--boot", "200"]) == 0


def test_no_fire_start_is_better_than_put_out():
    # Without a head start two drops per fire put every fire out; a starting cell that is
    # knocked down counts as unburned again, and the saved area stays at 100 %.
    rows = evaluate(grassland(), "greedy", seeds_of("val")[:8], CONFIG)
    assert all(r["Success"] == 1.0 and r["Area_Saved"] == 1.0 and r["Drops"] == 8 for r in rows)


def test_rollout_planner_runs_through_the_evaluation_and_holds_its_own():
    seeds = seeds_of("val")[:16]
    config = {**CONFIG, "preburn_s": 300.0}
    burned = {name: np.mean([r["Percent_Burned"] for r in evaluate(grassland(), name, seeds, config)])
              for name in ("greedy", "rollout")}
    assert burned["rollout"] <= 1.05 * burned["greedy"], burned


def test_separation_is_watched_when_it_is_kept():
    rows = evaluate(grassland(), "greedy", seeds_of("val")[:4], {**CONFIG, "safe_m": 3.0, "preburn_s": 300.0})
    assert all(r["Min_Separation"] >= 3.0 - 1e-3 and r["Drops"] > 20 for r in rows)
