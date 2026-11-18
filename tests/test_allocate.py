"""The allocate task from Python: baselines, the environment and the evaluation runner."""

import numpy as np
import pytest
import torch

from roshan import AllocateEnv, _core
from roshan.baselines import assign
from roshan.eval import evaluate, write_csv
from roshan.eval import paired as pe
from roshan.eval.run import seeds_of
from roshan.scenario import Scenario

STATION = [10.0, 10.0]
CONFIG = {"time_limit_s": 600.0, "random_ignitions": 4, "random_wind": True}


def grassland(drones=4):
    fuel = np.full((40, 60), _core.fuel_models().index("GR2") + 1, np.uint8)
    return Scenario(fuel=fuel, drones=[(2 + i, 2) for i in range(drones)], stations=[(3, 5), (56, 36)],
                    ignitions=[(30, 20)])


def plan(method, drones, fires, water=None, busy=None, mask=None):
    """Slots chosen for drones at `drones` with fires at `fires`; slot 0 is the station."""
    targets = np.array([STATION, *fires], np.float32)
    n = len(drones)
    water = np.ones(n, np.float32) if water is None else np.asarray(water, np.float32)
    task_xy, kind, decide = np.zeros((n, 2), np.float32), np.zeros(n, np.float32), np.ones(n, bool)
    for i, slot in (busy or {}).items():  # drone i is already flying to that slot
        task_xy[i], kind[i], decide[i] = targets[slot], 1.0, False
    mask = np.ones(len(targets), bool) if mask is None else np.asarray(mask, bool)
    return assign(np.asarray(drones, np.float32), water, task_xy, kind, decide, water >= 0.5,
                  targets, mask, 1, method).tolist()


@pytest.mark.parametrize("method", ["greedy", "hungarian"])
def test_drones_take_distinct_fires(method):
    slots = plan(method, [[0, 0], [600, 600], [-200, 300]], [[100, 200], [-300, 400], [500, -500], [700, 700]])
    assert 0 not in slots and -1 not in slots, "nobody idles or refills while fires remain"
    assert len(set(slots)) == 3


def test_hungarian_travels_less_than_greedy():
    # Greedy grabs the closest pair (drone 1, fire 2: 50 m) and strands drone 0 on the far fire.
    drones, fires = [[-900, 0], [-400, 0]], [[-450, 0], [0, 0]]
    targets = np.array([STATION, *fires])
    travel = lambda slots: sum(np.linalg.norm(targets[s] - d) for s, d in zip(slots, np.array(drones)))  # noqa: E731
    greedy, hungarian = plan("greedy", drones, fires), plan("hungarian", drones, fires)
    assert greedy == [2, 1] and hungarian == [1, 2]
    assert travel(hungarian) < travel(greedy)


@pytest.mark.parametrize("method", ["greedy", "hungarian"])
def test_a_drone_without_water_goes_to_the_station(method):
    slots = plan(method, [[0, 0], [600, 600], [-200, 300]], [[100, 200], [-300, 400], [500, -500]], water=[0.1, 1, 1])
    assert slots[0] == 0 and 0 not in slots[1:]


@pytest.mark.parametrize("method", ["greedy", "hungarian"])
def test_a_fire_someone_is_flying_to_is_left_alone(method):
    slots = plan(method, [[0, 0], [600, 600], [-200, 300]], [[100, 200], [-300, 400], [500, -500]], busy={0: 3})
    assert slots[0] == -1, "the busy drone is not asked"
    assert 3 not in slots[1:]


@pytest.mark.parametrize("method", ["greedy", "hungarian"])
def test_empty_slots_are_skipped_and_spare_drones_wait(method):
    slots = plan(method, [[0, 0], [600, 600], [-200, 300]], [[100, 200], [900, 900]], mask=[1, 1, 0])
    assert slots.count(1) == 1 and slots.count(-1) == 2 and 2 not in slots


def test_unknown_method_is_an_error():
    with pytest.raises(ValueError, match="unknown method"):
        plan("random", [[0, 0]], [[1, 1]])


def test_environment_buffers_and_seeds():
    env = AllocateEnv(3, grassland(), config=CONFIG)
    spec = env.spec
    assert (spec["drones"], spec["stations"], spec["slots"]) == (4, 2, 2 + spec["config"]["max_targets"])
    assert spec["size_m"] == (600.0, 400.0)
    assert env.decide.all(), "every drone decides at the start"
    assert env.target_mask[:, :2].all() and env.targets[:, :2, 2].all(), "stations fill the first slots"

    env.reset([5, 5, 6])
    assert env.seed.tolist() == [5, 5, 6]
    assert torch.equal(env.targets[0], env.targets[1]) and not torch.equal(env.targets[0], env.targets[2])
    assert env.fingerprint[0] == env.fingerprint[1] != env.fingerprint[2]
    with pytest.raises(ValueError, match="one seed per environment"):
        env.reset([1, 2])
    with pytest.raises(ValueError, match="unknown field"):
        AllocateEnv(1, grassland(), config={"tank": 1})
    with pytest.raises(ValueError, match="no drones"):
        AllocateEnv(1, grassland(drones=0))


def test_evaluation_is_the_same_for_any_batch_size_and_thread_count():
    seeds = seeds_of("val")[:12]
    rows = evaluate(grassland(), "greedy", seeds, CONFIG)
    assert rows == evaluate(grassland(), "greedy", seeds, CONFIG, batch=1, threads=1)
    assert [r["Episode_Seed"] for r in rows] == seeds and len({r["Fire_Fingerprint"] for r in rows}) == 12
    assert set(seeds_of("val")).isdisjoint(seeds_of("test"))


def test_a_preburn_starts_episodes_with_a_spread_fire_that_does_not_count_against_the_planner():
    seeds = seeds_of("val")[:12]
    config = {**CONFIG, "preburn_s": 300.0}
    idle, greedy = (evaluate(grassland(), name, seeds, config) for name in ("none", "greedy"))
    assert greedy == evaluate(grassland(), "greedy", seeds, config, batch=1, threads=1)
    plain = evaluate(grassland(), "none", seeds, CONFIG)
    for i, g, p in zip(idle, greedy, plain):
        assert i["Burned_Start"] == g["Burned_Start"] > p["Burned_Start"] == 4
        assert i["Time"] == 600.0 and i["Area_Saved"] == 0.0, "an idle episode saves nothing and runs full length"
        # Saved area is measured on what caught fire after the start, so it can reach 1.
        new, new_idle = g["Burned_Cells"] - g["Burned_Start"], g["Burned_NoAction"] - g["Burned_Start"]
        assert g["Area_Saved"] == 1.0 - new / new_idle
    assert all(0.0 < g["Area_Saved"] < 1.0 for g in greedy), "a head start makes the fire harder, not hopeless"


def test_planners_save_area_and_the_paired_table_accepts_the_files(tmp_path):
    seeds = seeds_of("val")[:24]
    arms = {name: evaluate(grassland(), name, seeds, CONFIG) for name in ("none", "greedy", "hungarian")}
    burned = {name: np.mean([r["Percent_Burned"] for r in rows]) for name, rows in arms.items()}
    assert burned["greedy"] < 0.5 * burned["none"] and burned["hungarian"] < 0.5 * burned["none"]
    assert all(r["Area_Saved"] == 0.0 and r["Drops"] == 0 for r in arms["none"])
    assert np.mean([r["Area_Saved"] for r in arms["greedy"]]) > 0.5
    assert all(r["Min_Separation"] >= 3.0 - 1e-3 for rows in arms.values() for r in rows)

    files = {name: str(write_csv(rows, tmp_path / f"{name}.csv")) for name, rows in arms.items()}
    a, b = pe.pair_arms([pe.load_csv(files["greedy"])], [pe.load_csv(files["hungarian"])])
    stats = pe.paired_stats(a, b, 0.95, 200, np.random.default_rng(0))
    assert stats["n"] == 24 and "burned_diff_mean" in stats
    assert pe.main(["--arm", "greedy", files["greedy"], "--arm", "hungarian", files["hungarian"], "--boot", "200"]) == 0


def test_rollout_planner_runs_through_the_evaluation_and_holds_its_own():
    seeds = seeds_of("val")[:16]
    burned = {name: np.mean([r["Percent_Burned"] for r in evaluate(grassland(), name, seeds, CONFIG)])
              for name in ("greedy", "rollout")}
    assert burned["rollout"] <= 1.1 * burned["greedy"], burned
    env = AllocateEnv(2, grassland(), config=CONFIG)
    assert env.plan("greedy").shape == (2, 4) and env.plan("rollout", 60.0, 2).dtype == torch.int32
    with pytest.raises(ValueError, match="expected"):
        env.plan("magic")
