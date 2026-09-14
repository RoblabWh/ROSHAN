"""Checks for the planner assignment baselines (greedy nearest-pair vs Hungarian).

Both read the same observation dict as the pointer network; groundstation is
fire index 0. conftest.py puts src/pysim and build/ on sys.path.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from planner_agent import PlannerAgent

GS = [-0.9, -0.9]


def _state(drones, fires, goals=None, water=None, mask=None):
    drones = np.asarray(drones, np.float32)
    fires = np.asarray(fires, np.float32)
    if goals is None:
        goals = np.tile(fires[0], (len(drones), 1))
    s = {"drone_positions": drones[None], "fire_positions": fires[None],
         "goal_positions": np.asarray(goals, np.float32)[None]}
    if mask is not None:
        s["fire_positions_mask"] = np.asarray(mask, bool)[None]
    if water is not None:
        s["drone_water"] = np.asarray(water, np.float32)[None, :, None]
    return s


def _agent(n=3, threshold=0.0):
    a = PlannerAgent(num_drones=n)
    a.water_refuel_threshold = threshold
    return a


def _rows(actions, fires):
    """Index of each action row in the fire table (asserts every action is a table entry)."""
    fires = np.asarray(fires, np.float32)
    idx = []
    for a in actions[0]:
        hits = np.flatnonzero(np.all(np.isclose(fires, a), axis=1))
        assert len(hits) == 1, f"action {a} not a fire/GS coordinate"
        idx.append(int(hits[0]))
    return idx


@pytest.mark.parametrize("method", ["greedy", "hungarian"])
def test_shape_and_distinct_fires(method):
    fires = [GS, [0.1, 0.2], [-0.3, 0.4], [0.5, -0.5], [0.7, 0.7]]
    drones = [[0.0, 0.0], [0.6, 0.6], [-0.2, 0.3]]
    actions = getattr(_agent(), f"{method}_actions")(_state(drones, fires))
    assert actions.shape == (1, 3, 2) and actions.dtype == np.float32
    rows = _rows(actions, fires)
    assert 0 not in rows, "no drone should idle at GS while fires remain"
    assert len(set(rows)) == 3, "fires must be distinct"


def test_hungarian_lower_total_distance_than_greedy():
    # Greedy grabs the globally nearest pair (d1->f0, 0.05) and strands d0 on f1 (0.9);
    # Hungarian pays 0.45 + 0.4 instead.
    fires = [GS, [-0.45, 0.0], [0.0, 0.0]]
    drones = [[-0.9, 0.0], [-0.4, 0.0]]
    a = _agent(n=2)
    g = a.greedy_actions(_state(drones, fires))
    h = a.hungarian_actions(_state(drones, fires))
    dist = lambda act: float(np.linalg.norm(act[0] - np.asarray(drones), axis=-1).sum())
    assert _rows(g, fires) == [2, 1]
    assert _rows(h, fires) == [1, 2]
    assert dist(h) < dist(g)


@pytest.mark.parametrize("method", ["greedy", "hungarian"])
def test_low_water_drone_goes_to_groundstation(method):
    fires = [GS, [0.1, 0.2], [-0.3, 0.4], [0.5, -0.5]]
    drones = [[0.0, 0.0], [0.6, 0.6], [-0.2, 0.3]]
    a = _agent(threshold=0.25)
    actions = getattr(a, f"{method}_actions")(_state(drones, fires, water=[0.1, 1.0, 1.0]))
    rows = _rows(actions, fires)
    assert rows[0] == 0 and 0 not in rows[1:]


@pytest.mark.parametrize("method", ["greedy", "hungarian"])
def test_committed_fire_not_reassigned(method):
    fires = [GS, [0.1, 0.2], [-0.3, 0.4], [0.5, -0.5]]
    drones = [[0.0, 0.0], [0.6, 0.6], [-0.2, 0.3]]
    goals = [fires[3], GS, GS]  # drone 0 committed to fire 3, not there yet
    actions = getattr(_agent(), f"{method}_actions")(_state(drones, fires, goals=goals))
    assert 3 not in _rows(actions, fires)


@pytest.mark.parametrize("method", ["greedy", "hungarian"])
def test_masked_fires_and_surplus_drones(method):
    fires = [GS, [0.1, 0.2], [0.9, 0.9]]  # fire 2 is padding
    drones = [[0.0, 0.0], [0.6, 0.6], [-0.2, 0.3]]
    actions = getattr(_agent(), f"{method}_actions")(_state(drones, fires, mask=[1, 1, 0]))
    rows = _rows(actions, fires)
    assert rows.count(1) == 1 and rows.count(0) == 2 and 2 not in rows
