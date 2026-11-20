"""Experiment files: what is written can be read and run again, and a wrong file is refused."""

import json
import tomllib

import pytest

from roshan import AllocateEnv, experiment
from roshan.eval.run import run_experiment
from roshan.scenario import Scenario


def test_text_round_trip():
    tables = {"run": {"kind": "planner", "lr": 1e-05, "init": None, "on": True}, "sim": {"dt_s": 0.1, "spin": {"lt_m": 80.0}},
              "arms": {"a.b": {"policy": "greedy"}}, "eval": {"pair": ["a", "b"]}}
    back = tomllib.loads(experiment.dumps(tables))
    tables["run"].pop("init")  # None is left out
    assert back == tables


def test_a_file_only_runs_where_it_belongs(tmp_path):
    path = experiment.write(tmp_path / "x.toml", {"run": {"kind": "fly"}, "navigate": {"drones": 2}})
    assert experiment.training(path, "fly", "navigate")["navigate"] == {"drones": 2}
    with pytest.raises(ValueError, match="kind"):
        experiment.training(path, "planner", "allocate")
    with pytest.raises(ValueError, match="not used here"):
        experiment.load(path, ["eval", "arms"])
    with pytest.raises(ValueError, match="has no"):
        experiment.merged({"envs": 1}, {"env": 2}, "run")


def test_viewer_settings_round_trip(tmp_path):
    changed = {"allocate": {"tank_l": 250.0}, "sim": {"spin": {"lt_m": 90.0}}}
    experiment.save_world(tmp_path / "s.toml", json.dumps(changed))
    assert json.loads(experiment.world(tmp_path / "s.toml")) == changed
    # Writing over the viewer's own file is fine; over a training file it would drop [run].
    experiment.save_world(tmp_path / "s.toml", json.dumps(changed))
    (tmp_path / "train.toml").write_text("[run]\nenvs = 4\n\n[sim]\ndt_s = 0.1\n")
    with pytest.raises(ValueError, match=r"\[run\]"):
        experiment.save_world(tmp_path / "train.toml", json.dumps(changed))
    assert "[run]" in (tmp_path / "train.toml").read_text()


def test_settings_no_world_can_run_with_are_refused():
    """They used to reach an assert in the simulator (a panic) instead of an error."""
    import roshan
    from roshan.scenario import Scenario

    scenario = Scenario.load(experiment.ROOT / "scenarios/demo.npz")
    with pytest.raises(ValueError, match="spin.dt_s"):
        roshan.AllocateEnv(1, scenario, sim={"spin": {"dt_s": 0.0}})
    with pytest.raises(ValueError, match="goals"):
        roshan.NavigateEnv(1, config={"goals": 0})


def test_evaluation_writes_what_ran(tmp_path):
    exp = {"eval": {"scenario": "scenarios/demo.npz", "n": 4, "pair": ["none", "greedy"]}, "allocate": {"tank_l": 250},
           "arms": {"none": {"policy": "none"}, "greedy": {"policy": "greedy"}}}
    text = run_experiment(exp, tmp_path)
    assert "| greedy |" in text and "paired: greedy - none" in text
    assert (tmp_path / "greedy.csv").exists()
    # The record holds every value, and running it gives the same environment.
    ran = experiment.load(tmp_path / "evaluation.toml", ["eval", "sim", "allocate", "arms"])
    assert ran["allocate"]["tank_l"] == 250 and ran["allocate"]["drop_l"] > 0 and ran["arms"]["greedy"]["horizon"] == 300
    spec = lambda **kw: AllocateEnv(1, Scenario.load("scenarios/demo.npz"), **kw).spec  # noqa: E731
    assert spec(config=ran["allocate"], sim=ran["sim"]) == spec(config=exp["allocate"])
    with pytest.raises(ValueError, match="has no"):
        run_experiment({**exp, "arms": {"x": {"polcy": "greedy"}}}, tmp_path)
    # The record is completed when the run is: results, and the simulator they were made on.
    done = tomllib.loads((tmp_path / "evaluation.toml").read_text())
    assert done["record"]["finished"] and done["record"]["simulator"] == experiment.simulator_digest()
    assert done["result"]["greedy"]["burned"] < done["result"]["none"]["burned"]
    assert (tmp_path / "code.patch").exists() == done["record"]["uncommitted_changes"]
    with pytest.raises(SystemExit, match="already holds"):
        experiment.claim(tmp_path)
    assert experiment.claim(tmp_path, force=True) == tmp_path and experiment.claim(tmp_path / "new")


def _names(tables):
    return {k: _names(v) if isinstance(v, dict) else None for k, v in tables.items()}


@pytest.mark.parametrize("path", sorted(experiment.ROOT.glob("experiments/*.toml")), ids=lambda p: p.name)
def test_example_files_list_every_setting(path):
    """What can be set is what the files in experiments/ show: a new setting has to be added to them."""
    data = tomllib.loads(path.read_text())
    kind = "eval" if "eval" in data else data["run"]["kind"]
    full = experiment.defaults(kind)
    if kind == "eval":
        arm = full["arms"]["greedy"]
        full["arms"] = {name: arm for name in data["arms"]}
    assert _names(data) == _names(full)


def test_maps_of_experiments_are_known(tmp_path):
    """The viewer refuses to overwrite a map that an experiment file runs on."""
    assert "planner.toml" in experiment.uses("scenarios/demo.npz")
    assert experiment.uses(tmp_path / "mine.npz") == []
    assert experiment.scenario_of(experiment.ROOT / "experiments/fly.toml") is None
