"""expand_flags + paths.run_dir: the flag layer must append after positional args, and
--model-dir must bring the checkpoint's own config.yaml in as a base overlay."""
import os
import yaml
from config_loader import expand_flags, split_args, load_config, model_snapshot


def test_expand_flags_appends_after_positionals(tmp_path):
    argv = ["config/agent/planner.yaml", "--eval", "settings.seed=3", "--baseline", "hungarian",
            "--run-dir", "experiments/x/runs/h", "--n", "7", "--model-dir", str(tmp_path), "--model-name", "best_obj"]
    rest, snapshot = expand_flags(argv)
    overlays, dotlist = split_args(rest)
    assert snapshot is None  # no config.yaml in tmp_path
    assert overlays == ["config/agent/planner.yaml", "config/mode/eval.yaml", "config/baseline/hungarian.yaml"]
    assert dotlist[0] == "settings.seed=3"
    assert set(dotlist[1:]) == {f"paths.model_directory={tmp_path}", "paths.model_name=best_obj",
                                "paths.run_dir=experiments/x/runs/h", "settings.auto_train.max_eval=7"}


def test_model_dir_snapshot_sets_hierarchy_but_not_run_settings(tmp_path):
    snap = {"settings": {"hierarchy_type": "planner_agent", "rl_mode": "train", "mode": 2, "seed": 2},
            "paths": {"model_directory": "models/OLD", "model_name": "", "run_dir": "experiments/old", "init_map": "x.tif"},
            "environment": {"agent": {"planner_agent": {"num_agents": 5}}}}
    with open(tmp_path / "config.yaml", "w") as f:
        yaml.safe_dump(snap, f)
    rest, snapshot = expand_flags(["--watch", "--model-dir", str(tmp_path)])
    overlays, dotlist = split_args(rest)
    cfg = load_config([snapshot] + overlays, dotlist)
    assert cfg.settings.hierarchy_type == "planner_agent"
    assert cfg.settings.rl_mode == "eval" and cfg.settings.mode == 0 and cfg.settings.log_eval is False
    assert cfg.settings.seed == -1  # training seed dropped -> base default
    assert cfg.paths.model_directory == str(tmp_path) and cfg.paths.run_dir == "" and cfg.paths.init_map == "x.tif"
    assert cfg.environment.agent.planner_agent.num_agents == 5


def test_eval_overlay_wins_over_exp_and_run_dir_loads():
    cfg = load_config(["config/agent/planner.yaml", "config/exp/planner_hard.yaml", "config/mode/eval.yaml",
                       "config/baseline/greedy.yaml"], ["paths.run_dir=/tmp/arm"])
    assert cfg.settings.rl_mode == "eval" and cfg.settings.mode == 2
    assert cfg.environment.agent.planner_agent.heuristic_goals is True
    assert cfg.paths.run_dir == "/tmp/arm"
