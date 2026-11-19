"""Viewer, scenario editor and policy debugger.

    uv run python -m roshan.ui                                # demo scenario, greedy planner
    uv run python -m roshan.ui --scenario scenarios/x.npz     # a saved scenario
    uv run python -m roshan.ui --planner rollout              # off, greedy, head, flank or rollout
    uv run python -m roshan.ui --planner runs/planner/planner.pt   # a trained planner
    uv run python -m roshan.ui --fly runs/navigate/fly.pt     # watch the trained fly policy
    uv run python -m roshan.ui --fly none --drones 2          # the fly task, steered by hand
    uv run python -m roshan.ui --experiment runs/eval_demo/evaluation.toml --seed 1000003
                                                              # that episode of that evaluation

Started plainly, the viewer uses its own settings (some spread-rate noise, no time limit worth
the name). With `--experiment` (an experiment file or the record of a run) the scenario and the
world are the experiment's, and `--seed` picks the episode: the fire is then the one in the
evaluation's CSV, which the "fire #" in the status bar confirms (Fire_Fingerprint). A planner
checkpoint given with `--planner` brings the task and simulation it was trained on.

A scenario with drones runs the allocate task: the planner hands out targets, drones fly, drop
water and refill. Click a drone ("select a drone") to open the inspector with its observation
and reward terms. The File menu opens scenarios (.npz) and land-cover GeoTIFFs, saves, and can make a scenario the
one shown at every start; the Policy menu loads planner and fly checkpoints while it runs.
The Settings window loads and saves experiment files (see roshan.experiment).
"""

import argparse
import json
import time
from pathlib import Path

import torch

from roshan import _core, experiment, geo
from roshan.scenario import Scenario


def run(scenario=None, policy=None, planner="greedy", auto_close_s=None, save_dir="scenarios", select=None,
        fire_model="rothermel", preburn_s=0.0, path=None, settings=None, seed=None):
    """Open the viewer on a scenario and block until it closes. Call from the main thread.

    `scenario` is a Scenario or None for the demo. `planner` is "off", "greedy", "head", "flank", "rollout" or
    "policy". With `policy(obs) -> one target slot per drone` (obs is a dict of arrays: drones,
    targets, target_mask, action_mask, globals, decide), the policy is asked whenever drones
    need a decision. It runs on the simulation thread. With `preburn_s` the fires burn that
    long before the drones start. `path` is the file the scenario came from, which Save writes
    back to. `settings` is the world of an experiment, a dict with "sim", "allocate" and
    "navigate" (any may be missing): a section given replaces the viewer's own defaults, and
    `fire_model` and `preburn_s` then do not apply to it. `seed` is the episode to start on.
    Returns (frames, ticks, policy_calls).
    """

    def save(d, path=None):
        """To `path`, or without one to a new time-stamped file (which never overwrites).
        Returns the file written. A map that experiment files run on is not overwritten: every
        result made on it would silently stop matching it."""
        edited = Scenario.from_dict(d)
        if path is None:
            path = Path(save_dir) / f"{edited.meta.get('name', 'scenario')}-{time.strftime('%Y%m%d-%H%M%S')}.npz"
        elif users := experiment.uses(path):
            raise ValueError(f"{Path(path).name} is the map of {', '.join(users)}; use Save as with another name")
        return str(edited.save(path))

    def fetch(lat, lon, width_km, height_km):
        return geo.fetch(lat, lon, width_km, height_km).to_dict()

    start = None if scenario is None else scenario.to_dict()
    return _core.run_ui(start, policy, auto_close_s, save, fetch, planner=planner, select=select,
                        fire_model=fire_model, open=open_scenario, load_policy=load_policy, preburn_s=preburn_s,
                        load_settings=experiment.world, save_settings=experiment.save_world,
                        path=None if path is None else str(path),
                        settings=None if settings is None else json.dumps(settings), seed=seed)


def open_scenario(path):
    """A scenario file, or a land-cover GeoTIFF turned into one, as the viewer wants it."""
    path = Path(path)
    if path.suffix.lower() in (".tif", ".tiff"):
        return geo.import_tif(path).to_dict()
    return Scenario.load(path).to_dict()


def load_policy(path):
    """A checkpoint as (is it a pilot, viewer policy, JSON of the task it was trained on, JSON
    of the simulation settings it was trained on)."""
    spec = torch.load(path, weights_only=True)["spec"]
    if "agent" in spec:  # the fly task's observation
        policy, config = fly_policy(path)
        return True, policy, json.dumps(config), json.dumps(spec["sim"])
    return False, planner_policy(path), json.dumps(spec["config"]), json.dumps(spec["sim"])


def trained_world(checkpoint):
    """The task and simulation a planner checkpoint was trained on, as viewer `settings`. A
    checkpoint from an older simulator keeps only its task."""
    from roshan import NavigateEnv

    spec = torch.load(checkpoint, weights_only=True)["spec"]
    try:
        NavigateEnv(1, sim=spec["sim"])  # the cheapest thing that checks simulation settings
    except ValueError as why:
        print(f"{checkpoint}: its simulation settings no longer exist here ({why}); the viewer's are used")
        return {"allocate": spec["config"]}
    return {"allocate": spec["config"], "sim": spec["sim"]}


def planner_policy(checkpoint, device=None):
    """The trained planner network of a checkpoint as a viewer policy (its best guess)."""
    from roshan.rl.train_planner import load

    net, device = load(checkpoint, device)
    names = ("drones", "targets", "target_mask", "action_mask", "globals", "decide")

    @torch.no_grad()
    def policy(obs):
        actions, _, _, _ = net(*(torch.from_numpy(obs[k]).to(device)[None] for k in names), sample=False)
        keep = obs["targets"].shape[0]  # the slot after the last target means "keep"
        return [-1.0 if a == keep else float(a) for a in actions[0].tolist()]

    return policy


def fly_policy(checkpoint, device=None):
    """The trained fly network of a checkpoint as a viewer policy (deterministic actions)."""
    from roshan.rl.nets import FlyNet

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(checkpoint, weights_only=True)
    net = FlyNet(len(saved["spec"]["agent"]), len(saved["spec"]["neighbor"]), saved.get("hidden", 128)).to(device)
    net.load_state_dict(saved["model"])
    net.eval()

    @torch.no_grad()
    def policy(obs):
        agent, neighbors, mask = (torch.from_numpy(obs[k]).to(device) for k in ("agent", "neighbors", "mask"))
        mean, _, _ = net(agent, neighbors, mask)
        return torch.tanh(mean).flatten().tolist()

    return policy, saved["spec"]["config"]


def fly(checkpoint=None, drones=None, nx=35, ny=36, cell_m=10.0, auto_close_s=None, select=None):
    """Open the viewer on the navigate task. With a checkpoint its policy flies the drones;
    without, they hover unless steered by hand. `drones` overrides the trained team size."""
    policy, config = (None, {}) if checkpoint is None else fly_policy(checkpoint)
    if drones is not None:
        config = {**config, "drones": drones}
    return _core.run_ui(
        None, policy, auto_close_s, navigate=json.dumps(config), nx=nx, ny=ny, cell_m=cell_m, select=select,
        open=open_scenario, load_policy=load_policy, load_settings=experiment.world, save_settings=experiment.save_world,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", help="a .npz file written by the viewer or by roshan.geo")
    parser.add_argument("--planner", default="greedy", help="off, greedy, head, flank, rollout, or a trained planner checkpoint (.pt)")
    parser.add_argument("--fire-model", choices=["rothermel", "firespin"], default="rothermel")
    parser.add_argument("--preburn", type=float, default=0.0, metavar="SECONDS",
                        help="let the fires burn this long before the drones start")
    parser.add_argument("--fly", metavar="CHECKPOINT", help="show the navigate task with this fly policy, or 'none'")
    parser.add_argument("--drones", type=int, help="number of drones in the navigate task")
    parser.add_argument("--experiment", metavar="FILE", help="show the scenario and world of this experiment file, or of "
                        "a run's experiment.toml / evaluation.toml, exactly as training and evaluation run them")
    parser.add_argument("--seed", type=int, help="the episode to show, for example an Episode_Seed of an evaluation CSV")
    args = parser.parse_args()
    if args.experiment and (args.fly or args.fire_model != "rothermel" or args.preburn):
        parser.error("--experiment brings its own settings: not together with --fly, --fire-model or --preburn")
    learned = args.planner.endswith(".pt")
    if not learned and args.planner not in ("off", "greedy", "head", "flank", "rollout"):
        parser.error(f"unknown planner {args.planner!r}")
    settings, scenario = None, args.scenario
    if args.experiment:
        settings = json.loads(experiment.world(args.experiment))
        scenario = scenario or experiment.scenario_of(args.experiment)
    elif learned:
        settings = trained_world(args.planner)
    if args.fly:
        fly(None if args.fly == "none" else args.fly, args.drones)
    else:
        run(
            Scenario.load(scenario) if scenario else None,
            path=scenario,
            settings=settings,
            seed=args.seed,
            policy=planner_policy(args.planner) if learned else None,
            planner="policy" if learned else args.planner,
            fire_model=args.fire_model,
            preburn_s=args.preburn,
        )
