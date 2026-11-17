"""Viewer, scenario editor and policy debugger.

    uv run python -m roshan.ui                                # demo scenario, greedy planner
    uv run python -m roshan.ui --scenario scenarios/x.npz     # a saved scenario
    uv run python -m roshan.ui --planner rollout              # off, greedy or rollout
    uv run python -m roshan.ui --planner runs/planner/planner.pt   # a trained planner
    uv run python -m roshan.ui --fly runs/navigate/fly.pt     # watch the trained fly policy
    uv run python -m roshan.ui --fly none --drones 2          # the fly task, steered by hand

A scenario with drones runs the allocate task: the planner hands out targets, drones fly, drop
water and refill. Click a drone ("select a drone") to open the inspector with its observation
and reward terms. The File menu opens scenarios (.npz) and land-cover GeoTIFFs, saves, and can make a scenario the
one shown at every start; the Policy menu loads planner and fly checkpoints while it runs.
"""

import argparse
import json
import time
from pathlib import Path

import torch

from roshan import _core, geo
from roshan.scenario import Scenario


def run(scenario=None, policy=None, planner="greedy", auto_close_s=None, save_dir="scenarios", select=None,
        fire_model="rothermel"):
    """Open the viewer on a scenario and block until it closes. Call from the main thread.

    `scenario` is a Scenario or None for the demo. `planner` is "off", "greedy", "rollout" or
    "policy". With `policy(obs) -> one target slot per drone` (obs is a dict of arrays: drones,
    targets, target_mask, action_mask, globals, decide), the policy is asked whenever drones
    need a decision. It runs on the simulation thread. Returns (frames, ticks, policy_calls).
    """

    def save(d, path=None):
        """To `path`, or without one to a new time-stamped file (which never overwrites)."""
        edited = Scenario.from_dict(d)
        if path is None:
            path = Path(save_dir) / f"{edited.meta.get('name', 'scenario')}-{time.strftime('%Y%m%d-%H%M%S')}.npz"
        return f"saved {edited.save(path)}"

    def fetch(lat, lon, width_km, height_km):
        return geo.fetch(lat, lon, width_km, height_km).to_dict()

    start = None if scenario is None else scenario.to_dict()
    return _core.run_ui(start, policy, auto_close_s, save, fetch, planner=planner, select=select,
                        fire_model=fire_model, open=open_scenario, load_policy=load_policy)


def open_scenario(path):
    """A scenario file, or a land-cover GeoTIFF turned into one, as the viewer wants it."""
    path = Path(path)
    if path.suffix.lower() in (".tif", ".tiff"):
        return geo.import_tif(path).to_dict()
    return Scenario.load(path).to_dict()


def load_policy(path):
    """A checkpoint as (is it a pilot, viewer policy, JSON of the task it was trained on)."""
    spec = torch.load(path, weights_only=True)["spec"]
    if "agent" in spec:  # the fly task's observation
        policy, config = fly_policy(path)
        return True, policy, json.dumps(config)
    return False, planner_policy(path), json.dumps(spec["config"])


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
    net = FlyNet(len(saved["spec"]["agent"]), len(saved["spec"]["neighbor"])).to(device)
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
        open=open_scenario, load_policy=load_policy,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", help="a .npz file written by the viewer or by roshan.geo")
    parser.add_argument("--planner", default="greedy", help="off, greedy, rollout, or a trained planner checkpoint (.pt)")
    parser.add_argument("--fire-model", choices=["rothermel", "firespin"], default="rothermel")
    parser.add_argument("--fly", metavar="CHECKPOINT", help="show the navigate task with this fly policy, or 'none'")
    parser.add_argument("--drones", type=int, help="number of drones in the navigate task")
    args = parser.parse_args()
    learned = args.planner.endswith(".pt")
    if not learned and args.planner not in ("off", "greedy", "rollout"):
        parser.error(f"unknown planner {args.planner!r}")
    if args.fly:
        fly(None if args.fly == "none" else args.fly, args.drones)
    else:
        run(
            Scenario.load(args.scenario) if args.scenario else None,
            policy=planner_policy(args.planner) if learned else None,
            planner="policy" if learned else args.planner,
            fire_model=args.fire_model,
        )
