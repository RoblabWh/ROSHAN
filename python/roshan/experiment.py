"""Experiment files: one TOML file says what a training or an evaluation run does.

A file may leave out what it keeps at the default, but the files in experiments/ list every
setting, so that what can be set is in plain sight (a test keeps them complete).
`python -m roshan.experiment planner|fly|eval` prints such a file with every default. Sections:

    [run]       what the trainer does: kind ("planner" or "fly"), envs, updates, seed, ...
    [net]       network size
    [sim]       the simulation               (SimConfig)
    [allocate]  the task of the planner      (AllocConfig)
    [navigate]  the task of the fly policy   (NavConfig)
    [reward]    reward weights
    [ppo]       PPO settings
    [eval], [arms.<name>]   an evaluation, see roshan.eval

Every run writes the file back into its output directory with every value filled in and a
`[record]` section (date, commit, command, a digest of how the simulator behaves), and when it
is done a `[result]` section. That file is what was run, and it can be run again. Uncommitted
code is saved next to it as `code.patch`.
The viewer's Settings window loads and saves the same `[sim]`, `[allocate]` and `[navigate]`.

ponytail: no inheritance between files and no sweeps; a variant is a copied file. Add a
`--set section.key=value` override when a sweep needs one.
"""

import functools
import hashlib
import json
import re
import subprocess
import sys
import time
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORLD = ("sim", "allocate", "navigate")


def load(path, sections):
    """The sections of an experiment file as dicts. A section outside `sections` is an error,
    so nothing in a file is ever silently unused. `[record]` is dropped: it describes a past run."""
    with open(path, "rb") as f:
        data = tomllib.load(f)
    data.pop("record", None)
    data.pop("result", None)
    extra = sorted(set(data) - set(sections))
    if extra:
        raise ValueError(f"{path}: {extra} not used here, this takes {sorted(sections)}")
    return data


def training(path, kind, task):
    """The sections of a training file for the trainer `kind` ("planner" or "fly"), whose task
    section is `task`. `[run] kind` is required, so a file only ever runs with its own trainer."""
    with open(path, "rb") as f:
        found = tomllib.load(f).get("run", {}).get("kind")
    if found != kind:
        raise ValueError(f'{path}: [run] kind is {found!r}, this trainer runs kind = "{kind}"')
    return load(path, ["run", "net", "sim", task, "reward", "ppo"])


def merged(defaults, given, where):
    """`defaults` with the values of `given`; a key that has no default is an error."""
    extra = sorted(set(given) - set(defaults))
    if extra:
        raise ValueError(f"[{where}] has no {extra}; it knows {sorted(defaults)}")
    return {**defaults, **given}


NOTES = {
    "run": 'what the trainer does; kind is "planner" or "fly"; "" means none',
    "net": "network size",
    "sim": "the simulation; names carry their units: _s seconds, _m metres, _l litres",
    "sim.spin": "the FireSPIN particle model, used when fire_model is firespin",
    "allocate": "the planner's task: drones are sent to fires",
    "navigate": "the fly task: drones fly to waypoints",
    "reward": "weight of each reward term",
    "ppo": "PPO settings",
    "eval": 'split is "val" or "test"; n = 0 runs all its seeds; pair = ["a", "b"] adds paired statistics of two arms',
    "arms": 'one section per planner to compare: policy is none, greedy, hungarian, rollout or a checkpoint (.pt);\n'
            '# horizon (seconds) and candidates belong to rollout, sample to a checkpoint;\n'
            '# fly is a fly checkpoint, "decentral", or "" for the scripted controller',
}


def defaults(kind):
    """Every setting of a "planner", "fly" or "eval" file, with its default."""
    import dataclasses

    from roshan import AllocateEnv, NavigateEnv
    from roshan.eval import run as ev
    from roshan.rl import train as fly
    from roshan.rl import train_planner as planner
    from roshan.rl.ppo import PPOConfig
    from roshan.scenario import Scenario

    if kind == "fly":
        spec = NavigateEnv(1).spec
        return {"run": fly.RUN, "net": fly.NET, "sim": spec["sim"], "navigate": spec["config"], "reward": fly.REWARD,
                "ppo": dataclasses.asdict(PPOConfig())}
    spec = AllocateEnv(1, Scenario.load(ROOT / planner.RUN["scenario"]), config=planner.TASK).spec
    if kind == "planner":
        return {"run": planner.RUN, "net": planner.NET, "sim": spec["sim"], "allocate": spec["config"],
                "reward": planner.REWARD, "ppo": dataclasses.asdict(planner.PPO)}
    return {"eval": ev.EVAL | {"scenario": planner.RUN["scenario"]}, "sim": spec["sim"], "allocate": spec["config"],
            "arms": {"greedy": ev.ARM | {"policy": "greedy"}}}


def dumps(tables, prefix=""):
    """TOML text for nested dicts of numbers, switches, text and lists. None is left out.
    Sections named in NOTES get their note as a comment."""
    key = lambda k: k if re.fullmatch(r"[A-Za-z0-9_-]+", k) else json.dumps(k)  # noqa: E731
    out = []
    for name, table in tables.items():
        nested = {k: v for k, v in table.items() if isinstance(v, dict)}
        plain = {k: v for k, v in table.items() if k not in nested and v is not None}
        note = f"# {NOTES[prefix + name]}\n" if prefix + name in NOTES else ""
        if not plain and nested:
            out.append(note.rstrip("\n")) if note else None
        if plain or not nested:
            out.append(f"{note}[{prefix}{key(name)}]\n" + "".join(f"{key(k)} = {json.dumps(v)}\n" for k, v in plain.items()))
        if nested:
            out.append(dumps(nested, f"{prefix}{key(name)}."))
    return "\n".join(out)


def digest(path):
    """Short digest of a file, to tell later whether a checkpoint is still the one that was used."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, errors="replace").stdout
    except OSError:
        return ""


def claim(out, force=False):
    """The output directory of a run. One that already holds something is refused unless
    `force`: a second run would overwrite the checkpoints and mix its curves into the first's."""
    out = Path(out)
    if not force and out.is_dir() and any(out.iterdir()):
        raise SystemExit(f"{out} already holds a run: choose another --out, or pass --force to write over it")
    return out


@functools.cache
def simulator_digest():
    """Changes whenever the simulator behaves differently (fire, water, drones or the demo map):
    what the greedy planner does on 8 fixed episodes of the dynamic task. Results and
    checkpoints made under another digest are stale. About a second, once per process."""
    from roshan.eval.run import evaluate
    from roshan.rl.train_planner import RUN, TASK
    from roshan.scenario import Scenario

    rows = evaluate(Scenario.load(ROOT / RUN["scenario"]), "greedy", list(range(3_000_000, 3_000_008)), TASK)
    seen = [(r["Fire_Fingerprint"], r["Burned_Cells"], r["Burned_NoAction"], r["Drops"], r["Water_L"]) for r in rows]
    return hashlib.sha256(json.dumps(seen).encode()).hexdigest()[:12]


def record(out=None, **more):
    """When, from which code and by which command a run was made. With uncommitted changes the
    commit does not say which code ran, so they are saved to `out`/code.patch."""
    changed = bool(_git("status", "--porcelain").strip())
    patch = None
    if changed and out is not None:
        text = _git("diff", "HEAD", "--binary")
        # ponytail: new files are added to the patch only while they are small text files.
        for name in _git("ls-files", "--others", "--exclude-standard").splitlines():
            if (ROOT / name).stat().st_size < 1_000_000:
                text += _git("diff", "--no-index", "--", "/dev/null", name)
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / "code.patch").write_text(text)
        patch = "code.patch"
    return {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "commit": _git("rev-parse", "HEAD").strip(),
        "uncommitted_changes": changed,
        "patch": patch,
        "command": " ".join(sys.argv),
        "simulator": simulator_digest(),
        "finished": False,
        **more,
    }


def finish(path, tables, seconds, result):
    """Rewrites the record of a run that completed: how long it took and what came out."""
    tables["record"] |= {"finished": True, "wall_time_s": round(seconds, 1)}
    return write(path, tables | {"result": result})


def recorded(path):
    """The `[record]` of an experiment file written by a run, or {} without one."""
    try:
        with open(path, "rb") as f:
            return tomllib.load(f).get("record", {})
    except OSError:
        return {}


def write(path, tables):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(tables))
    return path


def scenario_of(path):
    """The scenario file an experiment file runs on, or None (the fly task has none)."""
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return data.get("run", {}).get("scenario") or data.get("eval", {}).get("scenario") or None


def uses(scenario):
    """Names of the files in experiments/ that run on the scenario file `scenario`."""
    same = lambda named: named and (ROOT / named).resolve() == Path(scenario).resolve()  # noqa: E731
    return [path.name for path in sorted(ROOT.glob("experiments/*.toml")) if same(scenario_of(path))]


def world(path):
    """JSON of the `[sim]`, `[allocate]` and `[navigate]` sections of any experiment file, for the viewer."""
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return json.dumps({k: data[k] for k in WORLD if k in data})


def save_world(path, text):
    """Writes the viewer's settings (JSON of its three sections, every value)."""
    path = Path(path).with_suffix(".toml")
    path.write_text("# Settings saved from the viewer. For training, copy [sim] and the task into a file with [run].\n\n"
                    + dumps(json.loads(text)))
    return f"saved {path}"


if __name__ == "__main__":
    kind = sys.argv[1] if len(sys.argv) == 2 else ""
    if kind not in ("planner", "fly", "eval"):
        sys.exit("usage: python -m roshan.experiment planner|fly|eval     prints a file with every setting at its default")
    print(dumps(defaults(kind)), end="")
