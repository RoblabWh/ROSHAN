# ROSHAN2

Wildfire simulation and multi-UAV reinforcement learning. A reimplementation of ROSHAN
(master thesis, "Hierarchical Reinforcement Learning for autonomous UAV-Based Fire
Extinguishing") as a fast base for experiments: a Rust simulation core, PyTorch for learning,
and a viewer for looking at both.

- **How the systems work, in depth:** [docs/systems.md](docs/systems.md)
- **The plan and where the implementation departs from it:** [docs/design.md](docs/design.md)

<!-- images: put screenshots in docs/img/ and link them here -->

## Setup

Needs Rust (stable, via rustup), [uv](https://docs.astral.sh/uv/) and a C compiler. No other
system packages.

```
uv sync          # creates .venv, installs PyTorch etc., builds the Rust extension
```

`uv run ...` rebuilds the extension by itself whenever a Rust file changed.

## Quick start

```
uv run python -m roshan.ui                          # watch drones fight a fire (greedy planner)
uv run python -m roshan.ui --fire-model firespin    # the same with the particle fire model
uv run pytest                                       # Python tests
```

## Train and run a learned planner

The point of the project, start to end:

```
# 1. train: imitate the rollout planner (writes runs/planner/imitation.pt, planner.pt and
#    experiment.toml, the record of every setting used)
uv run python -m roshan.rl.train_planner experiments/planner.toml

# 2. watch it assign targets in the viewer
uv run python -m roshan.ui --planner runs/planner/planner.pt

# 3. measure it on the validation seeds, and against a baseline on the same fires
TASK='{"random_ignitions":2,"random_wind":true,"preburn_s":1800,"preburn_cells":20}'
uv run python -m roshan.eval --scenario scenarios/demo.npz --config "$TASK" --policy runs/planner/planner.pt --out learned.csv
uv run python -m roshan.eval --scenario scenarios/demo.npz --config "$TASK" --policy greedy --out greedy.csv
uv run python -m roshan.eval.paired --arm learned learned.csv --arm greedy greedy.csv

# 3, from one file: add the planner as an arm to experiments/eval_demo.toml, then
uv run python -m roshan.eval experiments/eval_demo.toml      # writes runs/eval_demo/
```

What a run does is written in an experiment file in `experiments/`
([details](docs/systems.md#experiment-files)); every run writes the full settings it used
into its output directory.

Wherever a planner name is accepted (`--policy` in the evaluation, `--planner` in the viewer),
a checkpoint path works too. Add `--updates N` in step 1 to continue with PPO after the
imitation.

## What you can run

**Viewer and scenario editor** ([details](docs/systems.md#viewer-and-scenario-editor))

| Command | What it does |
|---|---|
| `uv run python -m roshan.ui` | The demo scenario: 40 small drones fly out from a ground station, drop water and come home for a fresh battery and tank |
| `... --scenario scenarios/haard.npz` | A saved scenario |
| `... --planner off\|greedy\|head\|flank\|rollout\|triage\|triage_rollout` | Who assigns targets to the drones |
| `... --planner runs/planner/planner.pt` | A trained planner assigns the targets |
| `... --fire-model rothermel\|firespin` | Which fire model runs |
| `... --preburn 600` | Let the fires burn 10 minutes before the drones start |
| `uv run python -m roshan.ui --experiment runs/eval_demo/evaluation.toml --seed 1000001` | An episode exactly as an experiment ran it: its scenario, settings and seed (`fire #` in the status bar equals the CSV's `Fire_Fingerprint`) |
| `uv run python -m roshan.ui --fly runs/navigate/fly.pt` | Watch a trained fly policy |
| `uv run python -m roshan.ui --fly none --drones 2` | The fly task without a policy: pick a pilot in the side panel or steer a drone by hand |

In the window you can pause, change speed, ignite, drop water, place fires and ground
stations (Settings says how many drones each has), and inspect a drone's observation and reward terms. The menus open and save
scenarios (and remember one as the default), fetch real terrain, load planner and fly
checkpoints, switch fire model and task, set every simulation parameter, and choose what the
map shows (relief, contours, fire arrival lines, spread rate). Help lists the shortcuts.

**Scenarios from real terrain** ([details](docs/systems.md#real-terrain))

| Command | What it does |
|---|---|
| `uv run python -m roshan.geo fetch --lat 51.72 --lon 7.19 --km 2 1.5 --out scenarios/haard.npz` | Land cover and elevation of a place in Europe as a scenario |
| `uv run python -m roshan.geo import-tif old.tif --out scenarios/old.npz` | Convert a map of the old ROSHAN |

**Training** ([details](docs/systems.md#training))

| Command | What it does |
|---|---|
| `uv run python -m roshan.rl.train experiments/fly.toml` | Train the fly policy as the file says (minutes), writes `runs/fly/` |
| `uv run python -m roshan.rl.train_planner experiments/planner.toml` | Train the planner as the file says (imitating the rollout planner), writes `runs/planner/` |
| `uv run python -m roshan.experiment planner` | Print an experiment file with every setting at its default (`planner`, `fly` or `eval`) |
| `... experiments/planner.toml --updates 5 --out runs/try` | Flags override the file, for a quick try |
| `... --force` | Write over a run that is already in the output directory (refused otherwise); also for `roshan.eval` with a file |
| `uv run python -m roshan.rl.train --updates 450` | Train the fly policy without a file, writes `runs/navigate/` |
| `uv run python -m roshan.rl.train_planner --teacher rollout --imitation-steps 4000 --updates 0 --out runs/planner` | Train the planner by imitating the rollout planner, writes `imitation.pt` |
| `uv run python -m roshan.rl.train_planner --updates 300` | Train the planner with PPO, writes `planner.pt` |
| `uv run python -m roshan.rl.train --evaluate decentral` | Fly 1000 unseen episodes with an untrained pilot (`straight`, `scripted`, `decentral`) or a `fly.pt` |
| `uv run tensorboard --logdir runs` | Training curves |

**Evaluation** ([details](docs/systems.md#evaluation))

| Command | What it does |
|---|---|
| `uv run python -m roshan.eval experiments/eval_demo.toml` | Run every planner of the file on the same seeds, writes CSVs, `summary.md` and `evaluation.toml` to `runs/eval_demo/` |
| `uv run python -m roshan.eval experiments/eval_explore.toml` | Explorer drones find and follow two fires nobody was told about; `summary.md` adds what the drones saw (found after, map seen, what the belief gets wrong) |
| `... experiments/eval_demo.toml --config '{"drones_per_station":80}' --out runs/team_80` | The same file with a setting changed, for sweeps (`--sim`, `--split` and `--n` work the same way) |
| `uv run python -m roshan.eval --scenario scenarios/demo.npz --policy greedy --out greedy.csv` | Run a planner on the validation seeds: `none`, `greedy`, `head`, `flank`, `rollout`, `triage`, `triage_rollout`, or a checkpoint `.pt` |
| `uv run python -m roshan.eval experiments/eval_haard.toml` | The same benchmark on real terrain (Die Haard), writes `runs/eval_haard/` |
| `uv run python -m roshan.tune experiments/eval_demo.toml` | Tune the numbers of the triage planner on training seeds (about 6 minutes) |
| `... --config '{"random_ignitions":2,"random_wind":true}'` | Change the task: here two random fires and a randomly turned wind |
| `... --config '{..., "preburn_s":120}'` | Start with fires that have already spread for two minutes (`preburn_cells` caps their size) |
| `... --split test`, `... --n 200` | The test seeds; only the first 200 seeds |
| `... --sim '{"fire_model":"firespin"}'` | Change simulation settings |
| `... --fly runs/navigate/fly.pt` | Let the trained fly policy steer instead of the scripted controller |
| `... --fly decentral` | Let the hand-written decentralised pilot steer (it sees what the fly policy sees) |
| `uv run python -m roshan.eval.paired --arm greedy greedy.csv --arm head head.csv` | Paired statistics on two runs |

**Tests and benchmarks**

| Command | What it does |
|---|---|
| `uv run pytest` | Python tests |
| `cargo test --workspace --exclude roshan-py --release` | Rust tests |
| `uv run python spikes/ui_spike.py --auto` | Checks the viewer (opens windows briefly) |
| `uv run python -m roshan.bench` | Simulator throughput, thread sweep |
| `cargo run --release -p roshan-core --example bench` | Fire model timings and shape accuracy |
| `cargo run --release -p roshan-ui --example viewer` | Viewer without Python |

## Main parts

| Part | In one sentence |
|---|---|
| [Rothermel fire model](docs/systems.md#rothermel-fire-model) | The default: Rothermel spread rates for the 40 Scott & Burgan fuels, with wind, slope and moisture, on a grid |
| [FireSPIN fire model](docs/systems.md#firespin-fire-model) | The particle model of the thesis, ported; matches the old code within 1 % |
| [Water](docs/systems.md#water) | Dropped water slows fire and evaporates; a knocked-down cell can re-ignite |
| [Navigate task](docs/systems.md#the-navigate-task) | Drones fly to waypoints without colliding; trains the low-level fly policy |
| [Allocate task](docs/systems.md#the-allocate-task) | A swarm of small drones based at ground stations; a planner says which fire each one flies to, the rest (drop, flight home, battery and tank service) is automatic |
| [Planners](docs/systems.md#planners) | `greedy`, `head`, `flank`, `rollout` (simulates ahead), `triage` (leaves fronts dry that are too intense to be worth the water), `triage_rollout`, and a learned network |
| [Scenarios](docs/systems.md#scenarios) | A map plus fires, ground stations and weather in one `.npz` file |
| [Configuration](docs/systems.md#configuration) | Every setting with its default |

From Python:

```python
import roshan
from roshan.scenario import Scenario

env = roshan.AllocateEnv(64, Scenario.load("scenarios/demo.npz"), config={"random_ignitions": 2})
env.step(env.plan("greedy"))       # buffers such as env.drones, env.targets are torch tensors
```

## Results

A swarm of 40 L drones on the demo map, two random fires reached at 0.2 ha, 500 validation
seeds, 2026-11-19 (greedy planner unless stated):

| Team | Fires put out that are weak (up to 500 kW/m) on arrival | Area saved, all fires |
|---|---|---|
| 5 drones | 2.6 % | 4.9 % |
| 20 drones | 6.1 % | 24.7 % |
| 40 drones | 36.5 % | 43.4 % |
| 40 drones, rollout planner | 46.1 % | 46.8 % |
| 80 drones | 80.4 % | 66.9 % |
| 120 drones | 83.9 % | 75.1 % |

- It takes about 80 small drones to put out most weak fires, the size the literature
  arrives at (Ausonio et al. 2021); fires that have reached heavy shrub are not put out by
  any team, only held at the flanks.
- The planners differ, and what counts is triage: water is scarce, and a rule that leaves
  the most intense fronts dry (`triage`) saves 57 % of the area with 40 drones where the
  nearest-fire rule saves 44 %; with lookahead on top 65 % (2026-11-20, with 256 targets on
  offer; the table above was made with 64). No planner has been trained on this task yet;
  what a learned one has to beat and how to build it is in
  [docs/planner-findings.md](docs/planner-findings.md).
- Fly policy: 99.4 % success on 1000 unseen episodes with 4 drones after about 2 minutes of
  training.
- Speed (24-core laptop): 160 M drone-steps/s in the navigate simulator, about 1.0 M with PPO
  end to end; a 60-minute fire on a 100 x 100 map takes 3.5 ms.

All tables, intervals and conditions: [docs/systems.md#results](docs/systems.md#results).

## Layout

```
crates/roshan-core/   the simulation, pure Rust
crates/roshan-ui/     viewer and scenario editor (egui)
crates/roshan-py/     Python bindings (module roshan._core)
python/roshan/        environments, planners, training (rl/), evaluation (eval/), terrain (geo.py)
tests/                Python tests; Rust tests are in crates/roshan-core/tests/
scenarios/            scenario files
splits/               frozen validation and test seeds
runs/                 training and evaluation output (ignored by git)
docs/                 systems.md (how it works), design.md (plan)
```

## Status and limits

Milestones M0 to M8 of the plan are done. The main things to know before trusting a number:

- Water constants and the land-cover to fuel table come from published values, not from
  measured fires; the learned-planner numbers predate that calibration (see docs/systems.md).
- The planner sees the true fire by default. With `plan_on_belief` it works from what explorers and the drones' sensors have seen; that costs a third to more than half of the area saved. Explorers that track a known fire (`explorers_track`) and a belief that forecasts it (`forecast`) win back little of that: the loss is the time until a fire is found.
- One training seed stands behind every learned result.
- PPO on the planner task makes an imitated policy worse.

The full list: [docs/systems.md#known-limits](docs/systems.md#known-limits).
