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
# 1. train: imitate the rollout planner (writes runs/planner/imitation.pt and planner.pt)
uv run python -m roshan.rl.train_planner --teacher rollout --imitation-steps 4000 --updates 0 --out runs/planner

# 2. watch it assign targets in the viewer
uv run python -m roshan.ui --planner runs/planner/planner.pt

# 3. measure it on the validation seeds, and against a baseline on the same fires
TASK='{"random_ignitions":6,"random_wind":true}'
uv run python -m roshan.eval --scenario scenarios/demo.npz --config "$TASK" --policy runs/planner/planner.pt --out learned.csv
uv run python -m roshan.eval --scenario scenarios/demo.npz --config "$TASK" --policy hungarian --out hungarian.csv
uv run python -m roshan.eval.paired --arm learned learned.csv --arm hungarian hungarian.csv
```

Wherever a planner name is accepted (`--policy` in the evaluation, `--planner` in the viewer),
a checkpoint path works too. Add `--updates N` in step 1 to continue with PPO after the
imitation.

## What you can run

**Viewer and scenario editor** ([details](docs/systems.md#viewer-and-scenario-editor))

| Command | What it does |
|---|---|
| `uv run python -m roshan.ui` | The demo scenario: drones fly, drop water and refill |
| `... --scenario scenarios/haard.npz` | A saved scenario |
| `... --planner off\|greedy\|rollout` | Who assigns targets to the drones |
| `... --planner runs/planner/planner.pt` | A trained planner assigns the targets |
| `... --fire-model rothermel\|firespin` | Which fire model runs |
| `uv run python -m roshan.ui --fly runs/navigate/fly.pt` | Watch a trained fly policy |
| `uv run python -m roshan.ui --fly none --drones 2` | Steer a drone by hand |

In the window you can pause, change speed, ignite, drop water, place fires, drones and
stations, and inspect a drone's observation and reward terms. The menus open and save
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
| `uv run python -m roshan.rl.train --updates 450` | Train the fly policy (minutes), writes `runs/navigate/` |
| `uv run python -m roshan.rl.train_planner --teacher rollout --imitation-steps 4000 --updates 0 --out runs/planner` | Train the planner by imitating the rollout planner, writes `imitation.pt` |
| `uv run python -m roshan.rl.train_planner --updates 300` | Train the planner with PPO, writes `planner.pt` |
| `uv run tensorboard --logdir runs` | Training curves |

**Evaluation** ([details](docs/systems.md#evaluation))

| Command | What it does |
|---|---|
| `uv run python -m roshan.eval --scenario scenarios/demo.npz --policy greedy --out greedy.csv` | Run a planner on the validation seeds: `none`, `greedy`, `hungarian`, `rollout`, or a checkpoint `.pt` |
| `... --config '{"random_ignitions":6,"random_wind":true}'` | The dynamic task all planner results use |
| `... --split test`, `... --n 200` | The test seeds; only the first 200 seeds |
| `... --sim '{"fire_model":"firespin"}'` | Change simulation settings |
| `... --fly runs/navigate/fly.pt` | Let the trained fly policy steer instead of the scripted controller |
| `uv run python -m roshan.eval.paired --arm greedy greedy.csv --arm hungarian hungarian.csv` | Paired statistics on two runs |

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
| [Allocate task](docs/systems.md#the-allocate-task) | A planner assigns fires and refill stations to drones; a scripted controller flies them |
| [Planners](docs/systems.md#planners) | `greedy`, `hungarian`, `rollout` (simulates ahead), and a learned network |
| [Scenarios](docs/systems.md#scenarios) | A map plus fires, drones, stations and weather in one `.npz` file |
| [Configuration](docs/systems.md#configuration) | Every setting with its default |

From Python:

```python
import roshan
from roshan.scenario import Scenario

env = roshan.AllocateEnv(64, Scenario.load("scenarios/demo.npz"), config={"random_ignitions": 6})
env.step(env.plan("greedy"))       # buffers such as env.drones, env.targets are torch tensors
```

## Results

Area saved against doing nothing, dynamic task (4 drones, 6 random fires, random wind,
30 minutes), 500 test seeds:

| Planner | Area saved |
|---|---|
| greedy | 47.4 % |
| hungarian | 48.0 % |
| learned (imitation of rollout) | 51.8 % |
| rollout (needs the simulator at decision time) | 57.7 % |

- The learned planner beats one-step matching and recovers about 40 % of the gap to rollout.
  PPO does not improve on imitation yet; that is the open RL problem.
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
- The planner sees the true fire; sensing and belief maps are not built yet.
- One training seed stands behind every learned result.
- PPO on the planner task makes an imitated policy worse.

The full list: [docs/systems.md#known-limits](docs/systems.md#known-limits).
