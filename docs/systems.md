# ROSHAN2: how the systems work

The reference for what the system implements **today**. For running things, see the
[README](../README.md). For the plan this follows and where the implementation departs from
it, see [design.md](design.md).

1. [Architecture](#architecture)
2. [World, clock and randomness](#world-clock-and-randomness)
3. [Configuration](#configuration)
4. [The fire boundary](#the-fire-boundary)
5. [Rothermel fire model](#rothermel-fire-model)
6. [FireSPIN fire model](#firespin-fire-model)
7. [Water](#water)
8. [Scenarios](#scenarios)
9. [Real terrain](#real-terrain)
10. [The navigate task](#the-navigate-task)
11. [The allocate task](#the-allocate-task)
12. [Planners](#planners)
13. [Training](#training)
14. [Evaluation](#evaluation)
15. [Viewer and scenario editor](#viewer-and-scenario-editor)
16. [Tests](#tests)
17. [Results](#results)
18. [Not implemented yet](#not-implemented-yet)
19. [Known limits](#known-limits)

## Architecture

Three Rust crates and one Python package. Cargo enforces the direction of dependencies: the
core cannot see the UI or Python.

```
crates/roshan-core/   simulation. Pure Rust: no Python, no UI, no file or network access
  fuel.rs             the 40 Scott & Burgan fuel models
  rothermel.rs        Rothermel surface fire spread
  fire.rs             the fire boundary (state every model writes), Rothermel spread, water layer
  spin.rs             FireSPIN, the particle fire model of the thesis
  terrain.rs          grid, fuel, slope and land cover per cell
  world.rs            one environment: terrain, weather, fire, water, drones, clock
  scenario.rs         a map plus what is placed on it
  navigate.rs         the navigate task and its batched environment
  allocate.rs         the allocate (planner) task: tanks, drops, scripted flight, targets,
                      greedy and rollout planners
  rng.rs              counter-based random numbers
  examples/           bench.rs (fire and allocate timings, shape accuracy), spin_curve.rs (FireSPIN curve)
  tests/              behaviour tests: fire.rs, spin.rs, navigate.rs, allocate.rs
crates/roshan-ui/     viewer and scenario editor (egui); depends on roshan-core only
crates/roshan-py/     Python bindings, module roshan._core
python/roshan/
  env.py              NavigateEnv and AllocateEnv: batched environments with torch buffers
  baselines.py        the planners without learning, as functions of an environment
  eval/run.py         run a planner on fixed seeds, one CSV row per episode
  eval/paired.py      paired statistics on two such runs
  scenario.py         scenario files (.npz)
  geo.py              land cover and elevation download, land cover -> fuel table
  ui.py               viewer entry point
  bench.py            simulator throughput
  rl/ppo.py           PPO (advantage estimation, squashed Gaussian policy, update)
  rl/nets.py          FlyNet and PlannerNet
  rl/train.py         training and evaluation of the fly policy
  rl/train_planner.py imitation and PPO for the planner
  experiment.py       experiment files: reading, and the record every run writes
experiments/          experiment files (.toml) for training and evaluation
tests/                Python tests
spikes/               ui_spike.py (viewer check), firespin_reference.cpp (old FireSPIN loop)
scenarios/            scenario files
splits/               frozen validation and test seed lists (500 seeds each)
runs/                 training and evaluation output (ignored by git)
```

**The Python/Rust boundary.** The Python environment classes own every buffer as a torch
tensor on the CPU (pinned when CUDA is available). On each call Rust borrows the matching
NumPy views, writes in place and keeps no pointer afterwards. Arrays of the wrong type, shape
or memory layout are rejected, never copied silently. The environments of a batch are stepped
in parallel by a Rust thread pool (`threads`, 0 = all cores) with the interpreter lock
released.

**Names instead of positions.** Every environment exports `env.spec`: the names of the
observation columns, reward terms and summary fields, plus the full task and simulation
configuration. Rust writes reward *terms*; Python weights them. A checkpoint stores the spec
it was trained with.

## World, clock and randomness

- **World** (`world.rs`): terrain, weather, fire, water, drones and a tick counter. Cloning a
  world shares terrain and weather and copies only the dynamic state (about 60 µs for a
  100 x 100 map). The rollout planner relies on this.
- **Clock**: time is a whole number of ticks of `dt_s` (0.1 s). Drones move every tick; the
  fire advances every `fire_every` ticks (20, so every 2 s). The world's clock counts from
  the ignition. With a pre-burn the allocate task's episode starts later: its time limit,
  the `time` observation and the reported episode time count from that start, while the
  wind schedule, fire arrival times and the viewer's clock stay on the world's clock.
- **One fire step**: look up the wind for that time (and recompute everything that depends on
  it if it changed), let the water decay, then advance the fire model in as many sub-steps as
  it asks for.
- **Weather**: five fuel moistures for the episode and a wind schedule of (start time, speed,
  direction). Wind is uniform over the map. A scenario is refused when a wind entry has a
  negative or non-finite speed, a direction that is not a unit vector (it would silently
  scale the speed), or a start time before the entry in front of it.
- **Coordinates**: x grows with the column, y with the row; row 0 is north. Cell index is
  `y * nx + x`. Wind and slope vectors use the same axes.
- **Random numbers** (`rng.rs`): Philox4x32-10. A draw is a function of (seed, stream, counter,
  entity) and nothing else, so results do not depend on thread count or on what happened
  elsewhere on the map. Streams: spread noise, spawn, goal, weather, ignition, FireSPIN.
  Episode `e` of environment `i` is a function of (base seed, i, e).

## Configuration

Defaults live in Rust only. Python passes just the values that differ, as JSON; an unknown key
is an error. `env.spec["config"]` and `env.spec["sim"]` return the full values in use.

```python
AllocateEnv(64, scenario, config={"random_ignitions": 6}, sim={"fire_model": "firespin"})
```

### Experiment files

One TOML file in `experiments/` says what a training or an evaluation run does
(`python/roshan/experiment.py`). The files there list **every setting** with the value the
run uses, so what can be set is what the file shows; a test fails when a setting is missing
from one of them. `python -m roshan.experiment planner|fly|eval` prints a fresh file with
every default. The start of `experiments/planner.toml`
(`uv run python -m roshan.rl.train_planner experiments/planner.toml`):

```toml
[run]
kind = "planner"
scenario = "scenarios/demo.npz"
envs = 512
updates = 0
seed = 0
teacher = "rollout"
imitation_steps = 4000
imitation_envs = 128
init = ""
critic_updates = 0

[net]
hidden = 128

[sim]
dt_s = 0.1
...
```

A value left out of a file is the default (for `[sim]`, `[allocate]` and `[navigate]` the
simulator's). TOML has no "none": `""` stands for it in `teacher`, `init` and `fly`, `n = 0`
means all seeds, `pair = []` no paired statistics.

| Section | Holds | Read by |
|---|---|---|
| `[run]` | `kind` (`"planner"` or `"fly"`, required), `envs`, `updates`, `seed`; planner also `scenario`, `teacher`, `imitation_steps`, `imitation_envs`, `init`, `critic_updates`; fly also the empty map `nx`, `ny` (cells), `cell_m` | trainers |
| `[net]` | `hidden`, the width of the network's layers | trainers |
| `[sim]` | `SimConfig` values, the FireSPIN block as `[sim.spin]` | trainers, evaluation, viewer |
| `[allocate]` | `AllocConfig` values | planner trainer, evaluation, viewer |
| `[navigate]` | `NavConfig` values | fly trainer, viewer |
| `[reward]` | reward weights by term name | trainers |
| `[ppo]` | fields of `PPOConfig` (`rl/ppo.py`) | trainers |
| `[eval]`, `[arms.<name>]` | an evaluation, see [Evaluation](#evaluation) | evaluation |

- **Nothing is silently unused.** A section the reader does not take, a key it does not know,
  or a `kind` that is another trainer's is an error. The viewer is the exception: it reads the
  three world sections of any file.
- **The record.** Before it starts, every run writes the file back into its output directory
  with every value filled in (`runs/<name>/experiment.toml`, `evaluation.toml` for an
  evaluation), plus a `[record]` section: date, git commit, whether there were uncommitted
  changes, the command line, the scenario's fingerprint, the digest of every checkpoint
  used, and `finished = false`. This file is what ran; it does not change when a default
  changes later. It is itself a valid experiment file (`[record]` and `[result]` are ignored
  when read), so a run can be repeated from it.
- **When the run is done** the record gets `finished = true`, `wall_time_s` and a `[result]`
  section. Planner: `teacher_agreement` of the imitation and the training episodes of the
  last log line (`train_burned_fraction`, ...; sampled policy, not an evaluation). Fly
  policy: its closing evaluation. Evaluation: fire out, burned, area saved and drops per arm.
  A record with `finished = false` is a run that crashed or was stopped.
- **Uncommitted code** is saved next to the record as `code.patch` (`git diff HEAD`, plus new
  files while they are text under 1 MB), so the commit and the patch together are the code
  that ran.
- **`simulator`** in the record is a digest of how the simulator behaves: what the greedy
  planner does on 8 fixed episodes of the dynamic task on `scenarios/demo.npz`
  (`experiment.simulator_digest()`, about a second per process). It changes when the fire,
  the water model, the drones or the demo map change, which is exactly when earlier results
  and checkpoints stop being comparable. The evaluation uses it to warn about stale
  checkpoints (see [Evaluation](#evaluation)).
- **Output directory.** `runs/<name of the file>` unless `--out` says otherwise. A directory
  that already holds something is refused, so a second run cannot overwrite checkpoints or
  mix its TensorBoard curves into the first's; `--force` writes over it anyway.
- **Flags** of the trainers (`--updates`, `--envs`, `--seed`, ...) override the file, for a
  quick try; the record shows the values that were used.
- **No inheritance and no sweeps.** A variant is a copied file.
- **Not in a file.** The scenario is its own `.npz`; the record holds its path and
  fingerprint. Fixed in code: the optimiser (Adam, eps 1e-5, learning rate decaying linearly
  to zero), the imitation schedule (the teacher drives the first half of the steps), the fly
  policy's closing evaluation (1000 episodes), the 50 m view of a fly policy steering in the
  evaluation, the paired statistics (95 % intervals, 2000 bootstrap draws), and the fuel and
  calibration constants of the simulator ([Calibration](#calibration)).

**Simulation** (`sim=`, `SimConfig` in `world.rs`):

| Key | Default | Meaning |
|---|---|---|
| `dt_s` | 0.1 | length of a tick, seconds |
| `fire_every` | 20 | the fire advances every this many ticks |
| `water_crit_l_m2` | 0.4 | lowest water load (L/m²) that stops fire: what a weak front needs |
| `water_per_mw_grass_l_m2` | 0.35 | water (L/m²) per MW/m of fireline intensity that stops a front in grass |
| `water_per_mw_l_m2` | 0.6 | the same in shrub, slash and every fuel not named here |
| `water_per_mw_timber_l_m2` | 2.65 | the same in litter under trees (the timber models TL and TU) |
| `water_limit_kw_m` | 4000 | fireline intensity above which no amount of water stops a front |
| `water_tau_s` | 600 | time constant of water decay (an assumption, see [Calibration](#calibration)) |
| `ember_s` | 60 | how long a blocked burning cell keeps the ability to resume (a model parameter) |
| `sheltered_waf` | 0.2 | share of the 20-ft wind that reaches fuel under trees (timber models TU, TL) |
| `ros_noise` | 0.0 | log-normal noise on the Rothermel spread rate, per cell, fixed per episode |
| `fire_model` | `"rothermel"` | or `"firespin"` |
| `spin` | see [FireSPIN](#firespin-fire-model) | settings of the particle model |

**Allocate task** (`config=` of `AllocateEnv`, `AllocConfig` in `allocate.rs`):

| Key | Default | Meaning |
|---|---|---|
| `max_speed_ms` | 10 | drone speed |
| `tank_l` | 40 | tank size, litres |
| `drop_l` | 40 | water per drop, litres; with the tank's size a drone makes one drop per sortie |
| `drop_efficiency` | 0.7 | share of a drop that reaches the fuel |
| `service_s` | 60 | time on a pad to swap the battery and fill the tank |
| `pads` | 8 | drones a ground station services at once; the others queue |
| `endurance_s` | 900 | longest time in the air between two services |
| `drones_per_station` | 40 | drones based at every ground station of the scenario |
| `arrive_m` | 5 | distance at which a drone has reached a fire target |
| `station_m` | 12 | distance at which a drone docks at a station |
| `safe_m` | 0 | distance the scripted controller keeps between drones in the air; 0 = none, straight lines |
| `sense_m` | 0 | radius of the fire sensor of every drone in the air; 0 = no sensor (see Sensing below) |
| `block_cells` | 2 | side of a target block in cells |
| `max_targets` | 64 | fire targets offered at most |
| `max_interval_s` | 10 | how often docked drones that were told to wait are asked again |
| `time_limit_s` | 1800 | episode length |
| `random_ignitions` | 0 | if above 0: this many random fires instead of the scenario's |
| `random_wind` | false | the scenario's wind schedule is turned by one random angle per episode; its wind changes keep their time, speed and size |
| `wind_scale` | 1 | factor on every wind speed of the scenario's schedule |
| `preburn_s` | 0 | the fires burn unattended this long before the episode starts, seconds |
| `preburn_cells` | 0 | ... or until this many cells have caught fire, whichever comes first (0 = no cap) |

The defaults describe a battery multirotor of the 40 L class and a station that swaps
batteries; where they come from is in [Calibration](#calibration). Tank 300, drop 150,
service 10, a long endurance and 4 drones per station is the heavy-lift team the task
modelled until 2026-11-19.

**Navigate task** (`config=` of `NavigateEnv`, `NavConfig` in `navigate.rs`):

| Key | Default | Meaning |
|---|---|---|
| `drones` | 4 | team size |
| `max_speed_ms` | 10 | drone speed |
| `view_m` | 50 | view range; observations are divided by it |
| `collide_m` | 2 | two drones closer than this collide |
| `arrive_m` | 2.5 | distance at which a waypoint is reached |
| `goals` | 3 | waypoints per drone |
| `station_start` | 0.5 | probability that all drones start around a ground station |
| `time_factor` | 2.5 | time limit = `goals` x map diagonal / speed x this |
| `near_drone` | 0.4 | width of the band around other drones, in view lengths, for `approach_drone` |
| `near_edge` | 0.5 | the same for the map edge and `approach_edge` |

Training options are plain command-line flags and Python constants (see [Training](#training)).

## The fire boundary

Both fire models sit behind `fire::FireState`. Tasks, planners and the viewer use nothing
else, so a model can be swapped without touching them.

What every model writes:

| Layer | Meaning |
|---|---|
| `state` | per cell: unburned, burning or burned |
| `t_ignite` | per cell: the time it caught fire |
| `n_ignited` | number of cells that are burning or burned |

What every model provides: `ignite(cell)`, `step(ctx, water, now, dt, ember_s)`,
`substeps(...)` (how finely a fire step must be divided), `refresh_rates` (the wind changed),
`burning()`, `n_burning()`, `is_out()`, `digest()` (for state hashes), and `particles()` for
the viewer (empty for Rothermel).

A step reads `Ctx`: terrain, the per-fuel spread parameters under the current weather, the
episode seed, the configuration and the wind vector. The model is an `enum` with two
variants, not a trait object; `sim.fire_model` chooses it when the world is built.

## Rothermel fire model

The default. It answers two questions separately: how fast fire spreads in a given fuel
(`rothermel.rs`), and how that turns into cells igniting (`fire.rs`).

### Rate of spread

Rothermel (1972) with the Albini (1976) refinements, written to match the US Forest Service
Behave library step by step:

- All 40 Scott & Burgan (2005) fuel models; dynamic models move live herbaceous load to dead
  fuel as the herbs cure.
- Inputs: fuel model, five fuel moistures (1-h, 10-h, 100-h dead, live herbaceous, live woody),
  wind and slope.
- Wind is given at 10 m, divided by 1.15 to the 20-ft wind and reduced to midflame height with
  the unsheltered wind adjustment factor of the fuel bed. The timber models (TU, TL) describe
  fuel under trees and get the sheltered factor `sheltered_waf` instead (Andrews 2012: 0.1 in
  dense stands, 0.2 in open ones).
- Wind and slope combine as vectors into a fire ellipse: head rate `R`, direction and
  eccentricity `e` from the length-to-breadth ratio (Anderson 1983, capped at 8). The rate in
  a direction at angle θ from the head is `R (1 - e) / (1 - e cos θ)`.
- Also available: reaction intensity, fireline intensity, flame length, residence time.
- No wind-speed limit is applied (off by default in Behave too).
- Fuel that is wetter than its moisture of extinction does not burn.

The ellipse of a fuel on flat ground is computed once per wind change and cached; cells with
slope compute their own.

### Spread on the grid

Every burning cell is a *source* that carries the fire along 16 fixed directions: the 8
neighbours and the 8 knight moves. Each direction is an edge with progress from 0 to 1.

- **Cost.** The 16 edges of a source cross the same 17 cells, so their ellipses are computed
  once per ignition and kept for its edges (on a slope an ellipse costs about 40 times what
  it does on flat ground). Water only slows an edge whose path is wet: a dry path returns at
  once, so a drop somewhere on the map does not slow the rest of the fire. The number of
  burnable cells is counted once per terrain.
- **Edge rate.** One over the travel time between the two cell centres:
  `rate = 1 / (mean slowness x length)`, where slowness is seconds per metre in the edge's
  direction (one over the ellipse rate above) and the mean is taken over the cells the path
  crosses: source and target for a neighbour, source, target and the two cells in between for
  a knight move. If any of those cells cannot burn, the edge does not exist.
- **Diagonals** squeeze through a corner. They are blocked only if both cells beside the
  corner cannot burn. So a one-cell road, straight or diagonal, is not crossed; a one-cell gap
  in it is.
- **Ignition.** A cell ignites when the first edge reaches it, at the interpolated arrival
  time; its own edges start with the progress they would have made since then. If several
  edges arrive in one step, the earliest wins, whatever the order of processing. Only the
  target of a knight move ignites, not the cells it passes between.
- **Burning out.** A cell's fuel is used up at ignition time + residence time + the time the
  head needs to cross the cell. A source stays active while it has an edge left to complete;
  it becomes burned once no edge is left and its fuel is used up.
- **Sub-steps.** A fire step is divided so that no edge advances more than half its length
  per sub-step. The bound is the fastest head rate any fuel on the map can reach (wind blowing
  straight up the steepest slope), recomputed when the wind changes.
- **Wind changes** (wind schedule): all edge rates are recomputed, progress is kept.
- **Noise** (`ros_noise`): an optional mean-one log-normal factor on the head rate per cell,
  fixed for the episode.

Why 16 directions: the simulated fire is the polygon spanned by the edge directions inside the
true ellipse, so too few directions make a wind-driven fire too narrow. See
[Known limits](#known-limits) for the measured accuracy.

Nothing depends on the order in which sources are stored, so a difference in one place on the
map never changes the fire somewhere else.

## FireSPIN fire model

`spin.rs` is a port of the particle model of the thesis (old ROSHAN, `src/firespin`). Choose
it with `sim={"fire_model": "firespin"}` or `--fire-model firespin` in the viewer.

### How it works

A burning cell emits two kinds of heat-carrying particles. Each step, an unburned cell with
at least one particle in it is heated for the length of the step. It ignites once its heating
time reaches the ignition delay of its class, and then burns for the burning duration of its
class.

**Convection particles** are carried and scattered by the wind. With wind vector `U`,
fluctuation `u' = wind_a x |U|`, turbulent length `Lt` and Kolmogorov constant `C0`, the
velocity follows a Langevin equation:

```
v   += -(2 + 3 C0) / 4 x u' / Lt x (v - U) x dt  +  sqrt(C0 x u'^3 / Lt x dt) x N(0, 1)
pos += conv_fl x v x dt
heat -= heat / conv_tau_mem_s x dt
```

A particle starts at rest at a random point of its cell with heat `conv_y_st` and vanishes
when its heat falls below `conv_y_lim`, which with the defaults is after about 16 s. In calm
air `u'` is zero, so convection particles do not move at all.

**Radiation particles** sit on a slowly growing ring around the point they were emitted from:

```
angle  = random each step
radius += Sf_mean x dt + Sf_std x sqrt(dt) x N(0, 1)
heat  -= heat / tau x dt          tau = Lr / |Sf_mean + Sf_std x N(0, 1)|, drawn once
```

`Lr` is drawn uniformly from the class's range; very slow particles get the old code's cap of
`tau` = 2703 s. A radiation particle vanishes below `rad_y_lim`. A particle of either kind
that leaves the map is lost.

**Emission.** With the interval `(burning duration - 1) / particle count of the class`, cut to
whole seconds (at least 1), a burning cell emits one particle of a kind whenever the whole
seconds it has burned reach a multiple of that interval.

### Classes

Burning duration, ignition delay, particle counts and radiation constants per CLC+ land-cover
class are the old ones (`spin::CLASSES`):

| CLC+ class | Burns for | Ignites after | Convection | Radiation |
|---|---|---|---|---|
| generic (no land cover) | 120 s | 100 s | 10 | 4 |
| 1 sealed | 15000 s | 750 s | 1000 | 55 |
| 2 needle-leaved trees | 3600 s | 300 s | 80 | 6 |
| 3 broadleaved deciduous trees | 3600 s | 270 s | 60 | 6 |
| 4 broadleaved evergreen trees | 3600 s | 330 s | 100 | 6 |
| 5 low-growing woody plants | 720 s | 120 s | 40 | 4 |
| 6 permanent herbaceous | 360 s | 100 s | 20 | 2 |
| 7 periodically herbaceous | 360 s | 100 s | 20 | 2 |
| 8 lichens and mosses | 240 s | 100 s | 10 | 3 |
| 9 sparsely vegetated, 10 water, 11 snow and ice | does not burn | | | |

- Radiation: `Lr` 9 to 10 m and front speed 0.1 ± 0.025 m/s for every class, except generic
  (`Lr` 5 to 50 m) and sealed (0.1 ± 0.0075 m/s).
- Per cell, burning duration and ignition delay each stay as they are or change by 10 to 20 %
  up or down, as in the old code. The jitter follows from the seed and the cell alone.
- A scenario without land cover uses the generic class everywhere, with durations from
  `spin.generic_burn_s` and `spin.generic_ignite_s`.
- Which cells can burn at all is decided by the fuel map, as for Rothermel. So sealed ground,
  which burned slowly in the old model, does not burn here although it has a class.

### Settings (`sim.spin`)

| Key | Default | Meaning |
|---|---|---|
| `dt_s` | 0.5 | step of the particle model (old ROSHAN: 0.1) |
| `emit_convective`, `emit_radiation` | true | switch a particle kind off |
| `conv_y_st`, `conv_y_lim` | 1.0, 0.2 | start heat and vanishing limit, convection |
| `conv_tau_mem_s` | 10 | heat-loss time constant, convection |
| `conv_fl` | 0.15 | scale between particle velocity and movement |
| `conv_c0` | 1.98 | Kolmogorov constant |
| `lt_m` | 80 | turbulent length scale |
| `wind_a` | 0.314 | turbulence intensity |
| `rad_y_st`, `rad_y_lim` | 1.0, 0.165 | start heat and vanishing limit, radiation |
| `generic_burn_s`, `generic_ignite_s` | 120, 100 | the generic class |

All are the old configuration's values except `dt_s`.

### What differs from the old code, on purpose

- **Bugs fixed, not ported.** Convection particles were spawned at positions cut to whole
  metres; classes without particles divided by zero; a land-cover value outside the table
  indexed past it; the random numbers depended on earlier episodes.
- **Randomness** is counter-based per cell and per particle. An episode depends on its seed
  alone, and putting out one fire leaves a distant fire bit-identical.
- **Water** is the shared [water layer](#water) instead of a "flooded" cell state: a wet cell
  heats more slowly (heating time counts times the water multiplier), a soaked one not at all;
  particles in a soaked cell are lost; a soaked burning cell emits nothing, goes out after
  `ember_s` and can burn again later. It then burns its full duration again.
- **Step size** is a setting; burning and heating are counted in seconds, not steps.

### Behaviour to know

- It ignores slope and fuel moisture.
- Wind matters little: on the uniform map 10 m/s wind burns 4 % more cells than calm air.
- A single burning cell often dies out (3 of 5 seeds on the uniform map), as in the old
  model. A 3 x 3 block spreads reliably.
- With the land-cover classes fire spreads slowly: grass about 200 cells in 4000 s at 10 m/s
  wind, forest hardly at all.

### Checked against the old code

`spikes/firespin_reference.cpp` is a standalone transcription of the old C++ loop (same
equations, quirks and random generator) for the old uniform 50 x 50 map. The old program
itself could not be run without its window and Python stack. Ignited cells from a 3 x 3
ignition, mean of 20 seeds (standard deviation in brackets):

| 10 m/s wind | 1000 s | 2000 s | 3000 s | Time per run |
|---|---|---|---|---|
| old C++ loop, step 0.1 s | 191.2 (5.1) | 785.5 (12.4) | 1799.5 (22.2) | 1.5 s |
| port, step 0.1 s | 191.1 (4.1) | 782.9 (11.2) | 1791.5 (16.2) | 2.9 s |
| port, step 0.5 s (default) | 193.4 (4.9) | 787.1 (10.5) | 1801.7 (18.7) | 0.57 s |
| port, step 2 s | 196.7 (3.7) | 795.9 (10.0) | 1818.2 (13.7) | 0.30 s |

At 0 and 5 m/s the agreement is the same (within 1 % for steps of 0.1 and 0.5 s). Timings are
from a 16-thread desktop with one thread in use, November 2026. The port is slower per step
than the old loop because every particle draws fresh random numbers instead of reading a
recycled table; the default step of 0.5 s makes it faster overall.

Reproduce with:

```
g++ -O2 spikes/firespin_reference.cpp -o old && ./old <wind_ms> <seed> <end_s> <block>
cargo run --release -p roshan-core --example spin_curve -- <step> <wind_ms> <seed> <end_s> <block>
```

## Water

A layer of water load per cell in litres per square metre (`fire.rs`, `Water`), owned by the
world and read by both fire models.

- The load decays exponentially with time constant `water_tau_s`.
- A cell's multiplier is `1 - load / critical load`, clamped to 0..1.

Under **Rothermel** the critical load depends on how intense the fire is:

- Critical load = `max(water_crit_l_m2, k · I)`. `I` is Byram's fireline intensity in MW/m of
  the front that set the burning cell alight; `k` is `water_per_mw_grass_l_m2` for the grass
  models (GR), `water_per_mw_timber_l_m2` for litter under trees (TL, TU) and
  `water_per_mw_l_m2` for all others.
- `I` = heat per unit area of the cell's fuel × the speed of the front when it arrived. That
  speed is the slowest apparent speed from any of the eight neighbours that burned before, so
  a flank counts as a flank even though grid edges run obliquely to it. A cell lit by hand
  counts as a head. It is fixed when the cell ignites.
- A front lit with more than `water_limit_kw_m` is not affected by water at all.
- The result: flanks and backing fire are stopped by little water, a wind-driven head in
  heavy fuel is only slowed, and a head beyond the limit is not even slowed. See the table in [Calibration](#calibration).
- An edge's rate is multiplied by the multiplier of the wettest cell on its path.
- A burning cell whose every remaining edge is blocked for `ember_s` seconds dies. If it was
  held down by water on itself before its fuel was used up, it returns to unburned, and its
  burning neighbours get their edge towards it back, so they can set it alight again once the
  water is gone. Otherwise it is burned.
- The fire keeps counters for reports (`FireState::stats`): how many burning cells water
  kept from spreading until their flames died, the sum and maximum of their intensity, and
  the highest intensity any cell was lit with. `FireState::front(cell)` gives the water need
  and intensity of a burning cell that can still spread; a cell that only burns out has no
  front, and water on it changes nothing.
- Covered by tests: a wet strip holds a front and releases it when dry; a lasting strip
  contains the fire; a knocked-down cell at the edge of a live fire re-ignites; a load that
  holds the flank does not hold the head.

Under **FireSPIN** there is no intensity, so the critical load is always `water_crit_l_m2`;
see [the differences above](#what-differs-from-the-old-code-on-purpose).

A drone's drop adds `drop_efficiency x drop_l / cell area` to one cell: 40 L at 0.7 on a 10 m
cell is 0.28 L/m². Loads add up, so a cell at the lowest critical load takes two drops, and a
cell needs a little more than its critical load to stay stopped for `ember_s` while the water
evaporates (factor `exp(ember_s / water_tau_s)`, 1.105). The drones aim for that
(see [The allocate task](#the-allocate-task)).

## Scenarios

A scenario (`scenario.rs`, `python/roshan/scenario.py`) is a map and what is placed on it:

| Field | Meaning |
|---|---|
| `fuel` | fuel id per cell: 0 non-burnable, 1..40 = position in `roshan._core.fuel_models()` + 1 |
| `elevation` | metres per cell, or none for a flat map; slopes are derived from it |
| `landcover` | CLC+ class per cell, or none. FireSPIN takes its burning classes from it; Rothermel ignores it; the viewer colours the map by it |
| `ignitions`, `stations` | lists of (x, y) cells. The drones of the allocate task are based at the stations, `drones_per_station` at each |
| `moisture` | five fractions |
| `wind` | list of (start s, speed m/s at 10 m, to_x, to_y) |
| `meta` | name, georeference (`crs`, `origin`), data sources |

Files are `.npz` (arrays plus one JSON string) with a schema version (2; files of version 1
also held a start cell per drone, which loading drops). `Scenario.fingerprint()`
is a digest of the whole content. `roshan._core.check_scenario(dict)` validates a scenario the
way the simulator will: array shapes, fuel ids, placements inside the map, non-empty wind.

Shipped scenarios:

| File | What |
|---|---|
| `scenarios/demo.npz` | 160 x 120 cells (1.6 x 1.2 km), 1 station in the north-west corner: the planner benchmark map |
| `scenarios/demo_static.npz` | the same map with fuel almost too moist to burn |
| `scenarios/haard.npz` | fetched real terrain |
| `scenarios/baggersee.npz` | fetched real terrain |

## Real terrain

`python/roshan/geo.py`. Nothing is stored locally and no account is needed.

- **Land cover**: Copernicus **CLC+ Backbone 2021**, 10 m, requested per area from the EEA
  image server. This is the successor of CORINE Land Cover with eleven classes; it is fetched
  on demand, not imported as a database. Europe only.
- **Elevation**: Copernicus DEM GLO-30 from its public AWS bucket, resampled bilinearly.
- Both are put on one grid in EPSG:3035, snapped to the 10 m pixels of the land-cover raster.
  Up to 15000 x 4100 cells per request.
- **Land cover to fuel**: the `CROSSWALK` table in `geo.py`, from the European fuel
  classification of Aragoneses et al. 2023 (see [Calibration](#calibration)):

| CLC+ class | Fuel model |
|---|---|
| 1 sealed, 9 sparsely vegetated, 10 water, 11 snow and ice | non-burnable |
| 2 needle-leaved trees | TU2 |
| 3 broadleaved deciduous trees | TU3 |
| 4 broadleaved evergreen trees | TU2 |
| 5 low-growing woody plants | SH8 |
| 6 permanent herbaceous | GR6 |
| 7 periodically herbaceous | GR6 |
| 8 lichens and mosses | GR1 (not in the classification) |

- `refuel <file.npz>` rebuilds the fuel of a saved scenario from its stored land cover after
  the table changed.
- `import-tif` converts a GeoTIFF of CLC+ classes (the map format of the old ROSHAN).

## The navigate task

`navigate.rs`, used from Python as `roshan.NavigateEnv`. Every drone flies to a sequence of
waypoints without hitting another drone or leaving the map. One step is one tick. This is the
task the low-level fly policy is trained on.

Three pilots need no training (`NavEnv::pilot`, `NavigateEnv.pilot(kind)` in Python):

| Pilot | What it knows | What it does |
|---|---|---|
| `straight` | its own waypoint | full speed at the waypoint, easing off within 10 m; no avoidance |
| `scripted` | every drone's exact position, and where the drones before it will be next | the sidestepping rule of the allocate task's controller (`scripted_step`), safety distance `collide_m` + 1 m. Centralised: an upper bound, not a fair opponent for a fly policy |
| `decentral` | its own row of the observation, exactly what a fly policy gets | reciprocal velocity obstacles (van den Berg, Lin & Manocha 2008), `navigate::decentral` |

`decentral` is the fair baseline for a fly policy: every drone decides alone and at the same
moment, sees other drones only inside `view_m`, and assumes each neighbour keeps its velocity
and does half of the avoiding. Of 36 candidate velocities it takes the one closest to
"straight at the waypoint" that brings nothing closer than the safety distance within 5 s; a
near miss costs more the sooner it comes. It is a sampled version, not the exact ORCA linear
program. All three are selectable in the viewer's fly task; `python -m roshan.rl.train
--evaluate <pilot or fly.pt>` flies the 1000 evaluation episodes with any of them, and
`python -m roshan.eval --fly decentral` uses it under a planner
(`_core.decentral_pilot` on the pilot observation).

**Episode**

- Start: with probability `station_start` all drones on a small circle around a ground
  station in a map corner, otherwise at random places.
- Each drone gets `goals` random waypoints in sequence; a waypoint is reached within
  `arrive_m`. A drone that has finished keeps its last waypoint.
- Success: every drone finished. Failure: two drones closer than `collide_m`, or a drone
  outside the map. Cut off at the time limit.
- The step after an episode ended resets that environment and ignores its action, so the
  final observation is visible once. With `autoreset = False` finished environments stay
  frozen (evaluation: every seed runs once).

**Action**: per drone, wanted velocity as a fraction of `max_speed_ms`, two numbers; longer
vectors are scaled back to the maximum.

**Observation** (names in `env.spec`), distances divided by `view_m`. A single frame; velocity
is an explicit feature.

- `agent` [8]: `vx`, `vy`; `goal_dx`, `goal_dy` (vector to the waypoint, clipped to the
  view); `goal_cos`, `goal_sin` (angle between heading and waypoint); `speed`; `goal_dist`.
- `neighbors` [drones, 4]: slot 0 is the nearest map edge, the others the other drones; each
  is `dx`, `dy`, `dvx`, `dvy`. `mask` marks slots in view.

**Reward terms** per drone. Rust writes the terms; the trainer weights them.

| Term | Meaning | Weight in `rl/train.py` |
|---|---|---|
| `goal` | reached a waypoint | 1.0 |
| `success` | the episode succeeded (all drones) | 2.0 |
| `collision` | hit another drone | -3.0 |
| `out_of_bounds` | left the map | -3.0 |
| `timeout` | time ran out before this drone finished | -1.0 |
| `progress` | distance gained towards the waypoint, signed, in view lengths | 1.0 |
| `approach_drone` | increase in closeness to other drones inside a band | -0.5 |
| `approach_edge` | increase in closeness to the map edge inside a band | -0.5 |

`progress` must outweigh `approach_drone`: with the opposite balance the policy learns to
hover, which avoids every penalty and never reaches a waypoint.

**Buffers** of `NavigateEnv`, E environments of D drones:

| Buffer | Shape | Content |
|---|---|---|
| `actions` | [E, D, 2] | read by `step` |
| `agent` | [E, D, 8] | own state |
| `neighbors` | [E, D, D, 4] | map edge, then drones |
| `mask` | [E, D, D] | 1 where a neighbour slot is in view |
| `terms` | [E, D, 8] | reward terms |
| `terminated`, `truncated` | [E] | ended for good; cut off by the time limit |
| `info` | [E, 4] | `success`, `reason`, `seconds`, `goals`; valid where an episode ended |

## The allocate task

`allocate.rs`, used from Python as `roshan.AllocateEnv(num_envs, scenario)`. A swarm of
small drones is based at the scenario's ground stations, `drones_per_station` at each. A
planner says which fire each drone flies to; everything else a drone does is automatic. This
is the task planners are compared and trained on.

**A drone's life cycle.** Every drone starts docked at its station, serviced and ready.

1. Ready: the planner is asked. It answers with a fire target, or -1 to wait.
2. Sent: the drone takes off and flies to the centre of the target block.
3. Within `arrive_m`: it releases `drop_l` on one cell of the block (below). With water for
   another drop left it is asked again; otherwise it turns home to the nearest station.
4. Within `station_m` of the station: it docks and takes the pad that is free first. Service
   (battery and tank) takes `service_s`. A station has `pads` pads; drones that find none
   free wait, first come, first served (drones docking in the same tick: lowest index first).
5. Serviced: ready again.

A docked drone is no body: it is not drawn, not flown, and does not count for separation.

**Endurance.** A drone may be in the air for `endurance_s` between two services. It is a
check when a drone is sent: a fire can only be chosen if the flight there and on to the
station nearest the fire fits in what is left. The scripted flight is a straight line, so a
drone that was sent always makes it; the summary's `max_sortie_s` shows it.
`ponytail:` no battery level per tick; an outside pilot can overrun the endurance.

**Targets.** A table of slots: the scenario's ground stations first, then fire targets. The
map is divided into blocks of `block_cells` x `block_cells` cells (20 m with the defaults).
A block is a target while it holds a *front*: a burning cell that can still set a neighbour
alight. Cells that only burn out are not targets, since water on them changes nothing. The
target's place is the block's centre. Per target the table says how many drops it takes to
stop its fronts now (`need`), how many drops are already on their way (`inbound`) and how
intense its strongest front is. A front too intense for water (`water_limit_kw_m`) counts
towards the intensity but needs no drops, and gets none. The `max_targets` blocks with the
highest need are offered.
The table is rebuilt whenever a decision is due. A drone commits to a block, not to a slot.

**Where a drop lands.** On the front cell of the block that is furthest below the load that
stops it (the lowest cell index among equals), decided when the drone arrives. Drones that
arrive in the same tick drop one after the other in index order and see each other's water,
so drops spread over a block's fronts before any cell gets more than it needs.

**Action**: per drone a slot, or -1. Only drones whose `decide` flag is set are listened to.
-1 leaves a docked drone docked and sends a drone in the air home. `action_mask` says what a
deciding drone may choose: a fire with water for a drop and enough endurance, a station only
when in the air. A choice that is not allowed is ignored.

**Scripted flight.** Straight at the target at `max_speed_ms`. With `safe_m = 0` (the
default) nothing keeps drones apart: at swarm size they are taken to fly in separate height
lanes, and the work per tick grows only linearly with the number of drones. With `safe_m`
above zero the drones in the air move in index order; each takes the straight step, a step
turned by 45 or 90 degrees to either side, or stays, whichever first keeps `safe_m` from the
new positions of the drones before it and the current positions of those after it. Staying
always qualifies, so the distance holds by induction; a drone only takes off when nobody in
the air is within `safe_m` of it.

**Sensing.** With `sense_m` above zero every drone in the air carries a fire sensor: once
per fire step (2 s) it sees the state (unburned, burning, burned) of every cell whose centre
is within `sense_m` of it. All drones write into one shared map, the belief
(`crates/roshan-core/src/belief.rs`): per cell the state at the last look and when that was.
A cell nobody looks at keeps its old state, so the belief can hold fire that has gone out and
miss fire that is new. Docked drones see nothing. The sensor is a circle, has no errors, and
what one drone sees all know at once. The belief is filled from the fire and never read by
it, so sensing changes no fire, and **the planners still read the true fire**: the belief is
measured (four values in the summary below), not yet used. The rollout planner's lookaheads
carry no belief.

**Decision events.** One step runs the simulation tick by tick until a drone needs a
decision: it became ready, the block it was flying to lost its last front (checked at every
fire step), or it dropped and can drop again. Docked drones that were told to wait are asked
again every `max_interval_s` of the episode's clock. Drones on their way are not asked again.
The step returns `elapsed_s`, so a trainer can discount by real time.

**Episode.** Starts with the scenario's fires, or `random_ignitions` random ones, and
optionally a randomly turned wind (`random_wind`: the whole schedule turns by one angle). With a **pre-burn** (`preburn_s`,
`preburn_cells`) the fire model then runs on its own, before the drones are placed, until the
time is up, the cell count is reached or the fire is out. The drones meet a front the model
itself produced, with a burned interior, shaped by wind, slope, fuel and barriers; it is
bit-identical to what an idle episode has at that time. A fire that dies during the pre-burn
ends the episode on its first step (under FireSPIN a single burning cell often dies). The
episode's fingerprint is taken after the pre-burn. How long to pre-burn is a choice about
the task, not a calibrated value; see [Pre-burn](#pre-burn) for what it does to it. Ends when the fire is out (`terminated`)
or at `time_limit_s` (`truncated`). An episode is a function of its seed alone, and
`fingerprint` (burning cells and wind at the start) identifies the fire it started with.
`reset(seeds)` runs chosen seeds; `autoreset = False` freezes finished environments.

**Observation** (names in `env.spec`). Positions are fractions of the map size.

| Table | Columns |
|---|---|
| `drones` [D, 6] | `x`, `y`, `water` (tank fraction), `task_x`, `task_y` (where it is going; its own position if it has nowhere to go), `task_kind` (0 none, 1 fire, 2 station, 3 point, 4 docked) |
| `targets` [S, 7] | `x`, `y`, `is_station`, `cells` (front cells in the block / cells per block), `need` (drops that stop the block now), `inbound` (drops on their way), `intensity` (strongest front, MW/m; 0 under FireSPIN) |
| `globals` [5] | `wind_x`, `wind_y` (wind vector / 20 m/s), `time` (fraction of the time limit), `burning` (share of all cells), `burned` (share of burnable cells that are burning or burned) |
| `target_mask` [S] | 1 where the slot holds a target |
| `action_mask` [D, S] | 1 where that drone may choose that slot; only the rows of deciding drones are filled |
| `decide` [D] | 1 where the drone needs a decision now |

Task kind 3 (a point) only occurs when a drone is sent somewhere by hand in the viewer.

**Reward terms** per drone and step:

| Term | Meaning |
|---|---|
| `drop` | drops this drone made during the step |
| `refill` | tank fractions this drone took on when it docked |
| `burned` | cells that caught fire during the step, as a share of the burnable cells (same for every drone) |
| `fire_out` | 1 when the fire went out (same for every drone) |
| `timeout` | 1 when the time limit was reached (same for every drone) |

**Episode summary** (`info`):

| Value | Meaning |
|---|---|
| `fire_out`, `seconds` | whether the fire went out, and the episode's length without the pre-burn |
| `burned_cells`, `burned_fraction` | cells burning or burned at the end, and their share of the burnable cells |
| `water_l`, `drops` | water released and the number of drops |
| `min_separation_m` | smallest distance between two drones in the air; NaN where it was not watched (`safe_m = 0` under the scripted controller) |
| `start_cells` | cells that had caught fire when the episode started |
| `stopped_cells` | burning cells that water kept from spreading until their flames died |
| `stopped_kw_mean`, `stopped_kw_max` | fireline intensity of those cells, kW/m (0 under FireSPIN) |
| `peak_kw`, `start_kw` | highest fireline intensity any cell of the fire was lit with, kW/m: over the whole episode, and when the episode started |
| `first_drop_s` | time of the first drop, NaN if there was none |
| `queue_s` | drone-seconds spent waiting for a free pad |
| `max_sortie_s` | longest time a drone was in the air between two services |
| `detect_s` | time until a sensor first saw a burning cell; NaN if none did or there is no sensor |
| `seen_share` | share of the map's cells that were looked at at least once; NaN without a sensor |
| `belief_miss` | share of the burning cells the belief did not hold as burning, averaged over the fire steps; NaN without a sensor |
| `belief_phantom` | share of the believed-burning cells that did not burn (any more), averaged over the fire steps on which any cell was believed burning; NaN if none ever was |

**Other buffers**: `elapsed_s` [E], `terminated`, `truncated` [E], `seed`, `fingerprint` [E].
The fingerprint is a digest of the cells burning at the start of the episode and of the wind,
later wind changes included; a steady wind gives the value it gave before schedules were
part of it.

**Learned flight.** `AllocateEnv.fly_tick(velocity, actions=None, view_m=50)` advances one
tick with an outside pilot instead of the scripted controller. Afterwards `env.pilot` holds
what a fly policy sees next, in the format of the navigate task with each drone's task point
as its waypoint. Docked drones are not flown and are in nobody's view. Nothing keeps drones
apart, on the map or within their endurance in this mode; the summary reports the smallest
distance and the longest sortie. Learned flight is meant for small teams: the pilot's
observation has one row per pair of drones.

## Planners

A planner maps the allocate observation to one slot per drone that has to decide.

**Rules shared by greedy, head and flank** (`AllocEnv::assign` in the Rust core): the
deciding drones are taken in index order. Each goes to a block that still needs water after
what is already on its way and what the drones before it were just given, one drop per
drone, provided it can reach it. Drones left over answer -1: they wait at their station, or
fly home.

| Planner | How it decides |
|---|---|
| `none` | drones are never sent: the reference for how much a planner saves |
| `greedy` | the nearest block that still needs water |
| `head` | the most intense block that still needs water (nearer first among equals): attack the head |
| `flank` | the least intense block that still needs water: work along the flanks |
| `rollout` | greedy improved by simulating ahead (below) |
| learned | `PlannerNet` (see [Training](#training)); none has been trained on this task yet |

**Rollout planner** (`AllocateEnv.plan("rollout", horizon_s, candidates)`): multiagent
rollout with greedy as the base policy. It starts from the greedy assignment. The drones
that have to decide are then taken one at a time. For each, these alternatives are tried:
its greedy choice, -1, and the `candidates` (4) blocks it can reach that lack the most
water. Each is simulated `horizon_s` (300 s, enough for a flight and for a soaked front to
die) ahead on a copy of the world, with greedy deciding everything later. The alternative
with the fewest cells burned, then the fewest still burning, wins; ties keep greedy's choice.
By construction it does at least as well as greedy on the simulated horizon. It needs the
simulator at decision time, so it is a reference and a teacher, not a deployable policy.

The Hungarian assignment of earlier versions is gone: it pairs one drone with one fire, and
a block now takes several drops.

## Training

### PPO (`rl/ppo.py`)

One file for both tasks.

- Clipped objective, advantages normalised per minibatch, gradient clipping, Adam with
  `eps` 1e-5, linear learning-rate decay to zero.
- **Discount per simulated second.** A step that takes `s` seconds is discounted by
  `gamma_per_s ** s`, so changing the step length does not change the objective. The navigate
  task has a fixed step; the allocate task passes each step's `elapsed_s`.
- **Episode ends.** A true ending (`terminated`) drops the bootstrap; a cut-off episode
  (`truncated`) bootstraps from the value of its final observation. The step that only
  resets an environment is not used as data.
- `update(..., value_only=True)` trains only the value function, which lets a critic catch up
  with a policy that was trained some other way.
- Diagnostics: policy loss, value loss, entropy, clip fraction, approximate KL, explained
  variance.

| Setting | Fly policy | Planner |
|---|---|---|
| learning rate | 3e-4 | 3e-4 |
| `gamma_per_s` | 0.90 | 0.999 |
| GAE lambda | 0.95 | 0.95 |
| clip | 0.2 | 0.2 |
| epochs x minibatches | 4 x 8 | 4 x 4 |
| value coefficient | 0.5 | 0.5 |
| entropy coefficient | 0.003 | 0.01 |
| gradient norm | 0.5 | 0.5 |
| rollout length | 64 ticks | 64 decisions |
| environments | 1024 x 4 drones | 512 |

### Fly policy (`rl/train.py`, `FlyNet`)

- **Network.** Own features and each neighbour slot go through their own two-layer encoder
  (128 units). The neighbour slots are pooled by attention with the drone's own encoding as
  the query; slots out of view are masked. Own encoding, pooled neighbours and a flag "anyone
  in view" feed a two-layer trunk with a policy head (mean of two actions) and a value head.
  The weights work for any number of drones. The mean starts near zero ("hover").
- **Policy.** Gaussian with a learned, state-independent standard deviation, squashed by
  tanh. Log-probability and entropy bonus are those of the squashed action; scoring the
  unsquashed Gaussian lets its standard deviation grow without bound.
- **Training.** Every drone is one sample. Prints and logs to TensorBoard the share of
  episodes per outcome, PPO diagnostics and throughput.
- **Evaluation.** `evaluate(net, device, episodes, seed, config)` runs the deterministic
  policy on 1000 fresh episodes, each exactly once, from a seed far from the training seeds.
- **Settings.** `python -m roshan.rl.train experiments/fly.toml`: `[run]`, `[sim]`,
  `[navigate]`, `[reward]`, `[ppo]` (see [Experiment files](#experiment-files)). Without a
  file the defaults in `rl/train.py` run.
- **Output.** `runs/<name>/fly.pt`, `evaluation.json`, `experiment.toml`, `tensorboard/`.

### Planner (`rl/train_planner.py`, `PlannerNet`)

- **Network.** Every (drone, target) pair is scored by one small network from the drone's
  encoding, the target's features, the vector and distance between them, and two flags:
  *taken* (a drone that is not deciding now is already flying there) and *claimed* (a drone
  earlier in this decision just chose it). A separate score stands for "keep". Drones are
  decided one after the other, which lets the policy spread the team. Slots a drone may not
  choose are masked. The value is one number per environment, from the mean drone encoding, a
  mean over the encoded targets and the globals. The weights do not depend on the number of
  drones or targets. Hidden size 128.
- **Probabilities.** The log-probability of a joint decision is the sum over the drones that
  were asked; drones that were not asked contribute nothing to it or to the entropy.
- **Reward.** `-100 x burned + 1 x fire_out`: one point per percent of the burnable area
  burned, so the reward is the objective itself. Per environment, not per drone.
- **Settings.** `python -m roshan.rl.train_planner experiments/planner.toml`: `[run]`,
  `[sim]`, `[allocate]`, `[reward]`, `[ppo]` (see [Experiment files](#experiment-files)). The
  flags named below are the keys of `[run]` with the same names.
- **Imitation** (`--teacher rollout --imitation-steps N`). Cross-entropy against the teacher's
  choice on every decision, on 128 environments. For the first half of the steps the
  teacher's actions drive the environments; from then on the network's own do, while the
  teacher keeps labelling every state. So the network also sees, and learns to recover from,
  the situations its own mistakes lead to (DAgger). Writes `imitation.pt`.
- **PPO** runs for `--updates` afterwards (0 = imitation only). `--critic-updates N` trains
  only the value function for the first N updates; `--init` starts from saved weights.
- **Task.** With an experiment file: its `[allocate]` on top of the simulator's defaults.
  Without one: `{"random_ignitions": 6, "random_wind": true}` on `scenarios/demo.npz` unless
  `--config` says otherwise.
- **Output.** `runs/<name>/planner.pt`, `imitation.pt`, `experiment.toml`, `tensorboard/`.

**State of things:** imitation works; PPO does not improve on it and fine-tuning makes it
worse (see [Results](#learned-planner)).

### Checkpoints

A `.pt` file is a dict: `model` (weights), `net` (import path of the network class), `spec`
(the environment spec, including both configurations), `ppo` (settings), `reward` (weights),
`hidden` (the network width; 128 where it is missing), `schema_version`, and for the fly policy `evaluation`. `imitation.pt` holds `model`, `spec`
and `schema_version` only. No source files are copied. The readable form of the same
settings, with the rest of the run (scenario, seed, updates, teacher, commit), is the
`experiment.toml` next to the checkpoint.

## Evaluation

**Runner** (`eval/run.py`, `python -m roshan.eval`). Runs a planner on the seeds of
`splits/val.txt` or `splits/test.txt` (500 each, frozen; training never uses them), every
seed exactly once, and writes one CSV row per episode. The same seeds are also run with idle
drones. Results do not depend on batch size or thread count.

**An evaluation file** compares several planners on the same episodes in one run:
`python -m roshan.eval experiments/eval_demo.toml` writes `runs/eval_demo/` with one CSV per
arm, `summary.md` (a row per arm, and the paired statistics of `pair`) and `evaluation.toml`
(the [record](#experiment-files)). The idle run is made once for all arms.

```toml
[eval]
scenario = "scenarios/demo.npz"
split = "val"                    # or "test"; n = 100: only the first 100 seeds
pair = ["greedy", "learned"]     # optional

[allocate]                       # and [sim]: the task, on top of the simulator's defaults
random_ignitions = 2
random_wind = true

[arms.greedy]
policy = "greedy"

[arms.learned]
policy = "runs/planner/planner.pt"
sample = false                   # also: horizon, candidates, fly
```

`candidates` is the number of alternatives the rollout planner simulates per drone (4); the
other arm settings are the flags of the table below. **Stale checkpoints.** For every checkpoint among the arms the evaluation prints, and writes
into `summary.md` and the record (`warnings`), a WARNING line when the run that made it
recorded another `simulator` digest than the current one, when it has no record at all (made
before 2026-11-18), or when the settings stored in the checkpoint differ from the
evaluation's. It still runs: a different time limit is often meant, a different water model
rarely is.

`summary.md` also splits every arm's result by the peak fireline intensity of the fire when
the drones start (up to 500 kW/m, 500 to 2000, above), because small drones can only stop weak
fronts and a mean over all fires hides that.

With a file, `--config`, `--sim`, `--split` and `--n` change it for this run, and the record
holds what ran. A sweep is a shell loop:

```
for n in 10 20 40 80; do uv run python -m roshan.eval experiments/eval_demo.toml \
    --config "{\"drones_per_station\": $n}" --out runs/team_$n; done
```

A single run without a file takes flags
and writes the same record next to its CSV (`--out greedy.csv` gives `greedy.evaluation.toml`):

| Flag | Meaning |
|---|---|
| `--policy` | `none`, `greedy`, `head`, `flank`, `rollout`, or a planner checkpoint (`.pt`) |
| `--split`, `--n` | `val` or `test`; only the first N seeds |
| `--config`, `--sim` | JSON overrides of the allocate and the simulation configuration |
| `--horizon` | lookahead of the rollout planner, seconds |
| `--sample` | sample from a learned planner instead of taking its best guess |
| `--fly CHECKPOINT` | fly the drones with a trained fly policy instead of the scripted controller |
| `--fly decentral` | fly them with the hand-written decentralised pilot, which sees what a fly policy sees |

| CSV column | Meaning |
|---|---|
| `Episode_Index`, `Episode_Seed` | position in the split, seed |
| `Fire_Fingerprint` | identifies the fire the episode started with |
| `Success`, `Failure_Reason` | 1 if the fire went out; otherwise `Timeout` |
| `Time` | seconds from the start of the episode (after any pre-burn) until it ended |
| `Percent_Burned`, `Burned_Cells` | burned share of the burnable area; cells |
| `Burned_NoAction` | cells burned on the same seed with idle drones |
| `Area_Saved` | share saved of what caught fire after the episode started: 1 - (`Burned_Cells` - `Burned_Start`) / (`Burned_NoAction` - `Burned_Start`); 1 when starting cells were put out before they burned down |
| `Water_L`, `Drops` | water used |
| `Min_Separation` | smallest distance between two drones in the air, metres; empty (NaN) where it was not watched |
| `Burned_Start` | cells that had caught fire when the episode started (the ignitions, or the pre-burn) |
| `Stopped_Cells` | burning cells that water kept from spreading until their flames died |
| `Stopped_kW_Mean`, `Stopped_kW_Max` | fireline intensity of those cells, kW/m: what the drones actually stopped |
| `Peak_kW`, `Peak_kW_NoAction`, `Peak_kW_Start` | highest fireline intensity of the fire: with the planner, left alone, and when the drones started |
| `First_Drop_s` | time of the first drop |
| `Queue_s` | drone-seconds spent waiting for a free pad |
| `Max_Sortie_s` | longest time a drone was in the air between two services |
| `Detect_s`, `Seen_Share`, `Belief_Miss`, `Belief_Phantom` | what the drones saw of the fire (`detect_s` ... in the episode summary); empty (NaN) without a sensor |

**A trained planner** is evaluated like any other: `--policy runs/planner/planner.pt` (or
`imitation.pt`). It takes its best guess per decision unless `--sample` is given. From Python,
`roshan.rl.train_planner.load(path)` returns the network and `planner_policy(net, device)`
wraps it as `env -> actions`, the form `roshan.eval.run.evaluate` takes.

The runner refuses to continue if the fire of a seed differs between the policy and the idle
run.

**Paired statistics** (`eval/paired.py`, ported from the old ROSHAN). Per arm: success with
Wilson interval, time and burned area with bootstrap intervals. Between two arms: McNemar on
success, paired bootstrap and Wilcoxon on time and burned area. It refuses to pair runs whose
fires differ. Flags: `--arm NAME CSV` (repeat), `--pair A B`, `--ci`, `--boot`, `--seed`.

## Viewer and scenario editor

`uv run python -m roshan.ui`. A native window (egui) inside the Python process. The window
runs on the Python main thread with the interpreter lock released; the simulation runs on its
own thread behind a lock that the window takes once per frame to draw. Policies are called
from the simulation thread. What it runs depends on how it is started:

| Start | Runs |
|---|---|
| `python -m roshan.ui [--scenario x.npz]` | The scenario. With a ground station in it: the allocate task (drones fly out from the stations, drop water, come home for service). Without: the fire alone |
| `... --experiment FILE [--seed N]` | The scenario and world of an experiment file or of a run's `experiment.toml` / `evaluation.toml`, on episode N |
| `... --planner off\|greedy\|head\|flank\|rollout` | Who hands out targets (default greedy); can be switched in the window |
| `... --planner runs/planner/planner.pt` | A trained planner hands out targets (its best guess, no sampling); works with `planner.pt` and `imitation.pt` |
| `... --fire-model firespin` | The particle fire model; particles are drawn as dots (pale = convection, orange = radiation) |
| `... --preburn 600` | The fires burn that many seconds before the drones start; also `preburn_s` and `preburn_cells` in Settings |
| `python -m roshan.ui --fly runs/navigate/fly.pt` | The navigate task flown by that trained policy |
| `python -m roshan.ui --fly none --drones 2` | The navigate task without a policy: the scripted pilot flies, or steer by hand |
| `roshan.ui.run(scenario, policy=f, planner="policy")` | The allocate task with your own planner `f(obs) -> slot per drone` |

**Whose settings.** Started plainly, the viewer runs with settings of its own: spread-rate
noise 0.25 (so fires do not look drawn with a ruler) and a time limit of one day, where
training and evaluation use 0 and 1800 s. The fire of a seed is then not the fire an
evaluation saw. With `--experiment` every section the file has (`[sim]`, `[allocate]`,
`[navigate]`) is used exactly as training and evaluation use it, the scenario is the file's
unless `--scenario` names another, and `--seed` picks the episode, which Restart then
replays. A planner checkpoint given with `--planner` brings the task and the simulation it
was trained on. The status bar shows `fire #N` in the allocate task: the episode's
fingerprint, equal to `Fire_Fingerprint` in an evaluation's CSV when it is the same episode
(checked 2026-11-18: seed 1000001 of the dynamic task shows the fingerprint, burned share,
drops and water of its CSV row). Igniting, watering or sending by hand changes the episode
but not that number.

The window has a menu bar, a toolbar, a side panel, the map, and a status bar. Everything
below can be done while it runs.

- **Map**: one colour per land-cover class when the scenario has land cover (fetched and
  imported maps), otherwise one per fuel model, shaded by the relief; a legend lists what
  occurs on the map. Burning cells go from white-hot over yellow to red by age and light up
  their neighbours, burned cells glow as embers and cool to ash, dropped water is cyan.
  Drones are quadcopters with their nose towards where they fly, a number, a bar for the
  water on board, and a fading trail; a drone waiting for a decision is orange, the selected
  one yellow. Ground stations are helipads with two numbers beside them: drones ready to be
  sent / drones docked. Docked drones are not drawn. Fire targets on offer are reticles
  numbered with their slot, and the line from a drone to its target is orange (fire), blue (station) or
  white (a point). A ring splashes where water is dropped. Drag to pan, scroll to zoom.
  Hovering a cell shows land cover, fuel model, head spread rate, slope, water and height.
- **View menu**: colour the ground by land cover, elevation, spread rate or flame length
  (the last two for the wind at the start); switch relief shading, elevation contours, fire
  arrival lines every 5 minutes, the cell grid, the wind arrow, trails, target lines and
  numbers; show the whole map; keep the selected drone in view.
- **Toolbar**: pause, single fire step, restart, speed from 0.25x to 600x, and the left-click
  tools: select a drone (on a station: its next ready drone); send the selected drone;
  ignite now; drop water now; toggle a starting fire or a ground station; erase placements.
  The last three edit the scenario and restart it. How many drones a station has is
  `drones_per_station` in Simulation > Settings. Ignite and water reach `brush` cells around the click. Right click
  always drops water.
- **File menu**: open a scenario (`.npz`) or a land-cover GeoTIFF, open a recent file, fetch
  the terrain of a real place, go back to the built-in demo, save (to the file the scenario
  came from, or to a new `scenarios/<name>-<date>-<time>.npz`), save as, and take a
  screenshot (to `docs/img/`). "Open this scenario at every start" makes the current file
  the one shown when the viewer is started without `--scenario`.
- **Simulation menu**: switch between the scenario and the fly task, switch the fire model,
  choose whether Restart moves on to the next seed (the default) or replays the same one,
  jump to the first validation or test seed, and open Settings.
- **Settings window**: every parameter of the simulation (`SimConfig`, with the FireSPIN
  block), the allocate task (`AllocConfig`) and the fly task (`NavConfig`), under its own
  name. Changed values are bold and can be put back one by one. "Apply and restart" takes
  them over; values the simulation rejects change nothing. This is where the dynamic task
  (`random_ignitions`, `random_wind`), the time limit and the spread-rate noise are set.
  "Load ..." shows the `[sim]`, `[allocate]` and `[navigate]` of an
  [experiment file](#experiment-files) (missing values are the defaults, as in training),
  including the `experiment.toml` of a finished run; "Save ..." writes every value of all
  three tabs as such a file. To train on it, copy its `[sim]` and the task section into a
  training file (one with `[run]`).
- **Saving.** The window title names the scenario's file, with a star while it has unsaved
  edits; closing the window with unsaved edits asks first. Save writes to the file the
  scenario came from (also one given with `--scenario`), or to a new time-stamped file when
  it has none, and from then on to that file. A map that a file in `experiments/` runs on is
  never overwritten from the viewer, by Save or by Save as: every result made on it would
  stop matching it. Save it under another name.
- **Policy menu**: load a checkpoint. A planner checkpoint then hands out the targets, a fly
  checkpoint flies the drones; the task and the simulation settings are set up the way the
  checkpoint was trained (a checkpoint from an older simulator, whose settings no longer
  exist, keeps the viewer's simulation settings and the message says so). Both
  can be attached at once, which shows the planner with learned flight (nothing then keeps
  drones apart, as in the evaluation with `--fly`). Either can be detached again.
- **Side panel**: the seed; the planner (off, greedy, rollout with its look-ahead, or the
  attached policy); in the fly task the pilot (off, straight, scripted, decentral, or the
  attached fly policy; scripted is the default without a policy); the weather at the start
  (wind speed and direction, dead-fuel, live-herb and live-woody moisture) and later wind
  changes (after how many minutes, speed, direction).
- **Status bar**: time, burning cells, burned share, the fireline intensity of the most
  intense front burning now (Rothermel), current wind, drones flying / ready /
  in service (allocate task), map size, seed, the
  fire's fingerprint (allocate task), policy calls, and the last message.
- **Inspector**: "select" and click a drone. A panel shows every observation column of
  that drone by name, its neighbour slots (navigate), and every reward term for the last
  tick and summed over the episode. In the navigate task the view range and the collision
  distance are drawn around it.
- **By hand**: in the allocate task, "send" flies the selected drone to the clicked
  cell: to drop water on a burning cell's block, home when the cell holds a station,
  otherwise just there (and home afterwards). A docked drone takes off for it; one that is
  still being serviced cannot be sent. With the planner off nothing else moves. In the navigate task, "steer it myself" flies the
  selected drone with W A S D or the arrow keys while the policy flies the others.
- **Episodes**: when an episode of the allocate task ends the simulation pauses and shows the
  summary; the navigate task starts the next episode by itself and shows how the last one
  ended.
- **Keyboard**: Space pause, `.` step, R restart, F whole map, 1-7 tools, `[` `]` half or
  double speed, Esc close the inspector, Ctrl+O open, Ctrl+S save, Ctrl+Shift+S save as, F12
  screenshot. Help > Keyboard and mouse lists them.
- **Remembered between runs** (in the user's data directory, `roshan2/app.ron`): the default
  scenario, recent files, the view settings, the speed, the brush, the last fetch area, and
  the window's place and size. Runs with `auto_close_s` neither read nor write it.
- **Terrain of a real place** (File menu): latitude, longitude and size in km, or "choose on
  the street map": an OpenStreetMap view with the rectangle to fetch in its centre; drag the
  map under it. "Fetch" downloads in the background and replaces the map.
- **Policies** get a dict of named arrays and return a flat list: a planner one target slot
  per drone (-1 keeps), asked only when drones need a decision; a pilot two velocity
  fractions per drone, every tick. A policy that raises pauses the simulation and its
  traceback is shown. Ctrl+C in the terminal closes the window, also while it is hidden.
- **Without Python**: `cargo run --release -p roshan-ui --example viewer [navigate|firespin]`
  (no files, no fetch, no learned policy).
- **Cost**: the map picture is repainted and uploaded once per fire step, not per frame, and
  not at all while paused. On a 2000 x 2000 map a repaint takes about 20 ms; the unchanging
  part (land cover, relief, contours) is computed once per restart (about 0.2 s there, 1.3 s
  when colouring by spread rate or flame length).

## Tests

`cargo test --workspace --exclude roshan-py --release` and `uv run pytest`.

- **Rothermel kernel** (`rothermel.rs`, `fuel.rs`): Behave's own regression values (rate of
  spread, flame length, fireline intensity, live moisture of extinction, characteristic
  surface-to-volume ratio, length-to-width ratio, a high-wind case) and sanity checks on all
  40 fuel models.
- **Fire on the grid** (`tests/fire.rs`): head rate, round shape without wind, ellipse shape,
  thread-count independence, clone independence, a distant fire unaffected by suppression,
  wet strips, knock-down and re-ignition, barriers, wind shift.
- **FireSPIN** (`tests/spin.rs`): spread and reproducibility, clone independence, step size,
  a distant fire unaffected by suppression, soaked strip and bare ground, land-cover classes,
  configuration.
- **Navigate** (`tests/navigate.rs`): episode logic, reset semantics, time limit,
  thread-count independence, normalised and masked observations, the three untrained pilots.
- **Allocate** (`tests/allocate.rs`): separation with 16 drones over 10,000 drops, episodes
  that depend only on their seed (with and without a pre-burn), water saving area, decision,
  mask and tank rules, the pre-burn (equals the fire left alone, episode clock, cell cap, a
  fire that dies in it).
- **Python** (`tests/`): baselines, the allocate environment, the evaluation runner and paired
  statistics, environment buffers and validation, advantage estimation, the squashed policy,
  a PPO update, a short end-to-end training run, the planner network and imitation, both
  tasks on FireSPIN, scenario files, the terrain pipeline with the two downloads replaced,
  import of old maps.
- **Viewer** (`spikes/ui_spike.py --auto`): policy threading, Ctrl+C, every planner. Opens
  windows briefly.

## Results

The drone-swarm results are from 2026-11-19 and describe the current task. **The sections
"Planner comparison", "Pre-burn", "Learned planner" and "Learned flight under a planner"
further down were made before that**, with 4 drones of 300 L that refilled in 10 s, drops of
150 L without loss and the Hungarian planner; they do not describe the current task and no
planner has been trained on it yet. All planner numbers use the Rothermel model. Evaluation
results do not depend on the machine; timings do.

### Drone swarm

2026-11-19, commit "Planners order the water that will be missing when a drone arrives". `scenarios/demo.npz`,
all 500 validation seeds, 30 minutes, scripted flight, the settings of
`experiments/eval_demo.toml` unless a row says otherwise: two random fires that the drones
reach when 20 cells (0.2 ha) have caught fire, wind 5 m/s from a random direction, 40 drones
of 40 L at one station with 8 pads. "Start" is the highest fireline intensity of the fire
when the drones start: up to 500 kW/m in 230 episodes (fires in grass), above 2000 kW/m in
262 (a fire in the shrub), in between in 8. Left alone, 4.66 % of the map burns.

**Planners** (`uv run python -m roshan.eval experiments/eval_demo.toml`):

| Planner | Fire out | Area saved | Fire out, start up to 500 kW/m | Area saved, up to 500 | Area saved, above 2000 | Fronts stopped, mean kW/m |
|---|---|---|---|---|---|---|
| greedy | 17.0 % | 43.4 % | 36.5 % | 62.7 % | 26.9 % | 483 |
| head | 13.2 % | 22.0 % | 28.7 % | 42.3 % | 4.4 % | 825 |
| flank | 17.2 % | 44.3 % | 35.7 % | 60.6 % | 30.4 % | 275 |
| rollout | 21.6 % | 46.8 % | 46.1 % | 67.4 % | 29.0 % | 508 |

- Rollout against greedy, paired on the 500 fires: burned area -0.14 points (interval
  [-0.17, -0.11], Wilcoxon p < 1e-40); it puts out 23 fires greedy does not and loses none
  (McNemar p = 2e-7).
- Attacking the most intense front first is the worst rule: the water goes where it is
  least likely to be enough. Against a fire in the shrub, working along the flanks saves the
  most of the three simple rules.
- No front above 4000 kW/m is stopped (the model's limit), and the mean stopped front is
  at a few hundred kW/m: what the drones achieve, they achieve on weak fronts.
- Every drone was back within 477 s of taking off (endurance 900 s). Drones wait about
  250 to 285 s each per episode for a pad.

**The limit** (`experiments/eval_limit.toml`: three fires reached at 40 cells, wind 7.5 m/s;
no fire starts below 500 kW/m, 10.3 % of the map burns when left alone):

| Planner | Fire out | Area saved |
|---|---|---|
| greedy | 1.0 % | 15.7 % |
| head | 0.2 % | 1.9 % |
| flank | 1.4 % | 12.3 % |
| rollout | 1.4 % | 17.8 % |

**Team size** (greedy; the pads grow with the team, one per five drones, unless stated):

| Team | Fire out | Area saved | Fire out, start up to 500 kW/m | Area saved, up to 500 | Area saved, above 2000 |
|---|---|---|---|---|---|
| 5 drones | 1.2 % | 4.9 % | 2.6 % | 6.7 % | 3.5 % |
| 10 | 1.4 % | 10.8 % | 3.0 % | 13.6 % | 8.5 % |
| 20 | 2.8 % | 24.7 % | 6.1 % | 35.3 % | 15.9 % |
| 40 | 17.0 % | 43.4 % | 36.5 % | 62.7 % | 26.9 % |
| 80 | 40.4 % | 66.9 % | 80.4 % | 90.7 % | 46.1 % |
| 120 | 45.0 % | 75.1 % | 83.9 % | 93.1 % | 59.5 % |
| 120, rollout | 50.0 % | 79.1 % | 92.6 % | 95.8 % | 64.5 % |
| 80, 8 pads | 37.0 % | 59.9 % | 76.1 % | 88.3 % | 35.3 % |
| 120, 8 pads | 42.2 % | 66.6 % | 83.9 % | 92.2 % | 44.1 % |

A handful of 40 L drones does next to nothing. It takes about 80 to put out four fifths
of the weak fires, the size Ausonio et al. (2021) arrive at for one platform (80 to 120
drones for 70 m of front). Even 120 drones put out almost no fire that has reached the
shrub; they save area there by holding its flanks.

**How early the drones arrive, and the wind** (greedy, 40 drones; wind 2.5, 5 and 7.5 m/s):

| Cells alight at the start | Fire out at 2.5 / 5 / 7.5 m/s | Area saved at 2.5 / 5 / 7.5 m/s |
|---|---|---|
| 5 | 65.6 / 43.2 / 18.0 % | 77.9 / 65.0 / 43.6 % |
| 10 | 58.2 / 36.6 / 13.8 % | 71.4 / 58.7 / 37.6 % |
| 20 | 49.6 / 17.0 / 6.8 % | 65.0 / 43.4 / 25.6 % |
| 40 | 37.4 / 8.4 / 3.6 % | 56.5 / 35.8 / 20.3 % |
| 80 | 19.2 / 4.4 / 3.4 % | 45.8 / 28.4 / 16.6 % |

**What the result rests on** (greedy, 40 drones, one setting changed):

| Setting | Fire out | Area saved | Fire out, start up to 500 kW/m |
|---|---|---|---|
| benchmark | 17.0 % | 43.4 % | 36.5 % |
| 15 L tank | 1.8 % | 17.6 % | 3.9 % |
| 15 L tank, 107 drones, 21 pads (the same litres in the air) | 18.6 % | 43.9 % | 40.0 % |
| 100 L tank | 31.2 % | 65.7 % | 61.3 % |
| service 600 s (charging instead of swapping) | 1.8 % | 10.9 % | 3.9 % |
| service 600 s, a pad for every drone | 2.2 % | 14.4 % | 4.8 % |
| 4 pads | 7.2 % | 30.4 % | 15.7 % |
| 16 pads | 22.8 % | 47.3 % | 47.0 % |
| drop efficiency 0.5 | 5.2 % | 31.3 % | 11.3 % |
| drop efficiency 1.0 | 26.0 % | 52.3 % | 53.9 % |
| `water_tau_s` 300 | 13.4 % | 39.0 % | 28.7 % |
| `water_tau_s` 1200 | 17.8 % | 44.5 % | 38.3 % |
| `ember_s` 30 | 17.0 % | 43.6 % | 36.5 % |
| `ember_s` 120 | 13.0 % | 40.7 % | 27.8 % |
| `water_limit_kw_m` 2000 | 17.2 % | 48.4 % | 36.1 % |
| endurance 450 s; eucalypt curve for litter | unchanged | unchanged | unchanged |

- What counts is litres delivered per minute: 107 drones of 15 L do what 40 of 40 L do, a
  ten-minute charge instead of a one-minute swap ends the approach, and half the pads cost
  half the fires put out.
- The assumed drop efficiency matters most among the constants without a source: at 0.5 a
  cell takes three drops instead of two and a third of the fires put out remain. How long
  water lasts and `ember_s` matter less.
- A lower intensity limit saves more area, not less: no water is spent on fronts that are
  out of reach.
- Endurance and the litter curve do not bind on this map: the longest sortie is under
  8 minutes, and the map's litter (TL3) burns below the lowest critical load.

The sweeps were run through `roshan.eval.evaluate` with the settings above; the same runs
from the command line are
`uv run python -m roshan.eval experiments/eval_demo.toml --config '{"drones_per_station": 80, "pads": 16}' --out runs/team_80`.

### Performance

`cargo run --release -p roshan-core --example bench`, one thread, on an AMD Ryzen 9 9950X3D,
2026-11-18. The "hills" lines are the path training runs on (every cell on a slope, four
fuels, drones, a planner); "before" is the same bench without the ellipse cache, the dry-path
exit and the one-time burnable count. Evaluation CSVs of all four planners (32 seeds, with
and without spread-rate noise and a pre-burn) are byte-identical before and after.

| What | Before | Now |
|---|---|---|
| 30-minute fire, 160 x 120 hills, 6 ignitions | 8.2 ms | 4.0 ms |
| The same with one wet cell on the map | 12.0 ms | 6.1 ms |
| Fire step, flat 200 x 200 grass | 10 ns per burning cell | 9 ns |
| 60-minute fire, flat 100 x 100 | 2.4 ms | 2.2 ms |

The allocate task since 2026-11-19 (same machine, same day): 16 environments on the hills
map, three random fires that burn two minutes before the drones start, one station. The
time is for everything: fire, drones, target table and observation at every decision.

| Planner | 40 drones | 100 drones |
|---|---|---|
| greedy: 30 simulated minutes | 4.9 ms | 8.6 ms |
| rollout (300 s ahead, 4 candidates): 30 simulated minutes | 3.0 s | 19 s |
| rollout: per drone asked | 0.66 ms | 1.24 ms |

With greedy the task costs little more than the fire alone, because nothing is computed per
pair of drones. The rollout planner simulates up to six futures of 300 s for every drone it
is asked about, also for each of the drones that wait at the station every 10 s; that is the
place to save if it has to be faster.

The older figures below are from a 24-core Intel Core Ultra 9 275HX with an RTX 5090 Laptop
GPU, 2026-11-12, before that change; the two machines are not comparable.

| What | Result |
|---|---|
| Navigate, simulator only | 18 M drone-steps/s on 1 thread, 160 M on 24 |
| Navigate, PPO end to end | about 1.0 M drone-steps/s |
| 60-minute fire on a 100 x 100 map, one thread | 3.5 ms |
| Fire step | 14 ns per burning cell |
| Clone of a 100 x 100 world | about 60 µs |
| Terrain fetch, 2 km x 1.5 km | about 2 s |
| Allocate, simulator only, 256 environments, random actions | about 50k decisions/s, 38 simulated hours per second |
| 200 rollout-planner episodes | about 45 s |

### Fly policy

The same 1000 unseen episodes for every pilot (`python -m roshan.rl.train --evaluate`,
2026-11-17). Success means every drone reached all its waypoints; every failure was a
collision.

| Pilot | Decides from | 4 drones | 8 drones |
|---|---|---|---|
| straight | own waypoint only | 55.2 % | 23.0 % |
| fly policy (450 updates, about 2 minutes, one training seed; trained with 4 drones) | own observation | 99.4 % | 86.1 % |
| decentral (hand-written) | own observation | 100 % | 99.9 % |
| scripted (centralised, full information) | everything | 100 % | 100 % |

The hand-written decentralised pilot, which sees exactly what the fly policy sees, is at
least as safe as the trained policy with four drones and clearly safer with eight. Flight
time per waypoint was not compared.

Scripted flight in the allocate task, 16 drones, 10,008 water drops: closest approach 3.00 m
(the safety distance).

### Planner comparison

*Before 2026-11-19: the old task with 4 drones of 300 L (see the top of Results).*

200 validation seeds, 4 drones, 6 random ignitions, random wind, 30 minutes.
`scenarios/demo.npz` is the dynamic task; `scenarios/demo_static.npz` is the same map with
fuel almost too moist to burn, so the fires hardly spread (the situation of the old ROSHAN
benchmark).

| Dynamic task | Burned area | Area saved | Fire out |
|---|---|---|---|
| none | 9.97 % | 0 % | 0 % |
| greedy | 6.55 % | 35.8 % | 0.5 % |
| hungarian | 6.45 % | 36.9 % | 0.5 % |
| rollout | 5.29 % | 48.7 % | 1.5 % |

| Near-static task | Area saved | Fire out | Time to fire out, median |
|---|---|---|---|
| none | 0 % | 0 % | - |
| greedy | 97.0 % | 100 % | 463 s |
| hungarian | 97.5 % | 100 % | 53 s less than greedy (paired median) |
| rollout | 97.6 % | 100 % | 378 s |

Paired against greedy:

- Dynamic, rollout: burned area -1.26 points, interval [-1.45, -1.07], Wilcoxon p = 4e-29;
  fire out in 2 episodes where greedy failed and never the reverse (McNemar p = 0.5).
- Dynamic, hungarian: burned area -0.10 points, interval [-0.21, 0.00], Wilcoxon p = 0.56.
- Near-static: every planner puts every fire out and saves 97 % of the area. Rollout is
  70 s faster than greedy (paired median, interval [-92, -58]); hungarian 53 s ([-64, -24]).

So lookahead buys a fifth less burned area on the dynamic task, where one-step matching
(greedy, hungarian) is tied; on the near-static task nothing but a minute of time is at stake.
Planners save less than in the earlier version of this table (greedy saved 47.8 % then). Two
things changed at once, the demo map and the water model, and their effects were not
measured separately.

### Pre-burn

*Before 2026-11-19: the old task with 4 drones of 300 L (see the top of Results).*

Greedy planner on the dynamic task, first 100 validation seeds, 4 drones, 2026-11-17. "Cells
at the start" is the median number of cells alight or burned when the drones start.

| Pre-burn | Cells at the start | Burned, idle | Burned, greedy | Area saved | Drops per episode |
|---|---|---|---|---|---|
| none | 6 | 10.1 % | 6.7 % | 34.8 % | 70 |
| 120 s | 17 | - | 10.0 % | 13.1 % | 102 |
| until 30 cells | 30 | - | 11.3 % | 5.9 % | 135 |
| until 60 cells | 60 | - | 12.5 % | 2.7 % | 154 |
| 300 s | 73 | 13.1 % | 12.9 % | 2.1 % | 163 |
| 600 s | 261 | 16.3 % | 16.2 % | 0.8 % | 178 |

Four drones that put out one cell per 150 L drop cannot contain a fire that has had a head
start: after two minutes they save a third of what they save from the bare ignitions, after
five minutes almost nothing, although they drop more than twice as much water. A pre-burn
therefore needs a stronger team (more drones, larger or wider drops) to make a task worth
comparing planners on; which is a decision about the benchmark that has not been made.

The pre-burn itself costs about 5 ms per episode for ten simulated minutes on the demo map.
What costs is the larger fire afterwards: the greedy evaluation above takes about three
times as long with a five-minute pre-burn as without.

### Learned planner

*Before 2026-11-19: the old task with 4 drones of 300 L (see the top of Results).*

**Made before the calibration and on the previous demo map; not comparable with the table
above.** All runs use `--config '{"random_ignitions":6,"random_wind":true}'`, 4 drones, one trained
network (one seed). "Learned" is the imitation of the rollout planner, 4000 steps
(`runs/planner_bc_long/imitation.pt`).

Dynamic task, area saved:

| Planner | First 200 validation seeds | 500 test seeds |
|---|---|---|
| greedy | 47.8 % | 47.4 % |
| hungarian | 48.6 % | 48.0 % |
| learned (imitation, 4000 steps) | 52.1 % | 51.8 % |
| rollout | 59.9 % | 57.7 % |
| imitation, 1200 steps | 51.4 % | - |
| imitation + PPO fine-tuning | 43.6 % | - |
| PPO from scratch | 43.1 % | - |

Paired on the test seeds, burned area: learned is 0.36 points below hungarian (interval
[0.28, 0.45], Wilcoxon p = 2e-31) and 0.60 points above rollout ([0.49, 0.71], p = 1e-32).
So the learned planner beats one-step matching and recovers roughly 40 % of the gap to the
lookahead planner it imitates. **PPO does not improve on imitation; fine-tuning degrades it.**
That is the open RL problem.

Near-static task (200 validation seeds): learned saves 99.3 % of the area (greedy 99.2 %) and
is 18 s faster than greedy in the paired median (interval [8, 30]); it puts the fire out in
199 of 200 episodes, greedy in all 200.

### Learned flight under a planner

*Before 2026-11-19: the old task with 4 drones of 300 L (see the top of Results).*

Current simulator (2026-11-17), greedy planner, dynamic task, first 200 validation seeds, the
same fires for every row:

| Flight | Decides from | Area saved | Closest approach | Episodes below 2 m | Episodes below 3 m |
|---|---|---|---|---|---|
| scripted (centralised) | everything | 35.8 % | 3.00 m | 0 | 0 |
| decentral (hand-written) | own observation | 33.6 % | 1.77 m | 1 | 160 |
| fly policy (`runs/navigate/fly.pt`) | own observation | 30.4 % | 2.39 m | 0 | 5 |

Paired, burned area: decentral is 0.23 points above scripted (interval [0.13, 0.34]) and the
fly policy 0.32 points above decentral ([0.18, 0.44]). So of the two pilots that see the same
thing, the hand-written one loses less area and the trained one keeps its distance better:
decentral drops below its own 3 m safety distance in most episodes, where drones head for
the same fire or station, and once below the 2 m collision distance. Neither was built or
trained for flying under a planner. Each of the two flown runs takes about 45 minutes,
because the pilot is called from Python every tick.

The older table below was **made before the calibration and on the previous demo map.**
`runs/navigate/fly.pt` instead of the scripted controller, 500 validation seeds. The fly
policy was trained on the navigate task only, never under a planner:

| Planner | Flight | Area saved | Episodes with drones closer than 2 m | Closest approach |
|---|---|---|---|---|
| greedy | scripted | 46.4 % | 0 % | 3.00 m |
| greedy | learned | 39.0 % | 0.6 % | 0.70 m |
| learned | scripted | 50.5 % | 0 % | 3.00 m |
| learned | learned | 43.3 % | 2.6 % | 0.61 m |

Learned flight costs 0.73 to 0.75 points of burned area (paired, both intervals exclude 0)
and lets drones come close in a few episodes of 30 minutes; the scripted controller stays
the default. Per-tick learned flight is slow to evaluate (Python is called every tick).

The FireSPIN validation table is in [its own section](#checked-against-the-old-code).

## Calibration

Where the numbers that decide fire behaviour and the effect of water come from. Reproduce the
tables with `cargo run --release -p roshan-core --example calibrate`.

### Water

| Constant | Value | Source |
|---|---|---|
| `water_per_mw_l_m2` | 0.6 | Loane & Gould 1986, *Aerial suppression of bushfires* (CSIRO): depth of water that holds a fire, against fireline intensity, eucalypt forest. Read from the curves as reproduced in Plucinski et al. 2007 (Bushfire CRC Technical Report A0701, Fig. 2): about 0.6, 1.1, 2.0, 3.1 mm at 1, 2, 4, 6 MW/m. Read by eye, about ±20 %. |
| `water_per_mw_grass_l_m2` | 0.35 | The same curves for grass: about 0.35, 0.65, 1.2, 1.75 mm. |
| `water_per_mw_timber_l_m2` | 2.65 | Loane & Gould 1986 as quoted by Alexander (2000, NZ Forest Research Bulletin 197): 5.3 mm of water for pine-needle litter at 2 MW/m. Stechishen & Little 1971 (in Hansen 2012) agree in order: 1.5-2.2 mm at 0.5-1 MW/m in red-pine needles. Used for the litter models TL; for the understory models TU it is **assumed**. |
| `water_limit_kw_m` | 4000 | Alexander 2000 (after Alexander & De Groot 1988): direct attack on the head is generally not possible above 4000 kW/m; helicopters and air tankers with retardant are credited with 2000 to 4000. Andrews & Rothermel 1982 put the limit of retardant aircraft at about 1730 kW/m. Plain water from drones will rather lie at the lower figures: sensitivity runs use 2000. Without this limit the linear rule above lets enough drops stop any front: on the benchmark, drones stopped fronts above 4 MW/m in a fifth of the episodes (2026-11-19). |
| `water_crit_l_m2` | 0.4 | Stechishen & Little 1971 (Canadian Forestry Service FF-X-29), as tabulated in Hansen 2012 (Int. J. Wildland Fire 21: 525-536): 0.2-0.6 mm put out fires in pine needles below about 360 kW/m. |
| `sheltered_waf` | 0.2 | Andrews 2012 (USDA RMRS-GTR-266): fully sheltered fuel 0.1 in dense stands, 0.2 in open ones. |
| `water_tau_s` | 600 | **Assumed.** No measured holding time of plain water was found. Water-enhancing gels are reported to last 30-45 minutes at most, and plain water less; with 600 s a drop at twice the critical load holds for 7 minutes. |
| `ember_s` | 60 | **Assumed.** A parameter of this model without a measured counterpart. |
| drop footprint | one cell | **Assumed.** No measured ground pattern of a drone drop of this size was found. A drop wets the whole 10 m cell evenly. |
| `drop_efficiency` | 0.7 | **Assumed for drones.** Hansen 2012 and Penney et al. 2019 (Fire 2: 3) use 0.7 for water from ground hoses. For drops from aircraft, losses of 36 to 88 % are reported (Restás 2023, Fire 6: 351); a drone drops from far lower. Sensitivity runs use 0.5 and 1. |

Checks that were not used as inputs:

- US Forest Service retardant coverage levels (George & Johnson 1990, GTR INT-268) run from
  0.41 L/m² for grass to 2.46 L/m² for brush; long-term retardant is taken to be two to three
  times as effective as water.
- At 4 MW/m, the limit beyond which water does nothing here (Alexander 2000, NZ Forest
  Research Bulletin 197), the model asks for 2.4 L/m² in shrub and 10.6 L/m² in litter under
  trees.
- Until 2026-11-19 the eucalypt curve was used for litter under trees too. Pine-needle litter
  takes about five times as much (see `water_per_mw_timber_l_m2`).
- Ausonio, Bagnerini & Ghio 2021 (Drones 5: 17) size a drone swarm with a critical flow from
  Hansen's fire-point theory: about 4.3 L/min per metre of front for scrub at 10 km/h wind
  (their Fig. 3a). Here a 10 m front in grass at 5 m/s wind is stopped by 44 L on its cell
  (63 L carried) while the head takes 83 s to cross it: 3.2 L/min per metre delivered,
  4.5 carried. The dose on a cell and their flow agree at low intensity.

What the simulator does with it: head fire on flat ground, dead fuel 6 %, live 60/90 %. Per
wind speed at 10 m: rate of spread (m/min), fireline intensity (kW/m), water that stops it
(L/m²).

| Fuel | 2 m/s | 5 m/s | 10 m/s |
|---|---|---|---|
| GR2 | 2.2 / 103 / 0.40 | 7.2 / 330 / 0.40 | 18.9 / 867 / 0.40 |
| GR6 | 5.0 / 1107 / 0.40 | 16.4 / 3632 / 1.27 | 45.1 / 9971 / 3.49 |
| GS2 | 1.7 / 144 / 0.40 | 5.3 / 461 / 0.40 | 13.9 / 1210 / 0.73 |
| SH5 | 7.4 / 2215 / 1.33 | 20.6 / 6154 / 3.69 | 46.1 / 13750 / 8.25 |
| SH8 | 2.4 / 996 / 0.60 | 6.7 / 2755 / 1.65 | 15.3 / 6328 / 3.80 |
| TU2 | 0.5 / 41 / 0.40 | 1.3 / 97 / 0.40 | 3.0 / 227 / 0.60 |
| TU3 | 1.1 / 182 / 0.48 | 2.8 / 449 / 1.19 | 6.4 / 1037 / 2.75 |
| TL3 | 0.1 / 4 / 0.40 | 0.2 / 8 / 0.40 | 0.4 / 16 / 0.40 |

One 40 L drop at efficiency 0.7 leaves 0.28 L/m² on a 10 m cell. To stay stopped for
`ember_s` a cell needs its critical load times 1.105: two drops at 0.40 L/m², 6 at 1.33,
15 at 3.69. `--example calibrate` prints the drops per cell for every fuel.

### Drones and stations

The defaults of the allocate task describe battery multirotors that can be bought, and the
swarm of Ausonio et al. 2021 (80 to 120 drones of 20 to 30 L per platform, a 6-minute cycle,
battery and tank swapped at a mobile platform). Checked on 2026-11-19; product figures are
the manufacturers'.

| Constant | Value | Source |
|---|---|---|
| `tank_l`, `drop_l` | 40 | DJI FlyCart 30 carries 30 kg (40 kg on one battery), DJI Agras T50 a 40 L tank, the drones fire services ordered in 2026 (Seneca, FireSparrow) 38 to 76 L. Ausonio et al. assume 5 to 50 L and one drop per sortie. 300 L, the value until 2026-11-19, is what turbine prototypes and unmanned helicopters lift. |
| `max_speed_ms` | 10 | Ausonio et al.: 35 km/h loaded, 50 km/h empty. One speed is used. |
| `endurance_s` | 900 | DJI FlyCart 30: 18 minutes of hover with 30 kg, 29 minutes empty; the Agras T40 hovers 7 minutes fully loaded. 15 minutes for the whole sortie, loaded out and empty back, leaves a reserve. |
| `service_s` | 60 | Ausonio et al.: the automatic swap of battery and tank takes "a few seconds to a maximum of one minute". Charging the battery in the drone instead takes 9 to 12 minutes (DJI Agras); 600 is the variant for that. |
| `pads` | 8 | **Assumed.** Eight pads at 60 s serve eight drones a minute: Ausonio's platform, 120 drones on a 6-minute cycle, needs twenty. |
| `drones_per_station` | 40 | **A choice**, between the five drones one operator supervises (Seneca) and Ausonio's 120. Results sweep it. |
| `safe_m` | 0 | **Assumed** height lanes: drones on different sorties do not meet. |
| `sense_m` | 0 | Off by default. **Assumed** where used: 20 m, a downward camera on a drone that flies low to drop. |

Not modelled: endurance falling with the load carried (hover power grows with mass to the
power 1.5, which fits the three products above within 10 %), wind acting on the drones, and
what flying 40 drones beyond line of sight needs from a regulator. In the EU dropping
anything from a drone is outside the open category, and no authorisation of a firefighting
swarm was found.

### Land cover

The table follows Aragoneses et al. 2023 (*Classification and mapping of European fuels using
a hierarchical, multipurpose fuel classification system*, Earth Syst. Sci. Data 15:
1287-1315), Table 5, column for the humid regime. CLC+ carries no canopy cover, so forest is
taken as closed, and no vegetation height, so grass is taken as low. Two classes were checked
against observed fires:

**Scots pine** (Tanskanen et al. 2007, Int. J. Wildland Fire 16: 414-425; experimental fires
in Finnish stands). Rate of spread in m/min against the wind inside the stand:

| Dead fuel 9 % | 0 m/s | 0.2 | 0.4 | 0.6 | RMSE |
|---|---|---|---|---|---|
| observed | 0.66 | 1.05 | 1.65 | 2.61 | |
| TL3 | 0.05 | 0.07 | 0.10 | 0.13 | 1.57 |
| TL8 (the old table) | 0.20 | 0.25 | 0.33 | 0.42 | 1.36 |
| TU1 | 0.02 | 0.03 | 0.04 | 0.05 | 1.63 |
| **TU2** | 0.21 | 0.30 | 0.46 | 0.66 | 1.23 |
| TU5 | 0.23 | 0.35 | 0.51 | 0.68 | 1.20 |

At 14 % dead fuel moisture the picture is the same (TU2 0.37, TU5 0.34, TL8 0.50). TU2 and
TU5 are equally close; TU2 is kept because the published table names it. Every standard model
is a factor of two to four too slow here.

**Heath** (Davies et al. 2009, J. Appl. Ecol. 46: 1054-1063; 27 fires in Calluna). Heather
0.3 m high, dead fuel 20 %, live 75 %, wind at 10 m:

| | 2 m/s | 4 | 6 | 8 | RMSE |
|---|---|---|---|---|---|
| observed | 2.22 | 3.64 | 5.07 | 6.49 | |
| SH2 (the old table) | 0 | 0 | 0 | 0 | 4.64 |
| SH3 (low shrub in the published table) | 0.35 | 0.74 | 1.18 | 1.66 | 3.55 |
| **SH8** (medium shrub) | 2.07 | 4.43 | 7.12 | 10.05 | 2.09 |
| SH9 | 4.42 | 9.52 | 15.34 | 21.68 | 9.69 |

SH8 is used. SH2 and SH5 do not burn at all at this moisture.

Not used: the fuel loads Heisig et al. (2022, Fire 5: 29) measured in the Haard forest itself.
Their spread rates are simulated, not observed, and adding custom fuel models is a separate
step.

## Not implemented yet

From the plan in [design.md](design.md):

- Fitting the water constants and the land-cover table to measured fires; they are set from
  published values (see [Calibration](#calibration)).
- Training a planner on the drone-swarm task of 2026-11-19; the network accepts its
  observation, and older checkpoints are refused.
- The scripted explorer, and planners that read the belief instead of the true fire.
- Sweeps over experiment files and inheritance between them; a variant is a copied file.

## Known limits

- **Fire shape in strong wind.** The length-to-breadth ratio is within 7 % of the ellipse up
  to a ratio of 3.2 (16 m/s wind on grass). Along a grid axis the fire is 25 % too narrow at
  4.5 and 70 % at 6.8.
- **Forest still burns slower than observed.** With the calibrated table, pine forest (TU2)
  spreads at a quarter to a half of the rates measured in Scots pine stands
  (see [Calibration](#calibration)).
- **Meadows and cropland are probably too flammable.** The published table puts both on GR6,
  which carries a 3.6 MW/m head at 5 m/s wind in dry weather; managed green grassland rarely
  burns like that.
- **Sheltering from wind is all or nothing**: timber models get one sheltered factor, every
  other fuel the open-ground one. There is no canopy-cover layer.
- **The Rothermel kernel is checked on two fuel models** against Behave, and the fuel table on
  all forty. A reference table over all forty models has not been generated.
- **Water**: the critical load follows published values, but how long water lasts
  (`water_tau_s`), `ember_s`, the share of a drop that reaches the fuel and the footprint of
  a drone's drop (all of it spread evenly over one 10 m cell) are assumptions. A real drop of
  40 L wets a strip of a few metres; the model asks for the same litres per metre of front
  as Ausonio et al. at low intensity (see [Calibration](#calibration)), but cannot say where
  within a cell the water has to go.
- **The drones are simple**: one speed, an endurance that does not depend on the load, no
  wind on the drones, no separation at swarm size (height lanes are assumed), every station
  with the same number of drones, and all ready drones can take off in the same tick. A drone
  on its way is not sent elsewhere unless its target goes out. Endurance is checked when a
  drone is sent, not while it flies.
- **Under FireSPIN no fireline intensity exists**, so `intensity`, `Stopped_kW_*` and
  `Peak_kW*` are zero there. A cell's intensity is fixed when it ignites and does not follow a later wind
  change. Along an edge the wettest cell decides, which is slightly too cautious for partly
  wet paths.
- **FireSPIN is not calibrated either**: its constants are the old ones, and no planner
  comparison has been run on it yet. A knocked-down FireSPIN cell forgets how much of its
  fuel had burned. The cell inspector's spread rate is always Rothermel's.
- **PPO on the planner task makes an imitated policy worse**, and from scratch stays below
  greedy. Only imitation of the rollout planner works so far; one network, one seed.
- **One training seed** stands behind the fly-policy numbers.
- **The planner sees the true fire.** Drones can sense it (`sense_m`) and the belief they
  build is measured, but no planner reads it yet. The sensor has no errors, no smoke and no
  communication range.
- **The scripted controller has no deadlock handling** beyond sidestepping (only with
  `safe_m` above zero); none occurred in the tests. A drone hovering over a station would
  block its launches then.
- **Random ignitions use rejection sampling**: a map that is almost all non-burnable gets
  fewer fires than asked for.
- **The viewer's mouse and keyboard actions, menus, file dialogs and Settings window were not
  tested automatically.** Window captures confirm that the fire view, the allocate view with
  the inspector, the fly-task view and real terrain with relief render; unit tests cover the
  map painting, the settings form, learned flight under a planner and the switch to the fly
  task; the check script confirms that policies are called and every planner runs.
- **Colouring by spread rate or flame length shows the start**: it is not updated when the
  wind schedule moves on, and it uses the Rothermel rate also under FireSPIN.
- **The rollout planner in the viewer computes under the simulation lock** and can stall the
  picture for a moment at each decision.
