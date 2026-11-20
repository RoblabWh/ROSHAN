# ROSHAN2: platform reimplementation plan

## Context

ROSHAN (`/home/nex/Code/ROSHAN`, ~15k LOC C++, ~8.5k LOC Python) is the master-thesis system:
FireSPIN wildfire simulation plus hierarchical multi-UAV RL. It is reimplemented in
`/home/nex/Code/ROSHAN2` (empty today) as the base for future experiments. A paper follows later;
there is no deadline.

Why a reimplementation and not a cleanup:

- **Speed.** Training runs at ~300-400 env steps/s, one environment per process. Python owns the
  loop, resets travel through the GUI event pump, and a shared Python dict is the control plane.
- **Coupling.** Physics headers include ImGui/SDL/pybind11, RL belief maps live in the fire grid,
  a Node map server starts in every mode, and defaults live in YAML, Python and C++ at once.
- **The RL result is capped by the task, not the network.** The learned planner ties greedy and
  Hungarian because the benchmark is static routing: fire does not spread within a ~300 s mission
  and extinguishing is instant. Published learned allocators also trail Hungarian on static
  allocation. The fire model and the task must change before RL has headroom.

## Decisions taken in the brainstorm

| Topic | Decision |
|---|---|
| Goal | Platform first; experiments and paper build on it later |
| Core | Rust, natively batched environments, exposed to Python via PyO3/maturin; PyTorch stays for RL |
| Fire models | Rothermel-driven CA (default, first) and a FireSPIN port (second) behind one boundary |
| Suppression | Per-cell water load that decays and slows spread; no instant extinguish |
| UI | egui/eframe, in-process, launched from Python. Jobs: debug sim and policies, author scenarios |
| Low-level control | Scripted controller in the core by default; learned FlyAgent optional in the loop |
| Order | Core and fire, FlyAgent training, scenario authoring, then the planner track |
| Dropped | TD3, IQL, RND, learnable explorer, in-process Optuna, LLM box, startup wizard, Node server, public web demo, replay, remote attach |

## Architecture

```
ROSHAN2/
  Cargo.toml              workspace
  pyproject.toml          maturin (abi3 wheel) -> module roshan._core; dependencies via uv
  crates/roshan-core/     pure Rust: no pyo3, no egui, no file or network I/O
  crates/roshan-ui/       egui viewer/editor; depends on roshan-core only;
                          examples/viewer.rs runs without Python for fire-model work
  crates/roshan-py/       PyO3 bindings; feature "ui" pulls in roshan-ui, feature "map" the map picker
  python/roshan/
    env.py config.py bench.py
    rl/                   ppo.py, networks/
    baselines.py          greedy, Hungarian
    eval/                 run.py, paired.py
    geo/                  fetch.py, crosswalk.csv, import_tif.py
    ui.py                 python -m roshan.ui
  configs/ scenarios/ splits/ tests/ docs/
```

Cargo enforces the dependency direction: the core cannot see the UI or Python. That is the
structural fix for the old coupling.

### Core (roshan-core)

- **World** (one environment, `Clone`). Static terrain is shared through `Arc` (land cover, fuel
  id, elevation, slope, aspect, asset value), so a clone copies dynamic state only: fire layers,
  water load, drones as struct-of-arrays, stations, tick, episode statistics. One index convention
  everywhere (`idx = y*nx + x`); every test uses a non-square grid.
- **Clock.** Time is an integer tick count of the control step. Fire updates every k ticks, with
  automatic sub-steps so the front never crosses more than half a cell per step.
- **Weather.** Fuel moisture per episode and a wind schedule (one entry = constant wind). Wind
  shifts are part of what makes allocation dynamic.
- **Fire model boundary.** `step(dt, &Terrain, &Weather, &Water, &Rng, &mut FireState)`. Agents,
  tasks and the renderer read only the shared `FireState`. Rothermel is a concrete struct first;
  the boundary becomes an `enum FireModel` when FireSPIN lands (M8), with no trait objects.
- **Water layer** (owned by the World, read by every fire model). A drop adds litres/m² over a
  footprint; the load decays; `ros_mult = clamp(1 - w / w_crit)`; a burning cell is held only above
  `w_crit` and can re-ignite. Constants are config values marked "to calibrate".
- **RNG.** Counter-based Philox keyed by (episode seed, stream, tick, entity), with known-answer
  tests. A difference in suppression then changes only the cells it touches, results do not depend
  on thread count, and a different policy never perturbs the fire. `libm` for exp/pow in the fire
  kernel, so later experiments stay comparable across machines.
- **Tasks.**
  - `navigate`: continuous 2D velocity per drone to a waypoint, with collision avoidance.
  - `allocate`: one target per drone from a masked set. Targets are fire-front cells binned into
    blocks (default 50 m), top K by score, plus ground stations; block size of one cell reproduces
    the old per-cell targets. A drone commits to a point, not an index.
  - Each environment runs until the next decision event (arrival, tank empty, target gone, new
    fire, or maximum interval) and returns elapsed seconds plus a per-drone `decide` mask. Only
    drones that need a decision are asked. This replaces the commitment lock.
  - One definition of "drone is free" (the old code had four, with different tolerances).
- **Observations and rewards.** Rust writes named entity tables (drones, targets with mask) and a
  reward-term vector into caller-owned buffers; names are exported in `env.spec`. Python computes
  `reward = terms @ weights` and any shaping. A new reward combination is a config change; a new
  primitive is one row (name, dim, fn) in a Rust registry. Networks bind features by name and a
  checkpoint refuses to load against a different spec.
- **VecEnv.** N worlds stepped with rayon in its own thread pool, GIL released. Rust borrows the
  caller's arrays for the duration of a call and never stores pointers; dtype, shape and
  contiguity are checked and rejected, never copied silently. Same-step auto-reset with separate
  `terminated`/`truncated` flags and `final_obs` for bootstrapping. Evaluation runs with auto-reset
  off and an explicit seed list, so each seed runs exactly once.
- **Config.** `SimConfig` serde structs with defaults; unknown fields are rejected. Python gets the
  defaults as a dict, layers YAML on top (OmegaConf, as today) and passes JSON back. Training
  config is Python-only. No key is read by both sides.

### Fire models

1. **Rothermel CA (default).** Rothermel surface rate of spread with the 40 Scott & Burgan fuel
   models; wind, slope and moisture; elliptical directional spread; a burning cell accumulates
   ignition progress in its neighbours, with optional per-cell noise. SI units throughout, with the
   imperial conversion inside one module. Written from the published equations: ELMFIRE,
   Pyretechnics and ForeFire are AGPL, GPL or custom-licensed, so they are offline oracles only.
2. **FireSPIN port (M8).** Convection and radiation particles as in the thesis, with the known bugs
   fixed rather than ported: integer-truncated spawn jitter, undefined cast for particle-less
   classes, class index past the table, random buffer never reset between episodes.

### Python side

- **PPO.** One compact vectorised file. Gamma is defined per second and GAE discounts by
  gamma^elapsed, so changing the decision interval does not change the objective; fixed tick is
  the special case. Log-probability and entropy are summed over deciding drones only.
- **Checkpoints** hold the state dict, network import path and arguments, `env.spec`, sim config,
  git sha and a schema version. No copied source files.
- **Evaluation.** Episode summaries to CSV, paired statistics, frozen validation and test seed
  lists in `splits/`. Because random draws are shared, each scenario also gets an exact no-action
  run, which gives an "area saved" metric.
- **Geo.** `python -m roshan.geo fetch --bbox ...` reads CLC+ Backbone 2021 (10 m) from the EEA
  image server and elevation from the Copernicus 30 m DEM on AWS, both anonymous, resamples both
  onto one EPSG:3035 grid and applies a documented CLC+ to fuel-model table. This replaces the
  15 GB raster, GDAL in the core and the Node server. Uptime and rate limits of the EEA service
  are unverified; tests run offline against a fixture.
- **Logging.** TensorBoard.

### UI (roshan-ui)

- `python -m roshan.ui [--scenario ...] [--policy ckpt]`. The window runs on the Python main thread
  with the GIL released. The simulation sits behind a mutex and is stepped by its own thread; the
  UI locks once per frame to draw. The sim thread, not the UI thread, calls the PyTorch policy, and
  never while holding the sim lock. Ctrl+C is polled every frame. A policy exception pauses the sim
  and shows the traceback.
- **Authoring**: place fires, drones and ground stations, set weather, save and load scenarios;
  fetch terrain through `roshan.geo` from a bounding box, then from a map rectangle picker
  (`walkers`, behind the "map" feature because it adds an HTTP client to the extension).
- **Debug**: pan/zoom, pause/step/speed, click to ignite or drop water, cell inspector, live fire,
  wind and moisture parameters, drone inspector with observation table and reward terms, planner
  targets and `decide` mask, manual drone control, legend.
- Not carried over: startup wizard, train/eval control from the GUI, log tail, themes.

## Milestones

Vertical slices, each with a check. Commands assume `uv run` for Python.

| # | Slice | Acceptance check |
|---|---|---|
| M0 | Toolchain, workspace, two spikes. (a) egui window started from Python with a sim thread calling a torch policy. (b) Batched step into caller-owned pinned buffers. | (a) Policy called >100 times, SIGINT returns KeyboardInterrupt within 0.5 s, a second `run` in the same process works, import works without a display. (b) `np.shares_memory` holds, non-contiguous input is rejected, >=12x speedup at 24 threads on a dummy env. `pytest tests/test_buffers.py`, `python spikes/ui_spike.py --auto` |
| M1 | World, clock, RNG, Rothermel CA, water layer, minimal viewer | `cargo test -p roshan-core`: spread rate within 1% of a reference table generated once from an established Rothermel implementation; ellipse length-to-breadth within 10% of Anderson; round shape without wind; identical state hash at 1 and 24 threads; suppressing one corner leaves the far corner bit-identical; a wet strip holds a grass front and releases it after decay |
| M2 | `navigate` task, Python API, PPO, FlyAgent retrained | >=95% goal-reach with collision termination on, 4 drones, test seeds (old best: 98.75%); throughput target below; ported GAE tests pass |
| M3 | Scenario authoring: geo fetch, UI editor, map picker last | `pytest tests/test_geo.py` on a fixture; save, load, identical fingerprint; a real-area scenario authored and run |
| M4 | Scripted controller, drone water drops, `allocate` task, greedy and Hungarian, evaluation harness | Ported baseline and paired-eval tests pass; identical per-seed summaries at 1 vs 64 envs and 1 vs 24 threads; 10k missions with 16 drones and zero separation violations |
| M5 | **Headroom gate.** Rollout planner on cloned worlds; calibrate the dynamic regime | Paired interval for rollout minus greedy excludes 0 on the dynamic regime, greedy beats no-action, and the static regime reproduces today's tie. No planner training before this passes |
| M6 | UI debug tools: policy attach, observation and reward-term inspector, manual control | Checklist of the debug features above |
| M7 | PPO on `allocate`; FlyAgent-in-the-loop check | On the static regime the learned planner is within the interval of greedy; on the dynamic regime paired results against Hungarian and rollout are reported |
| M8 | FireSPIN port; fire boundary becomes an enum | Both models run both tasks; spread curve compared against the old C++ on a uniform map |
| Later | Sensing, belief maps, scripted explorer | When partial-observability experiments start. Sensing, the belief map and scripted explorers were built on 2026-11-20 (docs/systems.md, "The allocate task"). The planners still read the true fire: targets are built from each front's water need and intensity, not from a burning mask as this line once said, so the belief has to carry those before it can replace the truth |

### Throughput targets

Estimates, not measurements. Measured at M0/M1/M2/M4 as sustained rate over 60 s after warm-up,
with a thread sweep, and revised if wrong. The CPU is probably 8 performance plus 16 efficiency
cores, so expect 12-15x scaling, not 24x.

| What | Target |
|---|---|
| Fire step, 200x200 grid, 5% active cells | <=100 µs |
| 60-minute mission, single thread | <=100 ms |
| `navigate`, simulator only | >=2M drone-steps/s |
| `navigate`, end-to-end PPO | >=200k drone-steps/s (today 300-400 env steps/s) |
| `allocate`, simulator only, 256 envs | >=30k decisions/s |
| `allocate`, end-to-end PPO | >=10k decisions/s |

## Open research choices

Settled at M5 with measurements, not now; the platform keeps all of them configurable.

- **Physical regime.** Default: initial attack on small ignitions with small-UAV payloads
  (10-200 L, so a drop is at most a few cells and the tactic is wet lines). Team size, payload and
  `w_crit` are tuned until the regime is contested: neither trivial nor hopeless.
- **Headline metric.** Area saved against the no-action run, time to containment and value lost
  are all recorded; the headline is chosen when the paper is written.
- **Observability.** The planner sees the true fire state first; belief maps come with the
  "Later" milestone.

## Reuse from the old repository

| Old file | Use |
|---|---|
| `src/pysim/analysis/paired_eval.py`, `tests/test_paired_eval.py` | Port almost verbatim to `eval/paired.py` |
| `src/pysim/planner_agent.py` (`greedy_actions`, `hungarian_actions`), `tests/test_planner_baselines.py` | Port to `baselines.py`; commitment lock dropped |
| `src/pysim/algorithms/ppo.py` (SMDP GAE), `tests/test_ppo_smdp.py` | Port the gamma^k logic and tests into `rl/ppo.py` |
| `src/pysim/networks/network_fly.py`, `network_planner.py` | Port and trim (dead velocity history, integer id input) |
| `src/reinforcementlearning/feature_definitions.h` | Keep the registry idea (name, dim, fn) |
| `agents/fly_agent.cpp:142-250`, `agents/planner_agent.cpp:190-456` | Source for the reward-term list |
| `src/firespin/fire_generator.cpp` | Cluster, ring and scattered ignition patterns |
| `src/firespin/particles/*.cpp`, `cell_classes/*.h` | FireSPIN equations and per-class constants (M8) |
| `maps/*.tif` | Imported once as scenarios |

Not carried over: the step-budget formula (replaced by a mission duration in simulated seconds),
the shared status dict, loading networks from copied source files.

## Deliberately not built

WebAssembly build, replay files, remote attach, C ABI for PufferLib, arrival-time and level-set
solvers, spotting, spatial wind fields, heterogeneous vehicles, communication limits, checkpoint
compatibility with old models, GeoTIFF I/O in Rust, CI and Docker. Each is added when an
experiment needs it.

## First actions on approval

1. Delete the two stray files an explorer created in the old repo:
   `maps/35x36_Woods.tif.aux.xml` and `maps/Borkenberge.tif.aux.xml`.
2. `git init` in `/home/nex/Code/ROSHAN2`. No commits unless asked.
3. Install Rust with rustup (this touches `~/.cargo`, `~/.rustup` and the shell profile) and add
   maturin as a dev dependency.
4. Write `docs/design.md` from this plan, then start M0.

## Verification

- `cargo test --workspace` and `cargo bench -p roshan-core`
- `uv sync && uv run maturin develop --release && uv run pytest tests/`
- `uv run python -m roshan.bench navigate` and `... allocate` print the throughput table
- `uv run python -m roshan.rl.train configs/navigate.yaml` reaches the M2 success rate
- `uv run python -m roshan.eval --policy greedy --split val`, then
  `uv run python -m roshan.eval.paired --arm greedy ... --arm hungarian ...`
- `uv run python -m roshan.ui --scenario scenarios/woods35` for the UI checklists

## Status (2026-11-16)

M0 to M8 are done; what remains of the plan is the "Later" item (sensing, belief maps,
explorer) and the open tuning work (PPO on the planner task, calibration).

- **M8:** FireSPIN runs behind the same fire boundary as Rothermel (`sim.fire_model`), and its
  spread curve matches a transcription of the old C++ loop within 1 %.
- **M7:** the learned planner (imitation of the rollout planner) beats hungarian and trails
  rollout on the dynamic task and ties greedy on the near-static one; PPO does not improve on
  imitation (open problem); learned flight under a planner works but saves less area than the
  scripted controller.

What the system implements today is documented in [systems.md](systems.md), with all numbers
in its [Results](systems.md#results) section; commands are in the [README](../README.md).
Targets against measurements on the first development machine (24 cores, RTX 5090 Laptop):

| What | Measured | Target |
|---|---|---|
| `navigate`, simulator only | 18 M drone-steps/s on 1 thread, 160 M on 24 | >=2 M |
| `navigate`, end-to-end PPO | 1.0 M drone-steps/s | >=200k |
| Fly policy, 1000 unseen episodes, 4 drones | 99.4% success, 0.6% collision (one training seed, ~2 min) | >=95% |
| Straight-line pilot without avoidance | 50.6% success | baseline |
| 60-minute mission, 100x100, one thread | 3.5 ms | <=100 ms |
| Fire step | 14 ns per burning cell | <=100 µs at 2000 cells |
| World clone, 100x100 | ~60 µs | - |

Where the implementation differs from the plan above:

- **Fire spread** uses one travel-time edge per source and direction (16 directions) instead of
  summing progress per neighbour. Summing lets several burning neighbours speed a front up.
  Shape accuracy is in `crates/roshan-core/src/fire.rs` and printed by the bench example.
- **Reset**: the step after an episode ends resets the environment and ignores its action, so
  the final observation is visible once. This replaces same-step reset with a `final_obs` buffer.
- **Navigate observations** are a single frame (the old system stacked three); velocity is an
  explicit feature.
- **No Task trait yet.** `navigate` is concrete; the shared batching is extracted when
  `allocate` arrives (M4).
- **Benchmarks** are example binaries that print timings, not criterion.
- **Rothermel reference**: checked against the Behave library's regression values for two fuel
  models and against its fuel table for all forty. The planned table over all forty models
  needs an established Rothermel package installed and has not been generated.
- **Config** reaches Rust as JSON with unknown keys rejected, as planned; training flags are
  plain argparse until M4.
