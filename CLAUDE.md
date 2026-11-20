# ROSHAN2

Wildfire simulation and multi-UAV reinforcement learning platform: a Rust simulator with
Python bindings, training and evaluation in PyTorch, and an egui viewer. `README.md` lists
everything that can be run; `docs/systems.md` describes how each part works and is the
reference to read before changing a subsystem. `docs/design.md` is the original plan.

## Layout

- `crates/roshan-core/` — the simulation. Pure Rust: no Python, no UI, no file or network
  access. Fire models (`rothermel.rs`, `fire.rs`, `spin.rs`), water, terrain, the two tasks
  (`navigate.rs`, `allocate.rs`).
- `crates/roshan-ui/` — viewer and scenario editor; depends on the core only.
- `crates/roshan-py/` — Python bindings, module `roshan._core`.
- `python/roshan/` — environments, baselines, `eval/`, `rl/`, `geo.py` (real terrain),
  `scenario.py`, `experiment.py` (experiment files).
- `experiments/` — `.toml` files that say what a training or evaluation run does. Every run
  writes the full settings it used to `runs/<name>/experiment.toml` (`evaluation.toml`),
  with its results, the uncommitted code (`code.patch`) and a digest of the simulator's
  behaviour. A run directory that is not empty is refused without `--force`.
- `scenarios/` — `.npz` scenario files. `splits/` — frozen validation and test seeds.
  `runs/` — training and evaluation output, ignored by git.

## Commands

```
cargo fmt --all
cargo clippy --workspace --all-targets
cargo test --workspace
uv run pytest                                   # rebuilds the extension when Rust changed
uv run python -m roshan.ui                      # viewer on the demo scenario
cargo run --release -p roshan-core --example bench       # fire and allocate timings, shape accuracy
cargo run --release -p roshan-core --example calibrate   # tables behind the calibration
```

`uv run ...` rebuilds `roshan._core` whenever a Rust source changed, so Python never runs a
stale build. The planner benchmark is `experiments/eval_demo.toml` on `scenarios/demo.npz`.

## Rules

- **Runtime speed comes first** in the simulator. Nothing per cell or per tick that can be
  per fuel or per episode. Check `--example bench` before and after touching the fire step.
- **Determinism.** An episode depends on its seed alone, results do not depend on the thread
  count, and a change in one place on the map never changes the fire somewhere else. Tests
  enforce all three; keep new state local and order-independent.
- **Every viewer control has a tooltip** (`on_hover_text`), new ones included. A new config
  parameter needs an entry in `hint()` in `crates/roshan-ui/src/settings.rs`; a test fails
  without one.
- **`ponytail:` comments** mark deliberate simplifications and name their limit and the
  upgrade path. Keep them when editing nearby code and add one for any new shortcut.
- **Numbers need a source.** Constants that shape fire or water behaviour are documented in
  `docs/systems.md`, section "Calibration", with where they come from or the word "assumed".
  Do not change one without updating that section.
- **Docs follow the code.** A change in behaviour, a config key or a result updates
  `docs/systems.md` (and `README.md` if a command changed) in the same change. Results state
  the date and the conditions they were made under.

## Things that bite

- After changing `CROSSWALK` in `python/roshan/geo.py`, rebuild the fuel of saved real-terrain
  scenarios: `uv run python -m roshan.geo refuel scenarios/<file>.npz`.
- After changing `crates/roshan-ui/src/demo.rs`, regenerate `scenarios/demo.npz` and
  `scenarios/demo_static.npz` from `Scenario.from_dict(_core.demo_scenario())` (the static
  one has moisture `[0.14, 0.15, 0.16, 1.2, 1.5]` and the name `demo_static`).
  Both are the planner benchmark maps, so every planner result changes with them.
- Changing the fire, the water model or the demo map invalidates the planner results in
  `docs/systems.md` and the checkpoints in `runs/`. Re-run the untrained planners with
  `python -m roshan.eval` and mark learned results that were not redone as older.
- The viewer's own defaults (spread-rate noise 0.25, a one-day limit) are not an experiment's.
  To look at an episode of a run, start it with `--experiment <file> --seed <N>`; the
  `fire #` in the status bar must equal the CSV's `Fire_Fingerprint`.
- The viewer refuses to save over a map that a file in `experiments/` runs on.
- A scenario holds ground stations, not drones: `drones_per_station` in the allocate
  settings says how many are based at each. A scenario without a station runs the fire only.
- An evaluation row holds NaN where nothing was measured (`Min_Separation` without `safe_m`,
  `First_Drop_s` without a drop, `Detect_s` without a sensor): compare rows through `json.dumps`, not `==`.
- `Water` copies `water_crit_l_m2` when the world is built; changing the config on a running
  world has no effect on it.
- The viewer cannot be clicked from an automated session; menu, mouse and keyboard behaviour
  needs a person to try it.
