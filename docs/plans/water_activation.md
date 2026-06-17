# Water-as-resource activation plan (v2, revised post-decoupling)

Status: **ready to execute**. Prerequisite (decoupling `use_water_limit` from `use_heuristic`) is done.

Revision history:
- **v2 (2026-04-22)**: Rewritten after reviewing current code. Reorders steps so the reward-shaping fix comes *first* — it's a blocker, not cleanup. Flips the observation-design recommendation to a dedicated SET group. Flags two reward interactions (`FlyingTowardsGroundstation`, `SameGoalPenalty`) that would have made v1 untrainable.
- v1: pre-decoupling draft.

## Context

**Why**: The heuristic (greedy nearest-fire) is nearly optimal against the trained PlannerAgent because the current task is structurally a decomposable bipartite assignment problem — and greedy is near-optimal for that class. Wind observations didn't help because they don't change the structure.

**The structural fix**: Finite water + groundstation refueling converts the task from bipartite assignment into a **capacitated vehicle routing problem (CVRP)** — NP-hard, with non-decomposable costs. Greedy is provably suboptimal because refuel trips create temporal coupling (staggering, round-trip cost coupling, burst-vs-sustain budget). This is where a learned planner has genuine headroom over greedy.

**Intended outcome**: Trained planner beats upgraded water-aware heuristic by ≥10% on total reward / fires-extinguished / map-burned metrics.

**Key design choices (locked)**:
- Consumption: per-fire flat cost (already implemented — `fly_agent.cpp:356`: `water_capacity_ -= use_water_limit_ ? 1 : 0`).
- Refuel: time-penalty passive refill at groundstation (post-refactor — `fly_agent.cpp` passive block; rate = `water_capacity / recharge_time / dt` per step).

## Current state after decoupling refactor

| Piece | Location | State |
|---|---|---|
| `water_capacity_` field + reset + decrement | `fly_agent.{h,cpp}` | Wired, gated on `use_water_limit_` |
| Passive refuel at groundstation | `fly_agent.cpp` (planner-FlyAgent branch) | **New** — flag-free, fires whenever `use_water_limit_ && at_groundstation && water < max`. Applies to planner-spawned FlyAgents regardless of heuristic flag. |
| Heuristic FlyPolicy water-aware state machine | `fly_agent.cpp:418-444` | Unchanged — self-contained EXTINGUISH → FLY → RECHARGE loop |
| Config flags | `config.yaml:280-288` (`use_water_limit`, `water_capacity`, `recharge_time`) | All present, `use_water_limit: false` |
| Groundstation as planner action | `planner_agent.cpp:235-240` (index 0 in fire_positions RELATIONAL), unmasked (commit `0b36127`) | Ready |
| `water_level` observation hint | `feature_definitions.h:70-73` (commented example in FlyAgent schema) | Breadcrumb — shows registration pattern |

**The one true blocker**: water level is not in the planner observation. Also two reward interactions will sabotage training unless fixed first.

## Plan (ordered by dependency)

### Step 1: Fix `FlyingTowardsGroundstation` (BLOCKER)

Current code — `src/reinforcementlearning/agents/planner_agent.cpp:141-146`:

```cpp
if (goal_position == groundstation_pos && !fire_positions->empty()) {
    reward_components["FlyingTowardsGroundstation"] = parameters_.PlannerFlyingTowardsGroundstation_;
}
```

This fires **every step** a drone is assigned to the groundstation while fires remain — including every step of refueling. At -0.29 × ~20 refuel-steps, a single refuel trip costs -5.8. A full tank yields at best +2.44 from `ExtinguishFires` (4 × 0.61). Net: refuels are massively unprofitable; training will learn "never refuel." This penalty predates water activation — its original intent was blocking the degenerate "always assign groundstation" fallback during early training.

**Recommended fix** (gate by water level):

```cpp
double water_frac = fly_agent->GetWaterCapacity() / parameters_.GetWaterCapacity();
constexpr double kRefuelTolerance = 0.5;  // above 50% tank, groundstation is "unnecessary"
if (goal_position == groundstation_pos && !fire_positions->empty()
    && water_frac > kRefuelTolerance) {
    reward_components["FlyingTowardsGroundstation"] =
        parameters_.PlannerFlyingTowardsGroundstation_ * water_frac;
}
```

- When tank is above threshold: penalty applies, scaled by tank fullness (full tank = full penalty; threshold-full = half).
- When tank is low: no penalty at all — refueling is legitimate.
- Preserves original intent (discourage degenerate early-training behavior) while respecting water pressure.

Alternative (simpler, more aggressive): set `FlyingTowardsGroundstation: 0.0` in config. Lets training discover refuel strategy from other signals (`DistanceProgress`, `ExtinguishFires`, `MapBurnedTooMuch`). Cleaner if the penalty turns out not to have been doing much after other rewards were added.

**Decision**: start with the gated fix. Zero-out only if the gated version still suppresses refuels. Add threshold to config if it needs tuning:
```yaml
planner_agent:
  rewards:
    FlyingTowardsGroundStation: -0.29328...  # (existing)
    FlyingTowardsGroundStation_water_threshold: 0.5  # NEW
```

### Step 2: Exempt groundstation from `SameGoalPenalty`

Current code — `planner_agent.cpp:148-153`:

```cpp
std::set<std::pair<double, double>> unique_goals(fly_agent_goals.begin(), fly_agent_goals.end());
if (unique_goals.size() < fly_agent_goals.size()) {
    reward_components["SameGoalPenalty"] = parameters_.PlannerSameGoalPenalty_ *
        static_cast<double>(fly_agent_goals.size() - unique_goals.size());
}
```

The penalty fires when two drones share any goal — including the groundstation. With water activation, simultaneous refueling can be strategically correct (all drones depleted near-simultaneously after a big burn event). The penalty would incorrectly discourage this. Exempt the groundstation:

```cpp
std::vector<std::pair<double, double>> non_gs_goals;
non_gs_goals.reserve(fly_agent_goals.size());
for (const auto& g : fly_agent_goals) {
    if (g != groundstation_pos) non_gs_goals.push_back(g);
}
std::set<std::pair<double, double>> unique(non_gs_goals.begin(), non_gs_goals.end());
if (unique.size() < non_gs_goals.size()) {
    reward_components["SameGoalPenalty"] = parameters_.PlannerSameGoalPenalty_ *
        static_cast<double>(non_gs_goals.size() - unique.size());
}
```

Small, clean, independently defensible — worth doing even if we later back out of water activation.

### Step 3: Add `water_level` to the planner observation (core code change)

**Design decision (revised from v1)**: add a new SET group `drone_water` with `bulk_dims=1`. **Do NOT extend `drone_positions` in place** — it uses `MakePairBulkExtractor` (feature_definitions.h:23-39), a generic helper hardcoded to pair-of-floats; extending it would break reuse and force a custom extractor for what should stay a spatial-only group.

**Files to modify:**

**(a)** `src/reinforcementlearning/agents/agent_state.h` — add field:

```cpp
std::shared_ptr<std::vector<double>> drone_water_levels;  // one entry per drone, in [0,1]
```

**(b)** `src/reinforcementlearning/agents/planner_agent.cpp:BuildAgentState()` (~line 209) — populate alongside existing drone_positions loop:

```cpp
std::vector<double> drone_water;
drone_water.reserve(fly_agents_.size());
for (const auto& fly_agent : fly_agents_) {
    drone_water.push_back(
        fly_agent->GetWaterCapacity() / static_cast<double>(parameters_.GetWaterCapacity())
    );
}
// ...
state->drone_water_levels = std::make_shared<std::vector<double>>(std::move(drone_water));
```

Normalize by CURRENT capacity so the observation is always "fraction of tank remaining" — scale-invariant across any `water_capacity` tuning.

**(c)** `src/reinforcementlearning/feature_definitions.h:CreatePlannerAgentSchema()` — register a new SET group (after `drone_positions`, before `goal_positions`):

```cpp
auto& drone_water = schema.AddGroup("drone_water", FeatureGroupType::SET);
drone_water.bulk_dims = 1;
drone_water.entity_count = [](const AgentState& s) {
    return s.drone_water_levels
        ? static_cast<int>(s.drone_water_levels->size()) : 0;
};
drone_water.extract_bulk = [](const AgentState& s, float* data, bool* mask, int M, int D) {
    const auto& vec = *s.drone_water_levels;
    const int n = static_cast<int>(vec.size());
    for (int i = 0; i < M; i++) {
        mask[i] = i < n;
        data[i] = mask[i] ? static_cast<float>(vec[i]) : 0.0f;
    }
};
```

**(d)** `src/pysim/networks/network_planner.py` — the network consumes the feature schema by group name. Verify:
- The drone-state encoder (self-attention over per-drone features) can accept the new `drone_water` group, either by concatenation with `drone_positions`/`goal_positions` along the feature dim, or by adding a separate projection and fusing.
- If per-drone features are currently built by indexing a fixed tuple `(pos_x, pos_y, goal_x, goal_y)`, extend to `(pos_x, pos_y, goal_x, goal_y, water)`.
- Recent commit `0b36127` introduced an autoregressive decoder; its per-drone input tensor needs the water dim added.

This is the most network-specific work. Prototype with a print-shape-at-forward-pass pass to confirm the feature tensor flow.

### Step 4: Activate config and calibrate

```yaml
environment:
  agent:
    use_water_limit: true       # was false
    water_capacity: 4           # tune: ~3-5 extinguishes per tank
    recharge_time: 2.0          # TBD — see calibration below
    planner_agent:
      rewards:
        FlyingTowardsGroundStation_water_threshold: 0.5   # NEW (if using gated fix from Step 1)
```

Mirror to `config_planner_exp_a.yaml`, `config_planner_exp_b.yaml`, `notebook_config.yaml` as appropriate.

**Calibrating `recharge_time`:**

- Steps-to-fully-refill = `recharge_time / dt`.
- Target: full refill ≈ time for a typical one-way flight across the map. With `dt=0.1s` and a default map diagonal of ~30 cells at 10 m/s: ~1.5s one-way → `recharge_time: 1.5` to `3.0` gives 15-30 refill steps.
- **Starting value: `recharge_time: 2.0`**.
- Sweep: ×0.5, ×1, ×2.

### Step 5: Defer (not block) optional additions

Do not add the following unless Step 4 training shows pathology:

- **`RefuelComplete` bonus** (small positive when water hits max while at groundstation). Addresses the concern that refuels currently have no direct positive signal. Add if drones oscillate at the groundstation (leaving after only partial refill repeatedly).
- **`WaterDepleted` penalty** (small negative when water=0 and not at groundstation). Adds pre-emptive refuel pressure. Add only if drones consistently run empty before returning.
- **Curriculum** (start water_capacity=8, anneal to 4). `DistanceProgress` already gives dense gradients for approach, so curriculum is less critical than v1 implied. Add only if initial training diverges.
- **Heuristic baseline strengthening** (pre-emptive refuel threshold in `fly_agent.cpp:418`). Only relevant when running the fairness-comparison experiment; not needed for trained-planner training itself.

## Critical files to touch (updated)

| File | Change |
|---|---|
| `src/reinforcementlearning/agents/planner_agent.cpp:141-146` | Gate `FlyingTowardsGroundstation` by water level (Step 1) |
| `src/reinforcementlearning/agents/planner_agent.cpp:148-153` | Exempt groundstation from `SameGoalPenalty` (Step 2) |
| `src/reinforcementlearning/agents/agent_state.h` | Add `drone_water_levels` field (Step 3a) |
| `src/reinforcementlearning/agents/planner_agent.cpp:~209` (`BuildAgentState`) | Populate water levels (Step 3b) |
| `src/reinforcementlearning/feature_definitions.h:CreatePlannerAgentSchema` | Register `drone_water` SET group (Step 3c) |
| `src/pysim/networks/network_planner.py` | Consume new feature group in per-drone encoder and autoregressive decoder (Step 3d) |
| `src/firespin/model_parameters.h` + config YAML parse | Add `FlyingTowardsGroundStation_water_threshold` field if using gated fix (Step 1) |
| `config.yaml:284-288` + experiment configs | Flip flags, set tuning (Step 4) |

## Verification

1. Build: `cd build && cmake .. && make -j$(nproc)` — clean.
2. Module import: `.venv/bin/python -c "import firesim"`.
3. **Observation sanity**: log the `drone_water` group shape and values from one rollout step. Expect shape `(num_drones, 1)`, values in `[0, 1]`, non-constant across steps.
4. **Reward sanity**: log all reward components for one episode. Confirm:
   - `FlyingTowardsGroundstation` does NOT fire when the assigned drone has water < threshold.
   - `SameGoalPenalty` does NOT fire when two drones are both at the groundstation.
5. **Heuristic-mode smoke test** (unchanged behavior): `hierarchy_type: fly_agent`, `rl_mode: eval`, `use_heuristic: true`, `use_water_limit: true`, small map. Drones deplete, fly home, refill, return. Should behave identically to pre-refactor.
6. **Trained-planner smoke test** (the new path): `hierarchy_type: planner_agent`, `rl_mode: train`, `use_heuristic: false`, `use_water_limit: true`, short run (50k steps). Expect non-degenerate policy (drones occasionally refuel, occasionally don't based on water).
7. **Baseline comparison**: trained planner vs. (water-aware) heuristic. Log mean reward, fires extinguished, map burned %, mean water-starved-time per drone. Target: planner ≥10% better on ≥2 metrics.

## Open questions (user input may be needed)

- **Gated vs. zeroed `FlyingTowardsGroundstation`?** Gated (default) preserves original intent. Zero-out is simpler and defensible.
- **`FlyingTowardsGroundStation_water_threshold` default: 0.5?** Alternatives: 0.33 (only penalize nearly-full), 0.7 (penalize more aggressively). Tunable.
- **Starting `water_capacity` and `recharge_time`?** Proposed 4 / 2.0. Sweepable.
