//
// Created by nex on 21.06.25.
//

#include "planner_agent.h"
#include <iostream>


PlannerAgent::PlannerAgent(FireModelParameters &parameters, int total_id, int id, int time_steps) : Agent(parameters, 300) {
    total_id_ = total_id;
    id_ = id;
    agent_sub_type_ = "planner_agent";
    agent_type_ = PLANNER_AGENT;
    time_steps_ = time_steps;
    frame_skips_ = parameters_.planner_agent_frame_skips_;
    eval_mode_ = parameters_.init_rl_mode_ == "eval";
    frame_ctrl_ = 0;
}

void PlannerAgent::PerformPlan(PlanAction *action, const std::string &hierarchy_type,
                               const std::shared_ptr<GridMap> &gridMap) {

    // Count only fires extinguished in this step
    extinguished_fires_ = 0;

    // Iterate over all Actions and set a new goal for each FlyAgent.
    // Water depletion no longer forces a groundstation goal here; the planner's action is
    // honored directly, and refuel is handled passively when a drone co-locates with the groundstation.
    for (int i = 0; i < fly_agents_.size(); ++i) {
        auto fly_agent = fly_agents_[i];
        // Get the goal from the action (in normalized observation space)
        std::pair<double, double> goal = action->GetGoalFromAction(i);
        // Denormalize from observation space back to grid space:
        // inverse of (2 * grid_pos / norm_map) - 1, where norm_map = max(rows, cols)
        double norm_map = static_cast<double>(std::max(gridMap->GetRows(), gridMap->GetCols()));
        goal = {(goal.first + 1.0) * norm_map / 2.0, (goal.second + 1.0) * norm_map / 2.0};

        if (goal == std::make_pair(-1.0, -1.0)) {
            // If the goal is (-1.0, -1.0) we set the goal to the groundstation
            // This either happens in fly_policy_eval or if the network picks this goal
            goal = gridMap->GetGroundstation()->GetGridPositionDouble();
        }

        extinguished_fires_ += fly_agent->GetNumExtinguishedFires();
        fly_agent->SetNumExtinguishedFires(0); // Reset for next step
        fly_agent->SetGoalPosition(goal);
        // If any FlyAgent has extinguished the last fire we set the extinguished_last_fire_ to true
        if (!extinguished_last_fire_){
            extinguished_last_fire_ = fly_agent->GetExtinguishedLastFire();
        }
    }
    if (hierarchy_type == "planner_agent") {
        did_hierarchy_step = true;
    }
    if (extinguished_last_fire_){
        objective_reached_ = true;
    }
}

void PlannerAgent::Initialize(std::shared_ptr<ExploreAgent> explore_agent, std::vector<std::shared_ptr<FlyAgent>> fly_agents, const std::shared_ptr<GridMap> &grid_map) {
    explore_agent_ = std::move(explore_agent);
    fly_agents_ = std::move(fly_agents);
    // Initialize the agent states with the current grid map
    InitializePlannerAgentStates(grid_map);
}

void PlannerAgent::Reset(Mode mode,
                         const std::shared_ptr<GridMap>& grid_map,
                         const std::shared_ptr<FireModelRenderer>& model_renderer) {
    (void)mode; (void)model_renderer; // unused
    objective_reached_ = false;
    agent_terminal_state_ = false;
    did_hierarchy_step = false;
    reward_components_.clear();
    goal_idx_ = 0;
    revisited_cells_ = 0;
    extinguished_fires_ = 0;
    frame_ctrl_ = 0;
    extinguished_last_fire_ = false;
    prev_drone_distances_.clear();
    prev_num_burning_ = -1.0;
    prev_water_levels_.clear();
    prev_planner_goals_.clear();
    // PBRS state — capture B_init from the freshly-reset grid_map so Φ_fire = -B(s)/B_init
    // is well-defined throughout the episode. max(_, 1) prevents div-by-zero on degenerate
    // maps. phi_initialized_=false → first reward call emits F=0.
    prev_phi_         = 0.0;
    phi_initialized_  = false;
    B_init_           = std::max(grid_map->GetNumBurningCells(), 1);
    phi_fire_last_    = 0.0;
    phi_water_last_   = 0.0;
    phi_dist_last_    = 0.0;
    perfect_goals_.clear();
    agent_states_.clear();
    Initialize(explore_agent_, fly_agents_, grid_map);
}

AgentTerminal
PlannerAgent::GetTerminalStates(bool eval_mode, const std::shared_ptr<GridMap> &grid_map, int env_steps_remaining) {
    AgentTerminal t;

    // If the agent has finished his objective(calculated in the PlanAction) it has reached a terminal state and succeeded
    if (objective_reached_) {
        t.is_terminal = true;
    }

    if (grid_map->PercentageBurned() > 0.3) {
        // If the agent let the map burn too much it has reached a terminal state and died
        t.is_terminal = true;
        t.reason = FailureReason::Burnout;
    }

    // If the agent has taken too long it has reached a terminal state and died
    if (env_steps_remaining <= 0) {
        t.is_terminal = true;
        t.reason = FailureReason::Timeout;
    }

    if (!grid_map->HasBurningFires() && !objective_reached_) {
        // Map has burned down on its own
        t.is_terminal = true;
    }

    if (t.is_terminal && t.reason != FailureReason::None) { t.kind = TerminationKind::Failed; }
    else if (t.is_terminal) { t.kind = TerminationKind::Succeeded; }
    else { t.kind = TerminationKind::None; }

    agent_terminal_state_ = t.is_terminal;
    env_steps_remaining_ = env_steps_remaining;
    return t;
}

double PlannerAgent::CalculateReward(const std::shared_ptr<GridMap>& grid_map) {
    std::unordered_map<std::string, double> reward_components;
    double total_reward = 0;

    if (objective_reached_) { // Either Objective is reached
        reward_components["GoalReached"] = parameters_.PlannerGoalReached_;
        reward_components["FastExtinguish"] = parameters_.PlannerFastExtinguish_ * (static_cast<double>(env_steps_remaining_) / static_cast<double>(parameters_.total_env_steps_));
    } else if (env_steps_remaining_ <= 0) { // or the agent has taken too long
        reward_components["TimeOut"] = parameters_.PlannerTimeOut_;
    } else if (agent_terminal_state_ && !objective_reached_) { // or the agent has reached a terminal state without reaching the objective
        reward_components["MapBurnedTooMuch"] = parameters_.PlannerMapBurnedTooMuch_;
    }

    auto groundstation_pos = grid_map->GetGroundstation()->GetGridPositionDouble();
    auto fire_positions = grid_map->GetFirePositionsFromBurningCells();

    // Each individual FlyAgent contributes to the goal
    std::vector<std::pair<double, double>> fly_agent_goals;
    for(const auto& agent : fly_agents_) {
        auto goal_position = agent->GetGoalPosition();
        // Gated on !PBRS: refuel-vs-extinguish trade-off is expressed via Φ_water in
        // PBRS mode, so this independent action-quality penalty would interfere with
        // the policy-invariance guarantee of F = γΦ(s')−Φ(s).
        if (!parameters_.PlannerPbrsEnabled_
            && goal_position == groundstation_pos && !fire_positions->empty()) {
            // Penalize groundstation goals only when the drone has enough water to make
            // refueling "unnecessary". Below the threshold the trip is legitimate and
            // should carry no penalty (otherwise every refuel step eats -0.29 × N_steps,
            // swamping the rewards for learning refuel behavior).
            // Scale by water fraction: full tank = full penalty, threshold = half penalty,
            // empty = zero (though the threshold gate already handles that).
            double water_frac = parameters_.use_water_limit_
                ? (agent->GetWaterCapacity() / static_cast<double>(parameters_.GetWaterCapacity()))
                : 1.0;  // without water limit, penalty behaves as before (always full)
            if (water_frac > parameters_.PlannerFlyingTowardsGroundstationWaterThreshold_) {
                reward_components["FlyingTowardsGroundstation"] +=
                    parameters_.PlannerFlyingTowardsGroundstation_ * water_frac;
            }
        }
        fly_agent_goals.push_back(goal_position);
    }

    // Check if there are multiple agents going to the same goal.
    // The groundstation is excluded — simultaneous refueling is a legitimate strategic
    // choice (e.g. after a coordinated burn-down) and shouldn't be penalized.
    std::vector<std::pair<double, double>> non_gs_goals;
    non_gs_goals.reserve(fly_agent_goals.size());
    for (const auto& g : fly_agent_goals) {
        if (g != groundstation_pos) non_gs_goals.push_back(g);
    }
    std::set<std::pair<double, double>> unique_goals(non_gs_goals.begin(), non_gs_goals.end());
    const int duplicate_count = static_cast<int>(non_gs_goals.size() - unique_goals.size());
    // Always emit SameGoalCount (even when 0) so TensorBoard sees the full distribution,
    // not just the subset where duplicates occurred. Kept always-on (even in PBRS mode)
    // because it's a pure diagnostic — useful for spotting oscillation regardless of
    // which shaping path is active.
    reward_components["SameGoalCount"] = static_cast<double>(duplicate_count);
    // Gated on !PBRS: action-coordination penalty isn't expressible as a state
    // potential, so it stays as independent shaping outside PBRS mode and stays off
    // inside it (avoids contaminating the PBRS policy-invariance experiment).
    if (duplicate_count > 0 && !parameters_.PlannerPbrsEnabled_) {
        reward_components["SameGoalPenalty"] = parameters_.PlannerSameGoalPenalty_
            * static_cast<double>(duplicate_count);
    }

    // ─── Legacy dense shaping (skipped when PBRS is on; subsumed by F = γΦ(s')−Φ(s)) ───
    if (!parameters_.PlannerPbrsEnabled_) {
    reward_components["ExtinguishedFires"] = extinguished_fires_ * parameters_.PlannerExtinguishFires_;

    // WaterRefill: dense positive reward for actual tank refilling. Sums fractional
    // per-drone water increases this planner step. Mirrors ExtinguishedFires (rewards
    // outcome, not assignment intent). Non-zero only while a drone is at the
    // groundstation with water < max — no gaming surface.
    // EmptyTank: per-drone penalty for sitting at water=0 while fires remain. Turns
    // the silent "can't extinguish" state into a pushed signal the planner can credit.
    // Both gated on use_water_limit_ so toggling the feature restores prior reward shape.
    if (parameters_.use_water_limit_) {
        if (prev_water_levels_.size() == fly_agents_.size()) {
            const double cap = static_cast<double>(parameters_.GetWaterCapacity());
            if (cap > 0.0) {
                double refill_sum = 0.0;
                for (size_t i = 0; i < fly_agents_.size(); ++i) {
                    double delta = fly_agents_[i]->GetWaterCapacity() - prev_water_levels_[i];
                    if (delta > 0.0) refill_sum += delta / cap;
                }
                if (refill_sum > 0.0) {
                    reward_components["WaterRefill"] = refill_sum * parameters_.PlannerWaterRefill_;
                }
            }
        }
        prev_water_levels_.resize(fly_agents_.size());
        for (size_t i = 0; i < fly_agents_.size(); ++i) {
            prev_water_levels_[i] = fly_agents_[i]->GetWaterCapacity();
        }

        if (!fire_positions->empty()) {
            // Single pass: EmptyTank counts every empty drone (ambient pressure on the
            // empty state); FireGoalEmpty further counts only those with a non-groundstation
            // goal (targets the assignment decision, so a refueling-bound empty drone
            // bleeds only EmptyTank, not the larger FireGoalEmpty).
            int empty_drones = 0;
            int empty_with_fire_goal = 0;
            for (const auto& a : fly_agents_) {
                if (a->GetWaterCapacity() <= 0.0) {
                    ++empty_drones;
                    if (a->GetGoalPosition() != groundstation_pos) {
                        ++empty_with_fire_goal;
                    }
                }
            }
            if (empty_drones > 0) {
                reward_components["EmptyTank"] = empty_drones * parameters_.PlannerEmptyTank_;
            }
            if (empty_with_fire_goal > 0) {
                reward_components["FireGoalEmpty"] = empty_with_fire_goal * parameters_.PlannerFireGoalEmpty_;
            }
        }
    }

    // Spread-prevention: reward per-step reductions in total burning cells. Gives credit
    // for containing fire fronts (what the nearest-neighbor heuristic only rewards at terminal).
    double num_burning = static_cast<double>(grid_map->GetNumBurningCells());
    if (prev_num_burning_ >= 0.0) {
        double delta = prev_num_burning_ - num_burning;  // positive = spread contained
        reward_components["SpreadPrevention"] = delta * parameters_.PlannerSpreadPrevention_;
    }
    prev_num_burning_ = num_burning;

    // GoalCommit: per-drone bonus for keeping the same goal as the previous planner step,
    // when the drone hasn't yet reached it. Counterbalances autoregressive Categorical
    // sampling noise in the pointer decoder — without a sticky signal the policy re-rolls
    // assignments every planner step and drones oscillate between goals without ever
    // completing a trip. The "at goal" guard (within 1 cell) keeps re-evaluation free
    // once the drone has actually arrived, so this rewards commitment, not goal-camping.
    if (prev_planner_goals_.size() == fly_agents_.size()) {
        int committed = 0;
        for (size_t i = 0; i < fly_agents_.size(); ++i) {
            const auto cur = fly_agents_[i]->GetGoalPosition();
            const auto prev = prev_planner_goals_[i];
            const auto pos = fly_agents_[i]->GetGridPositionDouble();
            const double dx = pos.first - cur.first;
            const double dy = pos.second - cur.second;
            const bool at_goal = (dx * dx + dy * dy) < 1.0;
            if (cur == prev && !at_goal) ++committed;
        }
        if (committed > 0) {
            reward_components["GoalCommit"] = committed * parameters_.PlannerGoalCommit_;
        }
    }
    prev_planner_goals_.assign(fly_agents_.size(), {0.0, 0.0});
    for (size_t i = 0; i < fly_agents_.size(); ++i) {
        prev_planner_goals_[i] = fly_agents_[i]->GetGoalPosition();
    }

    // Distance-based progress reward: reward drones for getting closer to their goals,
    // but only when the (water, goal) combination is task-aligned. Without this filter,
    // an empty drone flying toward a fire it cannot extinguish — or a full drone flying
    // toward the groundstation it doesn't need — both earn the planner positive shaping
    // for the FlyAgent's competence on a strategically wasted assignment.
    // Normalize by map scale so the reward weight stays meaningful across map sizes.
    // Per-drone deltas (rather than mean-over-set) so eligible drones joining/leaving
    // the set don't fabricate a spurious progress signal.
    const double norm_map = static_cast<double>(std::max(grid_map->GetRows(), grid_map->GetCols()));
    const double water_max = static_cast<double>(parameters_.GetWaterCapacity());
    if (prev_drone_distances_.size() != fly_agents_.size()) {
        prev_drone_distances_.assign(fly_agents_.size(), -1.0);
    }
    double total_improvement = 0.0;
    int compared_drones = 0;
    for (size_t i = 0; i < fly_agents_.size(); ++i) {
        const auto& agent = fly_agents_[i];
        auto goal = agent->GetGoalPosition();
        const bool goal_is_gs = (goal == groundstation_pos);
        const double water = agent->GetWaterCapacity();
        bool eligible;
        if (goal_is_gs) {
            // Approaching groundstation only counts when there's actual room to refill.
            // Without use_water_limit, refueling is meaningless — never eligible.
            eligible = parameters_.use_water_limit_ && (water < water_max);
        } else {
            // Approaching a fire only counts when the drone can extinguish on arrival.
            // Without use_water_limit, every drone can always extinguish.
            eligible = !parameters_.use_water_limit_ || (water > 0.0);
        }
        if (!eligible) {
            prev_drone_distances_[i] = -1.0;
            continue;
        }
        auto pos = agent->GetGridPositionDouble();
        double dx = goal.first - pos.first;
        double dy = goal.second - pos.second;
        double dist = std::sqrt(dx * dx + dy * dy) / norm_map;
        if (prev_drone_distances_[i] >= 0.0) {
            total_improvement += (prev_drone_distances_[i] - dist);
            ++compared_drones;
        }
        prev_drone_distances_[i] = dist;
    }
    if (compared_drones > 0) {
        const double mean_improvement = total_improvement / static_cast<double>(compared_drones);
        reward_components["DistanceProgress"] = mean_improvement * parameters_.PlannerDistanceProgress_;
    }
    } // !PlannerPbrsEnabled_

    // ─── PBRS shaping: F = γΦ(s_t) − Φ(s_{t-1}). Policy-invariant (Ng+ 1999). ───
    if (parameters_.PlannerPbrsEnabled_) {
        const double phi = ComputePotential(grid_map);
        double F = 0.0;
        if (phi_initialized_) {
            F = parameters_.PlannerPbrsGamma_ * phi - prev_phi_;
        }
        prev_phi_         = phi;
        phi_initialized_  = true;
        reward_components["PBRS"]     = F;
        // Leading-underscore keys are diagnostics — logged to TensorBoard but
        // excluded from ComputeTotalReward (see agent.cpp). The Φ components
        // are already wrapped into F = γΦ(s')−Φ(s); summing them directly into
        // the reward double-counts the potential and breaks Ng+ 1999 invariance.
        reward_components["_PhiFire"]  = phi_fire_last_;
        reward_components["_PhiWater"] = phi_water_last_;
        reward_components["_PhiDist"]  = phi_dist_last_;
    }

    total_reward = ComputeTotalReward(reward_components);
    LogRewards(reward_components);
    reward_components_ = reward_components;
    this->SetReward(total_reward);
    return total_reward;
}

double PlannerAgent::ComputePotential(const std::shared_ptr<GridMap>& grid_map) {
    // Φ_fire(s) = -B(s)/B_init  ∈ [-1, 0]; closer to 0 = better.
    const double phi_fire = -static_cast<double>(grid_map->GetNumBurningCells())
                             / static_cast<double>(std::max(B_init_, 1));

    // Φ_water(s) = mean fractional tank ∈ [0, 1]. Disabled (returns 0) when water_limit
    // is off, so refuel shaping has no effect in that regime — matches existing semantics
    // for WaterRefill/EmptyTank.
    double phi_water = 0.0;
    if (parameters_.use_water_limit_ && !fly_agents_.empty()) {
        const double cap = static_cast<double>(parameters_.GetWaterCapacity());
        if (cap > 0.0) {
            double sum_frac = 0.0;
            for (const auto& a : fly_agents_) {
                sum_frac += std::clamp(a->GetWaterCapacity() / cap, 0.0, 1.0);
            }
            phi_water = sum_frac / static_cast<double>(fly_agents_.size());
        }
    }

    // Φ_dist(s) = -mean(eligible drone→goal distance) ∈ [-1, 0]. Eligibility mirrors the
    // legacy DistanceProgress block: empty drones with fire goals and full drones with
    // groundstation goals are excluded so distance-shaping can't fabricate progress on
    // strategically-misaligned assignments. Distance is map-normalized (already in [0,√2]).
    double phi_dist = 0.0;
    if (!fly_agents_.empty()) {
        const auto groundstation_pos = grid_map->GetGroundstation()->GetGridPositionDouble();
        const double norm_map = static_cast<double>(std::max(grid_map->GetRows(),
                                                             grid_map->GetCols()));
        const double water_max = static_cast<double>(parameters_.GetWaterCapacity());
        double dist_sum = 0.0;
        int eligible_count = 0;
        for (const auto& agent : fly_agents_) {
            const auto goal = agent->GetGoalPosition();
            const bool goal_is_gs = (goal == groundstation_pos);
            const double water = agent->GetWaterCapacity();
            bool eligible;
            if (goal_is_gs) {
                eligible = parameters_.use_water_limit_ && (water < water_max);
            } else {
                eligible = !parameters_.use_water_limit_ || (water > 0.0);
            }
            if (!eligible) continue;
            const auto pos = agent->GetGridPositionDouble();
            const double dx = goal.first - pos.first;
            const double dy = goal.second - pos.second;
            dist_sum += std::sqrt(dx * dx + dy * dy) / norm_map;
            ++eligible_count;
        }
        if (eligible_count > 0) {
            phi_dist = -dist_sum / static_cast<double>(eligible_count);
        }
    }

    phi_fire_last_  = phi_fire;
    phi_water_last_ = phi_water;
    phi_dist_last_  = phi_dist;

    return parameters_.PlannerPbrsWFire_  * phi_fire
         + parameters_.PlannerPbrsWWater_ * phi_water
         + parameters_.PlannerPbrsWDist_  * phi_dist;
}

void PlannerAgent::InitializePlannerAgentStates(const std::shared_ptr<GridMap> &grid_map) {
    // Initialize the agent states with the current grid map
    for (int i = 0; i < time_steps_; ++i) {
        agent_states_.push_front(BuildAgentState(grid_map));
    }
}

std::shared_ptr<AgentState> PlannerAgent::BuildAgentState(const std::shared_ptr<GridMap> &grid_map) {
    auto state = std::make_shared<AgentState>();

    // Map-based normalization: (2*grid_pos/norm_map)-1 maps [0, max_dim] → [-1, 1].
    // Using max(rows, cols) as a single factor preserves spatial aspect ratio.
    // This is the planner's GLOBAL reference frame (unlike the FlyAgent's local view_range).
    double norm_map = static_cast<double>(std::max(grid_map->GetRows(), grid_map->GetCols()));

    std::vector<std::pair<double, double>> drone_positions;
    std::vector<std::pair<double, double>> drone_goals;
    std::vector<double> drone_water;
    const double water_max = static_cast<double>(parameters_.GetWaterCapacity());
    for (const auto &fly_agent : fly_agents_) {
        auto gp = state_features::GridPositionDouble(fly_agent->GetLastState());
        drone_positions.push_back({(2.0 * gp.first / norm_map) - 1.0,
                                   (2.0 * gp.second / norm_map) - 1.0});
        const auto& last = fly_agent->GetLastState();
        drone_goals.push_back({(2.0 * last.goal_position.first / norm_map) - 1.0,
                               (2.0 * last.goal_position.second / norm_map) - 1.0});
        // Normalize by current max — gives scale-invariant "fraction of tank" signal.
        drone_water.push_back(water_max > 0.0
                                  ? std::clamp(fly_agent->GetWaterCapacity() / water_max, 0.0, 1.0)
                                  : 1.0);
    }

    // Centralized Training, Decentralized Execution (CTDE):
    // During training the planner has access to ground-truth fire positions
    // (centralized information), while at eval time it relies only on
    // explored/discovered fires via the fire map (decentralized observation).
    std::shared_ptr<std::vector<std::pair<double, double>>> raw_fires;
    if (eval_mode_) {
        raw_fires = grid_map->GetFirePositionsFromFireMap();
    } else {
        raw_fires = grid_map->GetFirePositionsFromBurningCells();
    }

    // Normalize fire positions to the same map-based scale as drone positions.
    auto normalized_fires = std::make_shared<std::vector<std::pair<double, double>>>();
    normalized_fires->reserve(raw_fires->size());

    // Index 0 is the groundstation — use its real normalized position
    auto gs_pos = grid_map->GetGroundstation()->GetGridPositionDouble();
    normalized_fires->emplace_back(
        (2.0 * gs_pos.first / norm_map) - 1.0,
        (2.0 * gs_pos.second / norm_map) - 1.0
    );

    // Normalize all fire cell positions (skip index 0 which was the dummy)
    for (size_t i = 1; i < raw_fires->size(); ++i) {
        const auto& fp = (*raw_fires)[i];
        normalized_fires->emplace_back(
            (2.0 * fp.first / norm_map) - 1.0,
            (2.0 * fp.second / norm_map) - 1.0
        );
    }

    // Fire count: log1p-normalized so the signal has useful magnitude across map
    // sizes. Raw num_fires/(rows*cols) gave values like 0.008 for a 35×36 map
    // with 10 burning cells — a near-zero input the network can barely read.
    // log1p saturates gracefully as fires multiply and stays in [0, ~1] for any
    // realistic burn up to the map area.
    int num_fires = static_cast<int>(raw_fires->size()) - 1;
    const double log_area = std::log1p(
        static_cast<double>(grid_map->GetRows() * grid_map->GetCols()));
    state->fire_count = std::clamp(
        std::log1p(static_cast<double>(std::max(num_fires, 0))) / log_area,
        0.0, 1.0);

    // Fire centroid: mean of raw fire positions, normalized to match fire_positions
    // space. Use an out-of-range sentinel when no fires are burning so the network
    // can distinguish "empty" from "fires at the map centre" — {0, 0} post-norm is
    // exactly the map centre and ambiguous.
    if (num_fires > 0) {
        double cx = 0.0, cy = 0.0;
        for (size_t i = 1; i < raw_fires->size(); ++i) {
            cx += (*raw_fires)[i].first;
            cy += (*raw_fires)[i].second;
        }
        cx /= num_fires;
        cy /= num_fires;
        state->fire_centroid = {(2.0 * cx / norm_map) - 1.0,
                                (2.0 * cy / norm_map) - 1.0};
    } else {
        state->fire_centroid = {-2.0, -2.0};
    }

    state->fire_positions = normalized_fires;
    state->drone_positions = std::make_shared<std::vector<std::pair<double, double>>>(drone_positions);
    state->goal_positions = std::make_shared<std::vector<std::pair<double, double>>>(drone_goals);
    state->drone_water_levels = std::make_shared<std::vector<double>>(std::move(drone_water));

    // Global wind vector: components normalized to [-1, 1] by a fixed reference speed.
    // kMaxWind = 2x the default config.yaml wind_uw (10.0 m/s); anything beyond is clamped.
    constexpr double kMaxWind = 20.0;
    auto wind = grid_map->GetWind();
    state->wind_vector = {
        std::clamp(wind->getWindSpeedComponent1() / kMaxWind, -1.0, 1.0),
        std::clamp(wind->getWindSpeedComponent2() / kMaxWind, -1.0, 1.0)
    };

    return state;
}

