//
// Created by nex on 21.06.25.
//

#include "planner_agent.h"


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

    // Iterate over all Actions and set a new goal for each FlyAgent
    for (int i = 0; i < fly_agents_.size(); ++i) {
        auto fly_agent = fly_agents_[i];
        std::pair<double, double> goal;
        if (parameters_.eval_fly_policy_ && (fly_agent->GetWaterCapacity() <= 0 || fly_agent->StillCharging())) {
            goal = std::make_pair(-1.0, -1.0);
            fly_agent->CommandRecharge(true);
        } else {
            // Get the goal from the action
            goal = action->GetGoalFromAction(i);
        }

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
    perfect_goals_.clear();
    agent_states_.clear();
    Initialize(explore_agent_, fly_agents_, grid_map);
}

AgentTerminal
PlannerAgent::GetTerminalStates(bool eval_mode, const std::shared_ptr<GridMap> &grid_map, int env_steps_remaining) {
    AgentTerminal t;
    //    int num_burning_cells = grid_map->GetNumBurningCells();
    //    int num_explored_fires = grid_map->GetNumExploredFires();
    explored_fires_equals_actual_fires_ = false; //grid_map->ExploredFiresEqualsActualFires();

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
        if (goal_position == groundstation_pos && !fire_positions->empty()) {
            // If the goal position is set to the groundstation it's probably bad
            reward_components["FlyingTowardsGroundstation"] = parameters_.PlannerFlyingTowardsGroundstation_;
        }
        fly_agent_goals.push_back(goal_position);
    }

    // Check if there are multiple agents going to the same goal
    std::set<std::pair<double, double>> unique_goals(fly_agent_goals.begin(), fly_agent_goals.end());
    if (unique_goals.size() < fly_agent_goals.size()) {
        // There are multiple agents going to the same goal
        reward_components["SameGoalPenalty"] = parameters_.PlannerSameGoalPenalty_ * static_cast<double>(fly_agent_goals.size() - unique_goals.size());
    }

    reward_components["ExtinguishedFires"] = extinguished_fires_ * parameters_.PlannerExtinguishFires_;

    total_reward = ComputeTotalReward(reward_components);
    LogRewards(reward_components);
    reward_components_ = reward_components;
    this->SetReward(total_reward);
    return total_reward;
}

void PlannerAgent::InitializePlannerAgentStates(const std::shared_ptr<GridMap> &grid_map) {
    // Initialize the agent states with the current grid map
    for (int i = 0; i < time_steps_; ++i) {
        agent_states_.push_front(BuildAgentState(grid_map));
    }
}

std::shared_ptr<AgentState> PlannerAgent::BuildAgentState(const std::shared_ptr<GridMap> &grid_map) {
    auto state = std::make_shared<AgentState>();

    std::vector<std::pair<double, double>> drone_positions;
    std::vector<std::pair<double, double>> drone_goals;
    for (const auto &fly_agent : fly_agents_) {
        drone_positions.push_back(fly_agent->GetLastState().GetGridPositionDoubleNorm());
        drone_goals.push_back(fly_agent->GetLastState().GetGoalPositionNorm());
    }

    //if eval get fire positions from explore agent else from gridmap(centralized training)
    if (eval_mode_) {
        auto explore_fire_positions = grid_map->GetFirePositionsFromFireMap();
        state->SetFirePositions(explore_fire_positions);
    } else {
        auto fire_positions = grid_map->GetFirePositionsFromBurningCells();
        state->SetFirePositions(fire_positions);
    }
    state->SetDronePositions(std::make_shared<std::vector<std::pair<double, double>>>(drone_positions));
    state->SetGoalPositions(std::make_shared<std::vector<std::pair<double, double>>>(drone_goals));

    return state;
}

