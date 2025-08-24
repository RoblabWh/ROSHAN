//
// Created by nex on 08.04.25.
//

#include "explore_agent.h"

ExploreAgent::ExploreAgent(FireModelParameters &parameters, int id, int time_steps) : Agent(parameters, 300) {
    id_ = id;
    agent_type_ = "explore_agent";
    time_steps_ = time_steps;
}

void ExploreAgent::Initialize(std::vector<std::shared_ptr<FlyAgent>> fly_agents, const std::shared_ptr<GridMap> &grid_map, const std::string &rl_mode) {
    fly_agents_ = std::move(fly_agents);
    agent_states_.clear();
    auto paths = GeneratePaths(grid_map->GetRows(), grid_map->GetCols(), fly_agents_.size(), fly_agents_.front()->GetRealPosition(),
                                 parameters_.explore_agent_view_range_);
    for (const auto& fly_agent : fly_agents_) {
        perfect_goals_.emplace_back(std::move(paths[fly_agent->GetId()]));
    }

    for (const auto& fly_agent : fly_agents_) {
        auto goal = perfect_goals_[fly_agent->GetId()][goal_idx_];
        std::pair<double, double> local_goal = std::make_pair(goal.first, goal.second);
        fly_agent->SetGoalPosition({local_goal.first, local_goal.second});
        fly_agent->SetRevisitedCells(revisited_cells_);
    }

    goal_idx_ < perfect_goals_[0].size() - 1 ? goal_idx_++ : goal_idx_ = 0;

    InitializeExploreAgentStates(grid_map);
}

std::pair<double, double>
ExploreAgent::GetGoalFromAction(const ExploreAction *action, const std::shared_ptr<GridMap> &grid_map) {
    double x_off = grid_map->GetXOff();
    double y_off = grid_map->GetYOff();

    auto goal_x = action->GetGoalX();
    auto goal_y = action->GetGoalY();

    // Nudge goal right at the edges
    goal_x = goal_x <= 0 ? (goal_x + x_off) : goal_x >= 1 ? (goal_x - x_off) : goal_x;
    goal_y = goal_y <= 0 ? (goal_y + y_off) : goal_y >= 1 ? (goal_y - y_off) : goal_y;

    // Calc the grid position
    goal_x = ((goal_x + 1) / 2) * grid_map->GetRows();
    goal_y = ((goal_y + 1) / 2) * grid_map->GetCols();

    // Nudge goal so the view range is in the grid PROBABLY DON'T DO THIS
//    int view_range_off = FireModelParameters::GetViewRange("fly_agent") / 2;
//    int view_off_x = goal_x <= static_cast<double>(gridMap->GetRows()) / 2 ? view_range_off - 1 : -view_range_off + 1;
//    int view_off_y = goal_y <= static_cast<double>(gridMap->GetCols()) / 2 ? view_range_off - 1: -view_range_off + 1;
//    goal_x += view_off_x;
//    goal_y += view_off_y;

//    std::cout << "ExploreAgent goal position (normalized): (" << action->GetGoalX() << ", " << action->GetGoalY() << ")" << std::endl;
//    std::cout << "ExploreAgent goal position: (" << goal_x << ", " << goal_y << ")" << std::endl;

    return std::make_pair(goal_x, goal_y);
}

std::pair<double, double> ExploreAgent::GetGoalFromCertain(std::deque<std::pair<double, double>> &goals, const std::shared_ptr<GridMap>& gridMap) {
    if (goals.empty()) {
        // If there are no more perfect goals, return the groundstation position
        return gridMap->GetGroundstation()->GetGridPositionDouble();
    }
    std::pair<double, double> goal = goals[goal_idx_];
    return goal;
}


void ExploreAgent::PerformExplore(ExploreAction *action, const std::string& hierarchy_type, const std::shared_ptr<GridMap>& gridMap) {
    // These values are used to calculate the reward, GetRevisitedCells must be called before anything else
    // because it is used to populate the ExploreMap with the input of the last step
    revisited_cells_ = gridMap->GetRevisitedCells();

//    auto goal = GetGoalFromAction(action, gridMap);

    //TODO Make it so that the network puts out multiple goals for multiple agents
    for (const auto& fly_agent : fly_agents_) {
        //auto goal = GetGoalFromCertain(perfect_goals_[fly_agent->GetId()], gridMap);
        auto goal = perfect_goals_[fly_agent->GetId()][goal_idx_];
        std::pair<double, double> local_goal = std::make_pair(goal.first, goal.second);
//        auto local_goal = fly_agent->CalculateLocalGoal(goal_x, goal_y);
        fly_agent->SetGoalPosition({local_goal.first, local_goal.second});
        fly_agent->SetRevisitedCells(revisited_cells_);
//        fly_agent->SetGoalPosition({goal_x, goal_y});
    }
    goal_idx_ < perfect_goals_[0].size() - 1 ? goal_idx_++ : goal_idx_ = 0;
    if(hierarchy_type == "explore_agent") {
        did_hierarchy_step = true;
    }
    this->UpdateStates(gridMap);
    // Pick first agent to get the exploration map scalar
    auto explore_map_scalar = GetLastState().GetExplorationMapScalar();
    if (explore_map_scalar >= 0.99) {
        objective_reached_ = true;
    }
}

std::shared_ptr<AgentState> ExploreAgent::BuildAgentState(const std::shared_ptr<GridMap>& grid_map) {
    auto state = std::make_shared<AgentState>();

    std::vector<std::shared_ptr<const std::vector<std::vector<double>>>> views;
    for (const auto& agent : fly_agents_) {
        const auto& latest_state = agent->GetLastState();
        auto orig_ptr = latest_state.GetTotalDroneViewPtr();
        auto copy_ptr = std::make_shared<const std::vector<std::vector<double>>>(*orig_ptr);
        views.push_back(copy_ptr);
        // TODO Rewrite this; this is a hack to get other states from the drone
        state->SetPosition(latest_state.GetPosition());
        state->SetCellSize(latest_state.GetCellSize());
    }
    state->SetMapDimensions({grid_map->GetRows(), grid_map->GetCols()});
    state->SetMultipleTotalDroneView(views);
    state->SetExplorationMap(grid_map->GetExploredMap());
//    state->SetExploredFires(grid_map->GetExploredFires());
    state->SetPerfectGoals(perfect_goals_);

    return state;
}

void ExploreAgent::InitializeExploreAgentStates(const std::shared_ptr<GridMap>& grid_map) {
    for(int i = 0; i < time_steps_; ++i) {
        agent_states_.push_front(BuildAgentState(grid_map));
    }
}

void ExploreAgent::UpdateStates(const std::shared_ptr<GridMap>& grid_map) {
    agent_states_.push_front(BuildAgentState(grid_map));

    // Maximum number of states i.e. memory
    if (agent_states_.size() > time_steps_) {
        agent_states_.pop_back();
    }
}

double ExploreAgent::CalculateReward() {
    int newlyExploredCells = 0;
    double alpha = 0.0001;
    double beta = -0.00001;

    for (const auto& fly_agent : fly_agents_) {
        newlyExploredCells += fly_agent->GetAndResetNewlyExploredCells();
    }

    std::unordered_map<std::string, double> reward_components;

    reward_components["NewlyExploredCells"] = alpha * static_cast<double>(newlyExploredCells);
    reward_components["RevisitedCellsPenalty"] = beta * static_cast<double>(revisited_cells_);

    if (objective_reached_ && agent_terminal_state_){
        reward_components["MapExplored"] = 10;
    } else if (agent_terminal_state_) {
        reward_components["Failure"] = -1;
    }

    double total_reward = ExploreAgent::ComputeTotalReward(reward_components);
    this->LogRewards(reward_components);
    reward_components_ = reward_components;
    return total_reward;
}

AgentTerminal ExploreAgent::GetTerminalStates(bool eval_mode, const std::shared_ptr<GridMap>& grid_map, int total_env_steps) {
    std::vector<bool> terminal_states;
    AgentTerminal t;
//    int num_burning_cells = grid_map->GetNumBurningCells();
//    int num_explored_fires = grid_map->GetNumExploredFires();
    explored_fires_equals_actual_fires_ = false; //grid_map->ExploredFiresEqualsActualFires();

    // If the agent has taken too long it has reached a terminal state and died
    // TODO currently hard-coded to false, this is because the agent is currently not network controlled
    if (false) {
        if (total_env_steps <= 0 && !eval_mode) {
            t.is_terminal = true;
            t.reason = FailureReason::Timeout;
        }

        if (objective_reached_) {
            t.is_terminal = true;
        }
        if (t.is_terminal && t.reason != FailureReason::None) { t.kind = TerminationKind::Failed; }
        else if (t.is_terminal) { t.kind = TerminationKind::Succeeded; }
        else { t.kind = TerminationKind::None; }
    }

    return t;
}