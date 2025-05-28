//
// Created by nex on 08.04.25.
//

#include "explore_agent.h"

ExploreAgent::ExploreAgent(FireModelParameters &parameters, int id, int time_steps) : Agent(parameters, 300) {
    id_ = id;
    agent_type_ = "ExploreAgent";
    time_steps_ = time_steps;
}

void ExploreAgent::Initialize(std::vector<std::shared_ptr<FlyAgent>> fly_agents, const std::shared_ptr<GridMap> &grid_map, const std::string &rl_mode) {
    fly_agents_ = std::move(fly_agents);
    agent_states_.clear();
    InitializeExploreAgentStates(grid_map);
    for (const auto& fly_agent : fly_agents_) {
        fly_agent->SetGoalPosition(std::make_pair(grid_map->GetRows() / 2, grid_map->GetCols() / 2));
    }
}

void ExploreAgent::PerformExplore(ExploreAction *action, const std::string& hierarchy_type, const std::shared_ptr<GridMap>& gridMap) {
    // These values are used to calculate the reward, GetRevisitedCells must be called before anything else
    // because it is used to populate the ExploreMap with the input of the last step
    revisited_cells_ = gridMap->GetRevisitedCells();

    double x_off = gridMap->GetXOff();
    double y_off = gridMap->GetYOff();

    auto goal_x = action->GetGoalX();
    auto goal_y = action->GetGoalY();

    // Nudge goal right at the edges
    goal_x = goal_x <= 0 ? (goal_x + x_off) : goal_x >= 1 ? (goal_x - x_off) : goal_x;
    goal_y = goal_y <= 0 ? (goal_y + y_off) : goal_y >= 1 ? (goal_y - y_off) : goal_y;

    // Calc the grid position
    goal_x = ((goal_x + 1) / 2) * gridMap->GetRows();
    goal_y = ((goal_y + 1) / 2) * gridMap->GetCols();

    // Nudge goal so the view range is in the grid PROBABLY DON'T DO THIS
//    int view_range_off = FireModelParameters::GetViewRange("FlyAgent") / 2;
//    int view_off_x = goal_x <= static_cast<double>(gridMap->GetRows()) / 2 ? view_range_off - 1 : -view_range_off + 1;
//    int view_off_y = goal_y <= static_cast<double>(gridMap->GetCols()) / 2 ? view_range_off - 1: -view_range_off + 1;
//    goal_x += view_off_x;
//    goal_y += view_off_y;

    //TODO Make it so that the network puts out multiple goals for multiple agents
    for (const auto& fly_agent : fly_agents_) {
        auto local_goal = fly_agent->CalculateLocalGoal(goal_x, goal_y);
        fly_agent->SetGoalPosition({local_goal.first, local_goal.second});
        fly_agent->SetRevisitedCells(revisited_cells_);
//        fly_agent->SetGoalPosition({goal_x, goal_y});
    }
    if(hierarchy_type == "ExploreAgent") {
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
        views.push_back(latest_state.GetTotalDroneViewPtr());
    }
    state->SetMultipleTotalDroneView(views);
    state->SetExplorationMap(grid_map->GetExploredMap());

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

std::vector<bool> ExploreAgent::GetTerminalStates(bool eval_mode, const std::shared_ptr<GridMap>& grid_map, int total_env_steps) {
    std::vector<bool> terminal_states;
    bool terminal_state = false;
    bool drone_died = false;
    bool drone_succeeded = false;
//    int num_burning_cells = grid_map->GetNumBurningCells();
//    int num_explored_fires = grid_map->GetNumExploredFires();
    explored_fires_equals_actual_fires_ = false; //grid_map->ExploredFiresEqualsActualFires();

    // If the agent has taken too long it has reached a terminal state and died
    if (total_env_steps <= 0 && !eval_mode) {
        terminal_state = true;
        drone_died = true;
    }

    if (objective_reached_) {
        terminal_state = true;
        drone_succeeded = true;
    }

    // If the agent has flown out of the grid it has reached a terminal state and died
//    if (GetOutOfAreaCounter() > 1) {
//        terminal_state = true;
//        drone_died = true;
//    }
//
//    // If the drone has reached the goal and it is not in evaluation mode
//    // the goal is reached because the fly agent is trained that way
//    auto explore_map = GetLastState().GetExplorationMapScalar();
//    if (explore_map >= 0.99 || explored_fires_equals_actual_fires_) {
//        terminal_state = true;
//    }

    terminal_states.push_back(terminal_state);
    terminal_states.push_back(drone_died);
    terminal_states.push_back(drone_succeeded);
    return terminal_states;
}
