//
// Created by nex on 08.04.25.
//

#include "fly_agent.h"

FlyAgent::FlyAgent(const std::shared_ptr<GridMap>& grid_map, FireModelParameters &parameters, int id, int timesteps) :
        id_(id),
        parameters_(parameters),
        rewards_(parameters_.GetRewardsBufferSize()) {

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double rng_number = dist(gen);

    std::pair<int, int> point;
    agent_type_ = "FlyAgent";
    if (agent_type_ == "FlyAgent") {
        if (rng_number <= parameters_.corner_start_percentage_) {
            point = grid_map->GetNonGroundStationCorner();
        }
        else if (rng_number <= parameters_.groundstation_start_percentage_) {
            point = grid_map->GetGroundstation()->GetGridPosition();
        }
        else {
            point = grid_map->GetRandomPointInGrid();
        }
    } else {
        int view_range_off = view_range_ / 2;
        point = grid_map->GetGroundstation()->GetGridPosition();
        int off_x = point.first <= grid_map->GetRows() / 2 ? view_range_off - 1 : -view_range_off + 1;
        int off_y = point.second <= grid_map->GetCols() / 2 ? view_range_off - 1: -view_range_off + 1;
        point.first += off_x;
        point.second += off_y;
    }

    double x = (point.first + 0.5); // position in grid + offset to center
    double y = (point.second + 0.5); // position in grid + offset to center
    position_ = std::make_pair(x * parameters_.GetCellSize(), y * parameters_.GetCellSize());
    goal_position_ = std::make_pair(0, 0);
    view_range_ = parameters_.GetViewRange();
    time_steps_ = timesteps;
    water_capacity_ = parameters_.GetWaterCapacity();
    out_of_area_counter_ = 0;
    is_alive_ = true;
}

void FlyAgent::Initialize(int mode, const std::shared_ptr<GridMap>& grid_map, const std::shared_ptr<FireModelRenderer>& model_renderer, const std::string& rl_mode, double rng_number) {
    if (mode == Mode::GUI_RL) {
        this->SetRenderer(model_renderer->GetRenderer());
        this->SetRenderer2(model_renderer->GetRenderer());
    }
    this->SetExploreDifference(0);
    int difference = grid_map->UpdateExploredAreaFromDrone(this->GetGridPosition(), this->GetViewRange());
    this->SetExploreDifference(this->GetExploreDifference() + difference);

    // Generate random number between 0 and 1
    std::pair<double, double> goal_pos = std::pair<double, double>(-1, -1);
    if (rng_number < parameters_.fire_goal_percentage_ || rl_mode == "eval") {
        goal_pos = grid_map->GetNextFire(this->GetGridPosition());
    }
    if (std::pair<double, double>(-1, -1) == goal_pos) {
        goal_pos = grid_map->GetGroundstation()->GetGridPositionDouble();
    }

    this->SetGoalPosition(goal_pos);
    this->InitializeGridMap(grid_map);
    this->SetLastDistanceToGoal(this->GetDistanceToGoal());
    this->SetLastNearFires(this->DroneSeesFire());
    this->SetLastDistanceToFire(this->FindNearestFireDistance());
}

void FlyAgent::InitializeGridMap(const std::shared_ptr<GridMap>& grid_map) {
    map_dimensions_ = std::make_pair(grid_map->GetRows(), grid_map->GetCols());
    auto drone_view = grid_map->GetDroneView(this->GetGridPosition(), this->GetViewRange());
    auto total_drone_view = grid_map->GetInterpolatedDroneView(this->GetGridPosition(), this->GetViewRange());
    auto explored_map = grid_map->GetExploredMap();
    auto fire_map = grid_map->GetFireMap();
    dispensed_water_ = false;
    for(int i = 0; i < time_steps_; ++i) {
        DroneState new_state = DroneState(std::pair<int,int>(0,0),
                                          parameters_.GetMaxVelocity(), drone_view, total_drone_view,
                                          explored_map, fire_map, map_dimensions_, position_,
                                          goal_position_, 0, parameters_.GetCellSize());
        drone_states_.push_front(new_state);
    }
}

void FlyAgent::PerformFly(FlyAction* action, const std::string& hierarchy_type, const std::shared_ptr<GridMap>& gridMap) {
    // Get the speed and water dispense from the action
    double speed_x = action->GetSpeedX(); // change this to "real" speed
    double speed_y = action->GetSpeedY();
    this->Step(speed_x, speed_y, gridMap);
    if (hierarchy_type == "FlyAgent") {
        this->FlyPolicy(gridMap);
        this->SetPerformedHierarchyAction(true);
    } else {
        this->SetReachedGoal(false);
        if (this->GetGoalPositionInt() == this->GetGridPosition()) {
            this->SetReachedGoal(true);
        }
    }
}

double FlyAgent::CalculateReward() {
    double distance_to_goal = GetDistanceToGoal();
    double last_distance_to_goal = GetLastDistanceToGoal();
    double distance_to_boundary = GetLastState().GetDistanceToNearestBoundaryNorm();
    double delta_distance = last_distance_to_goal - distance_to_goal;
    bool drone_in_grid = GetDroneInGrid();

    std::unordered_map<std::string, double> reward_components;
    double total_reward = 0;

    if (GetReachedGoal()) {
        reward_components["GoalReached"] = 5;
    }

    if (!drone_in_grid && last_terminal_state_) {
        reward_components["BoundaryTerminal"] = -2;
    }

    if (last_terminal_state_ && (last_total_env_steps_ <= 0)) {
        reward_components["TimeOut"] = -2;
    }

    if (delta_distance > 0) {
//        auto safety_factor = std::tanh(10 * distance_to_boundary);
        reward_components["DistanceImprovement"] = 0.2 * delta_distance; // * safety_factor;
    }

//    if (distance_to_boundary < 0.125) {
//        reward_components["Boundary"] = -0.5 * distance_to_boundary;
//    }

    total_reward = ComputeTotalReward(reward_components);
    LogRewards(reward_components);
    SetLastDistanceToGoal(distance_to_goal);
    reward_components_ = reward_components;
    this->SetReward(total_reward);
    return total_reward;
}

std::deque<std::shared_ptr<State>> FlyAgent::GetObservations() {
    std::deque<std::shared_ptr<State>> states;
    states.resize(drone_states_.size());

    for (size_t i = 0; i < drone_states_.size(); ++i) {
        states[i] = std::make_shared<DroneState>(drone_states_[i]);
    }

    return states;
}

//TODO TIDY UP !!!!

void FlyAgent::Render(std::pair<int, int> position, std::pair<int, int> goal_position_screen, int size) {
    renderer_.Render(position, size, view_range_, 0, active_);
    renderer2_.RenderGoal(goal_position_screen, size);
}

std::pair<double, double> FlyAgent::MoveByXYVel(double netout_speed_x, double netout_speed_y) {
    std::pair<double, double> velocity_vector = drone_states_[0].GetNewVelocity(netout_speed_x, netout_speed_y);
    auto adjusted_vel_vector = std::make_pair(velocity_vector.first * parameters_.GetDt(), velocity_vector.second * parameters_.GetDt());
    position_.first += adjusted_vel_vector.first;
    position_.second += adjusted_vel_vector.second;
    return adjusted_vel_vector;
}

std::pair<double, double> FlyAgent::MovementStep(double netout_x, double netout_y) {
    return this->MoveByXYVel(netout_x, netout_y);
}

void FlyAgent::UpdateStates(const std::shared_ptr<GridMap>& grid_map, std::pair<double, double> velocity_vector, int water_dispense) {
    // Update the states, last state get's kicked out
    auto explored_map = grid_map->GetExploredMap();
    auto fire_map = grid_map->GetFireMap();
    auto drone_view = grid_map->GetDroneView(this->GetGridPosition(), this->GetViewRange());
    auto total_drone_view = grid_map->GetInterpolatedDroneView(this->GetGridPosition(), this->GetViewRange());

    DroneState new_state = DroneState(velocity_vector,
                                      parameters_.GetMaxVelocity(),
                                      drone_view,
                                      total_drone_view,
                                      explored_map,
                                      fire_map,
                                      map_dimensions_,
                                      position_,
                                      goal_position_,
                                      water_dispense,
                                      parameters_.GetCellSize()
    );
    drone_states_.push_front(new_state);

    // Maximum number of states i.e. memory
    if (drone_states_.size() > time_steps_) {
        drone_states_.pop_back();
    }

    std::pair<int, int> drone_position = GetGridPosition();
    drone_in_grid_ = grid_map->IsPointInGrid(drone_position.first, drone_position.second);
    // Calculates if the Drone is in the grid and if not how far it is away from the grid
    CalcMaxDistanceFromMap();
}

bool FlyAgent::DispenseWaterCertain(const std::shared_ptr<GridMap>& grid_map) {
    std::pair<int, int> grid_position = GetGridPosition();
    bool cell_is_burning = false;
    if (grid_map->IsPointInGrid(grid_position.first, grid_position.second)){
        cell_is_burning = grid_map->At(grid_position.first, grid_position.second).IsBurning();
    }
    if (cell_is_burning) {
        dispensed_water_ = true;
        water_capacity_ -= 1;
        bool fire_extinguished = grid_map->WaterDispension(grid_position.first, grid_position.second);
        if (fire_extinguished) {
            if (grid_map->GetNumBurningCells() == 0) {
                extinguished_last_fire_ = true;
            }
        }
        extinguished_fire_ = fire_extinguished;
        return fire_extinguished;
    } else {
        dispensed_water_ = false;
        extinguished_fire_ = false;
        return false;
    }
}

void FlyAgent::DispenseWater(const std::shared_ptr<GridMap>& grid_map, int water_dispense) {
    // Returns true if fire was extinguished
    if (water_dispense == 1) {
        dispensed_water_ = true;
        std::pair<int, int> grid_position = GetGridPosition();
        bool fire_extinguished = grid_map->WaterDispension(grid_position.first, grid_position.second);
        if (fire_extinguished) {
            if (grid_map->GetNumBurningCells() == 0) {
                extinguished_last_fire_ = true;
            }
        }
        extinguished_fire_ = fire_extinguished;
    } else {
        dispensed_water_ = false;
        extinguished_fire_ = false;
    }
}

std::pair<double, double> FlyAgent::GetRealPosition() {
    return position_;
}

std::pair<int, int> FlyAgent::GetGridPosition() {
    int x, y;
    parameters_.ConvertRealToGridCoordinates(position_.first, position_.second, x, y);
    return std::make_pair(x, y);
}

std::pair<double, double> FlyAgent::GetGridPositionDouble() {
//    double x, y;
//    x = position_.first / parameters_.GetCellSize();
//    y = position_.second / parameters_.GetCellSize();
//    return std::make_pair(x, y);
    return this->GetLastState().GetGridPositionDouble();
}

// Checks whether the drone sees fire in the current fire status and return how much
int FlyAgent::DroneSeesFire() {
    std::vector<std::vector<int>> fire_status = GetLastState().GetFireView();
    int count = std::accumulate(fire_status.begin(), fire_status.end(), 0,
                                [](int acc, const std::vector<int>& vec) {
                                    return acc + std::count(vec.begin(), vec.end(), 1);
                                }
    );
    return count;
}

double FlyAgent::FindNearestFireDistance() {
    std::pair<int, int> drone_grid_position = GetGridPosition();
    double min_distance = std::numeric_limits<double>::max();
    std::vector<std::vector<int>> fire_status = GetLastState().GetFireView();

    for (int y = 0; y <= view_range_; ++y) {
        for (int x = 0; x <= view_range_; ++x) {
            if (fire_status[x][y] == 1) { // Assuming 1 indicates fire
                std::pair<int, int> fire_grid_position = std::make_pair(
                        drone_grid_position.first + y - (view_range_ / 2),
                        drone_grid_position.second + x - (view_range_ / 2)
                );

                double real_x, real_y;
                parameters_.ConvertGridToRealCoordinates(fire_grid_position.first, fire_grid_position.second, real_x, real_y);
                double distance = sqrt(
                        pow(real_x - position_.first, 2) +
                        pow(real_y - position_.second, 2)
                );

                if (distance < min_distance) {
                    min_distance = distance;
                }
            }
        }
    }

    return min_distance;
}

void FlyAgent::CalcMaxDistanceFromMap() {
    max_distance_from_map_ = 0;
    if (!drone_in_grid_) {
        IncrementOutOfAreaCounter();
        std::pair<double, double> pos = GetLastState().GetGridPositionDoubleNorm();
        double max_distance1 = 0;
        double max_distance2 = 0;
        if (pos.first < 0 || pos.second < 0) {
            max_distance1 = abs(std::min(pos.first, pos.second));
        } else if (pos.first > 1 || pos.second > 1) {
            max_distance2 = std::max(pos.first, pos.second) - 1;
        }
        max_distance_from_map_ = std::max(max_distance1, max_distance2);
    } else {
        ResetOutOfAreaCounter();
    }
}

double FlyAgent::GetDistanceToGoal() {
    return sqrt(pow(this->GetGoalPosition().first - this->GetGridPositionDouble().first, 2) +
                pow(this->GetGoalPosition().second - this->GetGridPositionDouble().second, 2)
    );
}

//void FlyAgent::OnExploreAction(std::shared_ptr<ExploreAction> action, std::shared_ptr<GridMap> gridMap) {
//    // Get the goal position from the action
//    double x_off = gridMap->GetXOff();
//    double y_off = gridMap->GetYOff();
//    double goal_x = action->GetGoalX();
//    double goal_y = action->GetGoalY();
//    // Nudge goal right at the edges
//    goal_x = goal_x <= 0 ? (goal_x + x_off) : goal_x >= 1 ? (goal_x - x_off) : goal_x;
//    goal_y = goal_y <= 0 ? (goal_y + y_off) : goal_y >= 1 ? (goal_y - y_off) : goal_y;
//    // Calc the grid position
//    goal_x = ((goal_x + 1) / 2) * gridMap->GetRows();
//    goal_y = ((goal_y + 1) / 2) * gridMap->GetCols();
//    // Nudge goal so the view range is in the grid
//    int view_range_off = view_range_ / 2;
//    int view_off_x = goal_x <= static_cast<double>(gridMap->GetRows()) / 2 ? view_range_off - 1 : -view_range_off + 1;
//    int view_off_y = goal_y <= static_cast<double>(gridMap->GetCols()) / 2 ? view_range_off - 1: -view_range_off + 1;
//    goal_x += view_off_x;
//    goal_y += view_off_y;
//    this->SetGoalPosition(std::make_pair(goal_x, goal_y));
//    if (agent_type_ == "ExploreAgent") {
//        this->SetDroneDidHierachyAction(true);
//    }
//}

void FlyAgent::Step(double speed_x, double speed_y, const std::shared_ptr<GridMap>& gridmap) {
    std::pair<double, double> vel_vector = this->MovementStep(speed_x, speed_y);
    int difference = gridmap->UpdateExploredAreaFromDrone(this->GetGridPosition(), this->GetViewRange());
    this->SetExploreDifference(this->GetExploreDifference() + difference);
    this->UpdateStates(gridmap, vel_vector, 0);
}

void FlyAgent::FlyPolicy(const std::shared_ptr<GridMap>& gridmap){
    this->SetReachedGoal(false);
    if(this->GetPolicyType() == 0) {
        if (this->GetGoalPositionInt() == this->GetGridPosition()) {
            this->SetReachedGoal(true);
            this->DispenseWaterCertain(gridmap);
            if (this->GetWaterCapacity() <= 0) {
                auto groundstation_position = gridmap->GetGroundstation()->GetGridPositionDouble();
                this->SetGoalPosition(groundstation_position);
                this->SetPolicyType(1);
            } else {
                auto next_fire = gridmap->GetNextFire(this->GetGridPosition());
                this->SetGoalPosition(next_fire);
            }
        }
    } else if (this->GetPolicyType() == 1) {
        if (this->GetGoalPositionInt() == this->GetGridPosition()) {
            this->SetPolicyType(2);
        }
    } else if (this->GetPolicyType() == 2) {
        if (parameters_.recharge_time_active_) {
            if (this->GetWaterCapacity() <= parameters_.GetWaterCapacity()) {
                this->SetWaterCapacity(this->GetWaterCapacity() + parameters_.GetWaterRefillDt());
            } else {
                this->SetPolicyType(0);
                this->SetGoalPosition(gridmap->GetNextFire(this->GetGridPosition()));
            }
        } else {
            this->SetWaterCapacity(parameters_.GetWaterCapacity());
            this->SetPolicyType(0);
            this->SetGoalPosition(gridmap->GetNextFire(this->GetGridPosition()));
        }
    }
}

std::vector<bool> FlyAgent::GetTerminalStates(bool eval_mode, const std::shared_ptr<GridMap>& grid_map, int total_env_steps) {
    std::vector<bool> terminal_states;
    bool terminal_state = false;
    bool drone_died = false;
    bool drone_succeeded = false;

    // If the agent has flown out of the grid it has reached a terminal state and died
    if (GetOutOfAreaCounter() > 1) {
        terminal_state = true;
        drone_died = true;
    }
    // If the drone has reached the goal and it is not in evaluation mode
    // the goal is reached because the fly agent is trained that way
    if (GetReachedGoal()) {
        if (!eval_mode){
            terminal_state = true;
        }
        drone_succeeded = true;
    }
    // If the agent has taken too long it has reached a terminal state and died
    if (total_env_steps <= 0 && !eval_mode) {
        terminal_state = true;
        drone_died = true;
    }

    // TODO CHANGE LATER
    // Terminals only for evaluation
    if (eval_mode){
        if (grid_map->PercentageBurned() > 0.30) {
            terminal_state = true;
            drone_died = true;
        }
        if (GetExtinguishedLastFire()) {
            // TODO Don't use gridmap_->IsBurning() because it is not reliable since it returns false when there
            //  are particles in the air. Instead, check if the drone has extinguished the last fire on the map.
            //  This also makes sure that only the drone that actually extinguished the fire gets the reward
            terminal_state = true;
        }
        if (!grid_map->IsBurning()) {
            terminal_state = true;
        }
    }

    terminal_states.push_back(terminal_state);
    terminal_states.push_back(drone_died);
    terminal_states.push_back(drone_succeeded);
    last_terminal_state_ = terminal_state;
    last_total_env_steps_ = total_env_steps;
    return terminal_states;
}

std::vector<bool> FlyAgent::TerminalExplore(bool eval_mode, const std::shared_ptr<GridMap> &grid_map, int total_env_steps) {
    std::vector<bool> terminal_states;
    bool terminal_state = false;
    bool drone_died = false;
    bool drone_succeeded = false;
//    int num_burning_cells = grid_map->GetNumBurningCells();
//    int num_explored_fires = grid_map->GetNumExploredFires();
    explored_fires_equals_actual_fires_ = false; //grid_map->ExploredFiresEqualsActualFires();

    if (GetReachedGoal()) {
        drone_succeeded = true;
    }

    // If the agent has flown out of the grid it has reached a terminal state and died
    if (GetOutOfAreaCounter() > 1) {
        terminal_state = true;
        drone_died = true;
    }

    // If the drone has reached the goal and it is not in evaluation mode
    // the goal is reached because the fly agent is trained that way
    auto explore_map = GetLastState().GetExplorationMapScalar();
    if (explore_map >= 0.99 || explored_fires_equals_actual_fires_) {
        terminal_state = true;
    }

    // If the agent has taken too long it has reached a terminal state and died
    if (total_env_steps <= 0 && !eval_mode) {
        terminal_state = true;
        drone_died = true;
    }

    terminal_states.push_back(terminal_state);
    terminal_states.push_back(drone_died);
    terminal_states.push_back(drone_succeeded);
    return terminal_states;
}

double FlyAgent::CalculateExploreReward(bool terminal_state, int total_env_steps){
    auto explore_map = GetLastState().GetExplorationMapScalar();
    auto explore_difference = GetExploreDifference();

    std::unordered_map<std::string, double> reward_components;

    if (explore_map >= 0.99 && terminal_state){
        reward_components["MapExplored"] = 5;
    } else if (explored_fires_equals_actual_fires_ && terminal_state) {
        reward_components["ExploredFires"] = 5;
    } else if (terminal_state) {
        reward_components["Failure"] = -10;
    }
//    reward_components["ExploreMapScalar"] = 0.01 * explore_map;
    reward_components["ExploreDifference"] = 0.001 * explore_difference;

    double total_reward = ComputeTotalReward(reward_components);
    LogRewards(reward_components);
    reward_components_ = reward_components;
    return total_reward;
}

double FlyAgent::ComputeTotalReward(const std::unordered_map<std::string, double>& rewards) {
    double total = 0;
    for (const auto& [key, value] : rewards) {
        total += value;
    }
    return total;
}

void FlyAgent::LogRewards(const std::unordered_map<std::string, double>& rewards) {
#ifdef DEBUG_REWARD_YES
    for (const auto& [key, value] : rewards) {
            std::cout << key << " Reward: " << value << "\n";
        }
        std::cout << "\n";
#endif
}


