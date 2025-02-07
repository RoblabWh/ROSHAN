//
// Created by nex on 13.07.23.
//

#include <iostream>
#include <utility>
#include "drone.h"

DroneAgent::DroneAgent(const std::shared_ptr<GridMap>& grid_map, std::string agent_type, FireModelParameters &parameters, int id) : id_(id), parameters_(parameters),
                                                                                             rewards_(parameters_.GetRewardsBufferSize()) {

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double rng_number = dist(gen);

    std::pair<int, int> point;
    if (rng_number <= parameters_.corner_start_percentage_) {
        point = grid_map->GetNonGroundStationCorner();
    } else
    if (rng_number <= parameters_.groundstation_start_percentage_) {
        point = grid_map->GetGroundstation()->GetGridPosition();
    }
    else {
        point = grid_map->GetRandomPointInGrid();
    }

    agent_type_ = std::move(agent_type);
    double x = (point.first + 0.5); // position in grid + offset to center
    double y = (point.second + 0.5); // position in grid + offset to center
    position_ = std::make_pair(x * parameters_.GetCellSize(), y * parameters_.GetCellSize());
    goal_position_ = std::make_pair(0, 0);
    view_range_ = parameters_.GetViewRange();
    time_steps_ = parameters_.GetTimeSteps();
    water_capacity_ = parameters_.GetWaterCapacity();
    out_of_area_counter_ = 0;
    is_alive_ = true;
}

void DroneAgent::Initialize(const std::shared_ptr<GridMap>& grid_map) {
    map_dimensions_ = std::make_pair(grid_map->GetRows(), grid_map->GetCols());
    auto drone_view = grid_map->GetDroneView(std::dynamic_pointer_cast<DroneAgent>(shared_from_this()));
    auto explored_map = grid_map->GetExploredMap();
    auto fire_map = grid_map->GetFireMap();
    dispensed_water_ = false;
    for(int i = 0; i < time_steps_; ++i) {
        DroneState new_state = DroneState(std::pair<int,int>(0,0),
                parameters_.GetMaxVelocity(), drone_view,
                explored_map, fire_map, map_dimensions_, position_,
                goal_position_, 0, parameters_.GetCellSize());
        drone_states_.push_front(new_state);
    }
}

void DroneAgent::Render(std::pair<int, int> position, std::pair<int, int> goal_position_screen, int size) {
    renderer_.Render(position, size, view_range_, 0, active_);
    renderer2_.RenderGoal(goal_position_screen, size);
}

std::pair<double, double> DroneAgent::MoveByXYVel(double netout_speed_x, double netout_speed_y) {
    std::pair<double, double> velocity_vector = drone_states_[0].GetNewVelocity(netout_speed_x, netout_speed_y);
    auto adjusted_vel_vector = std::make_pair(velocity_vector.first * parameters_.GetDt(), velocity_vector.second * parameters_.GetDt());
    position_.first += adjusted_vel_vector.first;
    position_.second += adjusted_vel_vector.second;
    return adjusted_vel_vector;
}

std::pair<double, double> DroneAgent::MovementStep(double netout_x, double netout_y) {
    return this->MoveByXYVel(netout_x, netout_y);
}

void DroneAgent::UpdateStates(const std::shared_ptr<GridMap>& grid_map, std::pair<double, double> velocity_vector, const std::vector<std::vector<std::vector<int>>>& drone_view, int water_dispense) {
    // Update the states, last state get's kicked out
    auto explored_map = grid_map->GetExploredMap();
    auto fire_map = grid_map->GetFireMap();
    DroneState new_state = DroneState(velocity_vector, parameters_.GetMaxVelocity(), drone_view,
                                      explored_map, fire_map, map_dimensions_,
                                      position_, goal_position_, water_dispense,
                                      parameters_.GetCellSize());
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

bool DroneAgent::DispenseWaterCertain(const std::shared_ptr<GridMap>& grid_map) {
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

void DroneAgent::DispenseWater(const std::shared_ptr<GridMap>& grid_map, int water_dispense) {
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

std::pair<double, double> DroneAgent::GetRealPosition() {
    return position_;
}

std::pair<int, int> DroneAgent::GetGridPosition() {
    int x, y;
    parameters_.ConvertRealToGridCoordinates(position_.first, position_.second, x, y);
    return std::make_pair(x, y);
}

std::pair<double, double> DroneAgent::GetGridPositionDouble() {
//    double x, y;
//    x = position_.first / parameters_.GetCellSize();
//    y = position_.second / parameters_.GetCellSize();
//    return std::make_pair(x, y);
    return this->GetLastState().GetGridPositionDouble();
}

// Checks whether the drone sees fire in the current fire status and return how much
int DroneAgent::DroneSeesFire() {
    std::vector<std::vector<int>> fire_status = GetLastState().GetFireView();
    int count = std::accumulate(fire_status.begin(), fire_status.end(), 0,
                                [](int acc, const std::vector<int>& vec) {
                                    return acc + std::count(vec.begin(), vec.end(), 1);
                                }
    );
    return count;
}

double DroneAgent::FindNearestFireDistance() {
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

void DroneAgent::CalcMaxDistanceFromMap() {
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

double DroneAgent::GetDistanceToGoal() {
    return sqrt(pow(this->GetGoalPosition().first - this->GetGridPositionDouble().first, 2) +
         pow(this->GetGoalPosition().second - this->GetGridPositionDouble().second, 2)
    );
}

void DroneAgent::OnFlyAction(std::shared_ptr<FlyAction> action, std::shared_ptr<GridMap> gridMap) {
    // Get the speed and water dispense from the action
    double speed_x = action->GetSpeedX(); // change this to "real" speed
    double speed_y = action->GetSpeedY();
    this->Step(speed_x, speed_y, gridMap);
}

void DroneAgent::OnExploreAction(std::shared_ptr<ExploreAction> action, std::shared_ptr<GridMap> gridMap) {
    // Get the goal position from the action
    double x_off = gridMap->GetXOff();
    double y_off = gridMap->GetYOff();
    double goal_x = action->GetGoalX();
    double goal_y = action->GetGoalY();
    goal_x = goal_x < 0 ? (goal_x + x_off) : (goal_x - x_off);
    goal_y = goal_y < 0 ? (goal_y + y_off) : (goal_y - y_off);
    goal_x = ((goal_x + 1) / 2) * gridMap->GetCols();
    goal_y = ((goal_y + 1) / 2) * gridMap->GetRows();
    this->SetGoalPosition(std::make_pair(goal_x, goal_y));
}

void DroneAgent::Step(double speed_x, double speed_y, const std::shared_ptr<GridMap>& gridmap) {
    std::pair<double, double> vel_vector;
    vel_vector = this->MovementStep(speed_x, speed_y);
    auto drone_view = gridmap->GetDroneView(std::dynamic_pointer_cast<DroneAgent>(shared_from_this()));
    gridmap->UpdateExploredAreaFromDrone(std::dynamic_pointer_cast<DroneAgent>(shared_from_this()));
    this->UpdateStates(gridmap, vel_vector, drone_view, 0);
}

void DroneAgent::PolicyStep(const std::shared_ptr<GridMap>& gridmap){
    if (agent_type_ == "FlyAgent") {
        this->SimplePolicy(gridmap);
    } else {
        this->ExplorePolicy(gridmap);
    }
}

void DroneAgent::SimplePolicy(const std::shared_ptr<GridMap>& gridmap){
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
                auto next_fire = gridmap->GetNextFire(std::dynamic_pointer_cast<DroneAgent>(shared_from_this()));
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
                this->SetGoalPosition(gridmap->GetNextFire(std::dynamic_pointer_cast<DroneAgent>(shared_from_this())));
            }
        } else {
            this->SetWaterCapacity(parameters_.GetWaterCapacity());
            this->SetPolicyType(0);
            this->SetGoalPosition(gridmap->GetNextFire(std::dynamic_pointer_cast<DroneAgent>(shared_from_this())));
        }
    }
}

void DroneAgent::ExplorePolicy(const std::shared_ptr<GridMap> &gridmap){

}

std::pair<bool, bool> DroneAgent::IsTerminal(bool eval_mode, const std::shared_ptr<GridMap>& grid_map, int total_env_steps) const {
    bool terminal_state = false;
    bool drone_died = false;
    if (GetOutOfAreaCounter() > 1) {
        terminal_state = true;
        drone_died = true;
    }
    if (GetReachedGoal() && !eval_mode) {
        terminal_state = true;
    }
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
    if (total_env_steps <= 0 && !eval_mode) {
        terminal_state = true;
        drone_died = true;
    }
    return std::make_pair(terminal_state, drone_died);
}

double DroneAgent::CalculateReward(bool terminal_state, int total_env_steps) {
    if (agent_type_ == "FlyAgent") {
        return CalculateFlyReward(terminal_state, total_env_steps);
    } else {
        return CalculateExploreReward(terminal_state, total_env_steps);
    }
}


double DroneAgent::CalculateFlyReward(bool terminal_state, int total_env_steps) {
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

    if (!drone_in_grid && terminal_state) {
        reward_components["BoundaryTerminal"] = -2;
    }

    if (terminal_state && (total_env_steps <= 0)) {
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
    return total_reward;
}

double DroneAgent::CalculateExploreReward(bool terminal_state, int total_env_steps){
    double distance_to_goal = GetDistanceToGoal();
    double last_distance_to_goal = GetLastDistanceToGoal();
    double distance_to_boundary = GetLastState().GetDistanceToNearestBoundaryNorm();
    double delta_distance = last_distance_to_goal - distance_to_goal;
    bool drone_in_grid = GetDroneInGrid();
    auto explore_map = GetLastState().GetExplorationMapScalar();

    std::unordered_map<std::string, double> reward_components;

    reward_components["explore_map"] = explore_map;

    double total_reward = ComputeTotalReward(reward_components);
    LogRewards(reward_components);
    SetLastDistanceToGoal(distance_to_goal);
    reward_components_ = reward_components;
    return total_reward;
}

double DroneAgent::ComputeTotalReward(const std::unordered_map<std::string, double>& rewards) {
    double total = 0;
    for (const auto& [key, value] : rewards) {
        total += value;
    }
    return total;
}

void DroneAgent::LogRewards(const std::unordered_map<std::string, double>& rewards) {
#ifdef DEBUG_REWARD_YES
    for (const auto& [key, value] : rewards) {
            std::cout << key << " Reward: " << value << "\n";
        }
        std::cout << "\n";
#endif
}