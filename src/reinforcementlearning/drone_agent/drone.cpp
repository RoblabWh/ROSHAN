//
// Created by nex on 13.07.23.
//

#include <iostream>
#include "drone.h"

DroneAgent::DroneAgent(std::pair<int, int> point, FireModelParameters &parameters, int id) : id_(id), parameters_(parameters) {
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

void DroneAgent::Initialize(GridMap &grid_map) {
    map_dimensions_ = std::make_pair(grid_map.GetRows(), grid_map.GetCols());
    auto drone_view = grid_map.GetDroneView(std::dynamic_pointer_cast<DroneAgent>(shared_from_this()));
    auto explored_map = grid_map.GetExploredMap();
    auto fire_map = grid_map.GetFireMap();
    dispensed_water_ = false;
    for(int i = 0; i < time_steps_; ++i) {
        DroneState new_state = DroneState(std::pair<int,int>(0,0),
                parameters_.GetMaxVelocity(), drone_view,
                explored_map, fire_map, map_dimensions_, position_,
                goal_position_, 0, parameters_.GetCellSize());
        drone_states_.push_front(new_state);
    }
}

void DroneAgent::Render(std::pair<int, int> position, int size) {
    renderer_.Render(position, size, view_range_, 0, active_);
}

std::pair<double, double> DroneAgent::MoveByAngle(double netout_speed, double netout_angle){
    // If you use this function you currently need to manually set the correct boundarys at the velocity max_vector in parameters_
    std::pair<double, double> velocity_vector = drone_states_[0].GetNewVelocity(netout_speed, netout_angle);
    double vel = velocity_vector.first * parameters_.GetDt();
    double angle = velocity_vector.second;
    position_.first += vel * cos(angle);
    position_.second += vel * sin(angle);
    return velocity_vector;
}

std::pair<double, double> DroneAgent::MoveByXYVel(double netout_speed_x, double netout_speed_y) {
    std::pair<double, double> velocity_vector = drone_states_[0].GetNewVelocity(netout_speed_x, netout_speed_y);
    position_.first += velocity_vector.first * parameters_.GetDt();
    position_.second += velocity_vector.second * parameters_.GetDt();
    return velocity_vector;
}

std::pair<double, double> DroneAgent::MovementStep(double netout_x, double netout_y) {
    return this->MoveByXYVel(netout_x, netout_y);
}

void DroneAgent::UpdateStates(GridMap &grid_map, std::pair<double, double> velocity_vector, const std::vector<std::vector<std::vector<int>>>& drone_view, int water_dispense) {
    // Update the states, last state get's kicked out
    auto explored_map = grid_map.GetExploredMap();
    auto fire_map = grid_map.GetFireMap();
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
    drone_in_grid_ = grid_map.IsPointInGrid(drone_position.first, drone_position.second);
    // Calculates if the Drone is in the grid and if not how far it is away from the grid
    CalcMaxDistanceFromMap();
}

bool DroneAgent::DispenseWaterCertain(GridMap &grid_map) {
    std::pair<int, int> grid_position = GetGridPosition();
    bool cell_is_burning = false;
    if (grid_map.IsPointInGrid(grid_position.first, grid_position.second)){
        cell_is_burning = grid_map.At(grid_position.first, grid_position.second).IsBurning();
    }
    if (cell_is_burning) {
        dispensed_water_ = true;
        water_capacity_ -= 1;
        bool fire_extinguished = grid_map.WaterDispension(grid_position.first, grid_position.second);
        if (fire_extinguished) {
            if (grid_map.GetNumBurningCells() == 0) {
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

void DroneAgent::DispenseWater(GridMap &grid_map, int water_dispense) {
    // Returns true if fire was extinguished
    if (water_dispense == 1) {
        dispensed_water_ = true;
        std::pair<int, int> grid_position = GetGridPosition();
        bool fire_extinguished = grid_map.WaterDispension(grid_position.first, grid_position.second);
        if (fire_extinguished) {
            if (grid_map.GetNumBurningCells() == 0) {
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
    double x, y;
    x = position_.first / parameters_.GetCellSize();
    y = position_.second / parameters_.GetCellSize();
    return std::make_pair(x, y);
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
        std::pair<double, double> pos = GetLastState().GetPositionNorm();
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

void DroneAgent::Step(double speed_x, double speed_y, const std::shared_ptr<GridMap>& gridmap) {
    this->SetReachedGoal(false);
    std::pair<double, double> vel_vector;
    if(this->GetPolicyType() == 0) {
        vel_vector = this->MovementStep(speed_x, speed_y);
        if (this->GetGoalPositionInt() == this->GetGridPosition()) {
            this->SetReachedGoal(true);
            this->DispenseWaterCertain(*gridmap);
            if (this->GetWaterCapacity() <= 0) {
                auto groundstation_position = gridmap->GetGroundstationPosition();
                this->SetGoalPosition(groundstation_position);
                this->SetPolicyType(1);
            } else {
                auto next_fire = gridmap->GetNextFire(std::dynamic_pointer_cast<DroneAgent>(shared_from_this()));
                this->SetGoalPosition(next_fire);
            }
        }
    } else if (this->GetPolicyType() == 1) {
        vel_vector = this->MovementStep(speed_x, speed_y);
        if (this->GetGoalPositionInt() == this->GetGridPosition()) {
            this->SetPolicyType(2);
        }
    } else {
        vel_vector = std::make_pair(0, 0);
        if (this->GetWaterCapacity() <= parameters_.GetWaterCapacity()) {
            this->SetWaterCapacity(this->GetWaterCapacity() + parameters_.GetWaterRefillDt());
        } else {
            this->SetPolicyType(0);
            this->SetGoalPosition(gridmap->GetNextFire(std::dynamic_pointer_cast<DroneAgent>(shared_from_this())));
        }
    }
//    drones_->at(drone_idx)->DispenseWater(*gridmap_, water_dispense);
    auto drone_view = gridmap->GetDroneView(std::dynamic_pointer_cast<DroneAgent>(shared_from_this()));
    gridmap->UpdateExploredAreaFromDrone(std::dynamic_pointer_cast<DroneAgent>(shared_from_this()));
    // TODO consider not only adding the current velocity, but the last netoutputs (these are two potential dimensions)
    this->UpdateStates(*gridmap, vel_vector, drone_view, 0);
}

void DroneAgent::OnDroneAction(std::shared_ptr<DroneAction> action, const std::shared_ptr<GridMap> gridMap) {
    // Get the speed and water dispense from the action
    double speed_x = action->GetSpeedX(); // change this to "real" speed
    double speed_y = action->GetSpeedY();
    this->Step(speed_x, speed_y, gridMap);
}
