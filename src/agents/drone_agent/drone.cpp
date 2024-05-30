//
// Created by nex on 13.07.23.
//

#include <iostream>
#include "drone.h"

DroneAgent::DroneAgent(std::shared_ptr<SDL_Renderer> renderer, std::pair<int, int> point, FireModelParameters &parameters, int id) : id_(id), parameters_(parameters) {
    renderer_ = DroneRenderer(renderer);
    int x = point.first; // position in grid
    int y = point.second; // position in grid
    position_ = std::make_pair(x * parameters_.GetCellSize(), y * parameters_.GetCellSize());
    view_range_ = 20;
    out_of_area_counter_ = 0;
}

void DroneAgent::Render(std::pair<int, int> position, int size) {
    renderer_.Render(position, size, view_range_, 0);
}

void DroneAgent::MoveByAngle(double netout_speed, double netout_angle){
    // If you use this function you currently need to manually set the correct boundarys at the velocity max_vector in parameters_
    std::pair<double, double> velocity_vector = drone_states_[0].GetNewVelocity(netout_speed, netout_angle);
    double vel = velocity_vector.first * parameters_.GetDt();
    double angle = velocity_vector.second;
    position_.first += vel * cos(angle);
    position_.second += vel * sin(angle);
}

void DroneAgent::Move(double netout_speed_x, double netout_speed_y) {
    std::pair<double, double> velocity_vector = drone_states_[0].GetNewVelocity(netout_speed_x, netout_speed_y);
    position_.first += velocity_vector.first * parameters_.GetDt();
    position_.second += velocity_vector.second * parameters_.GetDt();
}

void DroneAgent::Update(double netout_speed_x, double netout_speed_y, std::vector<std::vector<int>> terrain, std::vector<std::vector<int>> fire_status, std::vector<std::vector<int>> updated_map) {
    UpdateStates(netout_speed_x, netout_speed_y, terrain, fire_status, updated_map);
}

void DroneAgent::Initialize(std::vector<std::vector<int>> terrain, std::vector<std::vector<int>> fire_status, std::pair<int, int> size, double cell_size) {
    for(int i = 0; i < 4; ++i) {
        std::vector<std::vector<int>> map(size.first, std::vector<int>(size.second, -1));
        //std::pair<double, double> size_ = std::make_pair(size.first * parameters_.GetCellSize(), size.second * parameters_.GetCellSize());
        DroneState new_state = DroneState(0, 0, parameters_.GetMaxVelocity(), terrain, fire_status, map, size, position_, cell_size);
        drone_states_.push_front(new_state);
    }
}

void DroneAgent::UpdateStates(double netout_speed_x, double netout_speed_y, std::vector<std::vector<int>> terrain, std::vector<std::vector<int>> fire_status, std::vector<std::vector<int>> updated_map) {
    // Update the states, last state get's kicked out
    DroneState new_state = drone_states_[0].GetNewState(netout_speed_x, netout_speed_y, terrain, fire_status, updated_map, position_);
    drone_states_.push_front(new_state);

    // Maximum number of states i.e. memory
    if (drone_states_.size() > 4) {
        drone_states_.pop_back();
    }
}

bool DroneAgent::DispenseWater(GridMap &grid_map) {

    std::pair<int, int> grid_position = GetGridPosition();
    bool fire_extinguished = grid_map.WaterDispension(grid_position.first, grid_position.second);
    return fire_extinguished;

}

std::pair<double, double> DroneAgent::GetGridPositionDouble() {
    double x, y;
    x = position_.first / parameters_.GetCellSize();
    y = position_.second / parameters_.GetCellSize();
    return std::make_pair(x, y);
}

std::pair<double, double> DroneAgent::GetRealPosition() {
    return position_;
}

std::pair<int, int> DroneAgent::GetGridPosition() {
    int x, y;
    parameters_.ConvertRealToGridCoordinatesDrone(position_.first, position_.second, x, y);
    return std::make_pair(x, y);
}

// Checks whether the drone sees fire in the current fire status and return how much
int DroneAgent::DroneSeesFire() {
    std::vector<std::vector<int>> fire_status = GetLastState().GetFireStatus();
    int count = std::accumulate(fire_status.begin(), fire_status.end(), 0,
                                [](int acc, const std::vector<int>& vec) {
                                    return acc + std::count(vec.begin(), vec.end(), 1);
                                }
    );
    return count;
}

std::pair<double, double> DroneAgent::GetRealPositionFromGrid(int x, int y) {
    double real_x = x * parameters_.GetCellSize();
    double real_y = y * parameters_.GetCellSize();
    return std::make_pair(real_x, real_y);
}

double DroneAgent::FindNearestFireDistance() {
    std::pair<double, double> drone_real_position = GetRealPosition();
    std::pair<int, int> drone_grid_position = GetGridPosition();
    double min_distance = std::numeric_limits<double>::max();
    std::vector<std::vector<int>> fire_status = GetLastState().GetFireStatus();

    for (int y = 0; y <= view_range_; ++y) {
        for (int x = 0; x <= view_range_; ++x) {
            if (fire_status[x][y] == 1) { // Assuming 1 indicates fire
                std::pair<int, int> fire_grid_position = std::make_pair(
                        drone_grid_position.first + y - (view_range_ / 2),
                         drone_grid_position.second + x - (view_range_ / 2)
                );

                std::pair<double, double> fire_real_position = GetRealPositionFromGrid(fire_grid_position.first, fire_grid_position.second);
                double distance = sqrt(
                        pow(fire_real_position.first - drone_real_position.first, 2) +
                        pow(fire_real_position.second - drone_real_position.second, 2)
                );

                if (distance < min_distance) {
                    min_distance = distance;
                }
            }
        }
    }

    return min_distance;
}

//DroneAgent::~DroneAgent() {
//    drone_states_.clear();
//
//}
