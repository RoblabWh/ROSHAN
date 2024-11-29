//
// Created by nex on 15.07.23.
//

#include "drone_state.h"

DroneState::DroneState(std::pair<double, double> velocity_vector,
                       std::pair<double, double> max_speed,
                       std::vector<std::vector<std::vector<int>>> drone_view,
                       std::vector<std::vector<int>> exploration_map,
                       std::vector<std::vector<double>> fire_map,
                       std::pair<double, double> map_dimensions,
                       std::pair<double, double> position,
                       std::pair<double, double> goal_position,
                       int water_dispense,
                       double cell_size) {
    velocity_ = velocity_vector;
    drone_view_ = std::move(drone_view);
    exploration_map_ = std::move(exploration_map);
    fire_map_ = std::move(fire_map);
    max_speed_ = max_speed;
    position_ = position;
    goal_position_ = goal_position;
    water_dispense_ = water_dispense;
    map_dimensions_ = map_dimensions;
    cell_size_ = cell_size;
}

//double DroneState::DiscretizeOutput(double netout) {
//    if (netout <= -0.75) return -1.0;
//    if (netout <= -0.25) return -0.5;
//    if (netout <= 0.25) return 0.0;
//    if (netout <= 0.75) return 0.5;
//    return 1.0;
//}

//double DroneState::DiscretizeOutput(double netout) {
//    if (netout <= -0.9) return -1.0;
//    if (netout <= -0.7) return -0.8;
//    if (netout <= -0.5) return -0.6;
//    if (netout <= -0.3) return -0.4;
//    if (netout <= -0.1) return -0.2;
//    if (netout <= 0.1) return 0.0;
//    if (netout <= 0.3) return 0.2;
//    if (netout <= 0.5) return 0.4;
//    if (netout <= 0.7) return 0.6;
//    if (netout <= 0.9) return 0.8;
//    return 1.0;
//}

double DroneState::DiscretizeOutput(double netout) {
    if (netout <= -0.9) return -1.0;
    if (netout <= -0.8) return -0.9;
    if (netout <= -0.7) return -0.8;
    if (netout <= -0.6) return -0.7;
    if (netout <= -0.5) return -0.6;
    if (netout <= -0.4) return -0.5;
    if (netout <= -0.3) return -0.4;
    if (netout <= -0.2) return -0.3;
    if (netout <= -0.1) return -0.2;
    if (netout <= -0.05) return -0.1;
    if (netout < 0.05) return 0.0;
    if (netout >= 0.05) return 0.1;
    if (netout >= 0.1) return 0.2;
    if (netout >= 0.2) return 0.3;
    if (netout >= 0.3) return 0.4;
    if (netout >= 0.4) return 0.5;
    if (netout >= 0.5) return 0.6;
    if (netout >= 0.6) return 0.7;
    if (netout >= 0.7) return 0.8;
    if (netout >= 0.8) return 0.9;
    return 1.0;
}

std::pair<double, double> DroneState::GetNewVelocity(double netout_speed_x, double netout_speed_y) {
    // Netout determines the velocity CHANGE
//    auto speed_x = std::round(netout_speed_x * 10.0) / 10.0;
//    auto speed_y = std::round(netout_speed_y * 10.0) / 10.0;
//    auto speed_x = netout_speed_x;
//    auto speed_y = netout_speed_y;
    auto speed_x = DiscretizeOutput(netout_speed_x);
    auto speed_y = DiscretizeOutput(netout_speed_y);
    double new_speed_x = velocity_.first + speed_x * max_speed_.first;
    double new_speed_y = velocity_.second + speed_y * max_speed_.second;

    // Netout determines the velocity DIRECTLY #TODO: Why does this perform worse??
//     double new_speed_x = netout_speed_x * max_speed_.first;
//     double new_speed_y = netout_speed_y * max_speed_.second;

    // Clamp new_speed between -max_speed_.first and max_speed_.first
    new_speed_x = std::clamp(new_speed_x, -max_speed_.first, max_speed_.first);
    new_speed_y = std::clamp(new_speed_y, -max_speed_.second, max_speed_.second);

    // By Angle
    // double angle = fmod(new_speed_y, 2 * M_PI);

    // // This ensures that if the angle is negative, it's converted to a positive value within the range.
    // if (angle < 0) {
    //     angle += 2 * M_PI;
    // }
    // new_speed_y = angle;

    return std::make_pair(new_speed_x, new_speed_y);
}

std::vector<std::vector<std::vector<int>>> DroneState::GetDroneViewNorm() {

    std::vector<std::vector<std::vector<int>>> drone_view_norm(2, std::vector<std::vector<int>>(drone_view_[0].size(), std::vector<int>(drone_view_[0][0].size())));

    for (size_t i = 0; i < drone_view_[0].size(); ++i) {
        for (size_t j = 0; j < drone_view_[0][i].size(); ++j) {
            drone_view_norm[0][i][j] = static_cast<int>(drone_view_[0][i][j]) / int(static_cast<CellState>(CELL_STATE_COUNT - 1));
            drone_view_norm[1][i][j] = drone_view_[1][i][j];
        }
    }

    return drone_view_norm;
}

int DroneState::CountOutsideArea() {
    int outside_area = 0;
    for (size_t i = 0; i < drone_view_[0].size(); ++i) {
        for (size_t j = 0; j < drone_view_[0][i].size(); ++j) {
            if (drone_view_[0][i][j] == OUTSIDE_GRID) {
                outside_area++;
            }
        }
    }
    return outside_area;
}

std::pair<double, double> DroneState::GetPositionNorm() const {
    double x = (2.0 * position_.first / (map_dimensions_.first * cell_size_)) - 1;
    double y = (2.0 * position_.second / (map_dimensions_.second * cell_size_)) - 1;
    return std::make_pair(x, y);
}

std::pair<double, double> DroneState::GetGoalPositionNorm() const {
    double x = goal_position_.first / map_dimensions_.first;
    double y = goal_position_.second / map_dimensions_.second;
    return std::make_pair(x, y);
}

std::pair<double, double> DroneState::GetDeltaGoal() const {
    auto position = GetGridPositionDouble();
    double x = goal_position_.first - position.first;
    double y = goal_position_.second - position.second;
    return std::make_pair(x, y);
}

std::pair<double, double> DroneState::GetOrientationToGoal() const {
    double x = goal_position_.first - position_.first;
    double y = goal_position_.second - position_.second;
    double magnitude = sqrt(x * x + y * y);
    return std::make_pair(x / magnitude, y / magnitude);
//    double angle = atan2(y, x);
//    return std::make_pair(cos(angle), sin(angle));
}

std::pair<double, double> DroneState::GetGridPositionDoubleNorm() const {
    auto grid_position = GetGridPositionDouble();
    double x = grid_position.first / map_dimensions_.first;
    double y = grid_position.second / map_dimensions_.second;
    return std::make_pair(x, y);
}

std::pair<double, double> DroneState::GetGridPositionDouble() const {
    double x, y;
    x = position_.first / cell_size_;
    y = position_.second / cell_size_;
    return std::make_pair(x, y);
}

std::vector<std::vector<double>> DroneState::GetExplorationMapNorm() const {
    std::vector<std::vector<double>> exploration_map_norm(exploration_map_.size(), std::vector<double>(exploration_map_[0].size()));
    int max_value = map_dimensions_.first * map_dimensions_.second;
    for (size_t i = 0; i < exploration_map_.size(); ++i) {
        for (size_t j = 0; j < exploration_map_[i].size(); ++j) {
            exploration_map_norm[i][j] = static_cast<double>(exploration_map_[i][j]) / max_value;
        }
    }
    return exploration_map_norm;
}


