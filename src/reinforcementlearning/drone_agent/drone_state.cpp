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

std::pair<double, double> DroneState::GetNewVelocity(double next_speed_x, double next_speed_y) const {
    // Next Speed determines the velocity CHANGE
//    next_speed_x = DiscretizeOutput(next_speed_x, 0.05);
//    next_speed_y = DiscretizeOutput(next_speed_y, 0.05);
//    double new_speed_x = velocity_.first + next_speed_x * max_speed_.first;
//    double new_speed_y = velocity_.second + next_speed_y * max_speed_.second;

    // Netout determines the velocity DIRECTLY #TODO: Why does this perform worse??
     double new_speed_x = next_speed_x * max_speed_.first;
     double new_speed_y = next_speed_y * max_speed_.second;

    // Clamp new_speed between -max_speed_.first and max_speed_.first
    new_speed_x = std::clamp(new_speed_x, -max_speed_.first, max_speed_.first);
    new_speed_y = std::clamp(new_speed_y, -max_speed_.second, max_speed_.second);

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

std::pair<double, double> DroneState::GetPositionNormAroundCenter() const {
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
    auto largest_side = std::max(map_dimensions_.first, map_dimensions_.second);
//    auto diagonal = sqrt(map_dimensions_.first * map_dimensions_.first + map_dimensions_.second * map_dimensions_.second);
    double x = ((position.first - goal_position_.first) / largest_side) * 1;
    double y = ((position.second - goal_position_.second) / largest_side) * 1;
    return std::make_pair(x, y);
}

[[nodiscard]] std::pair<double, double> DroneState::GetVelocityNorm() const {
    auto largest_side = std::max(map_dimensions_.first, map_dimensions_.second);
    return std::make_pair(velocity_.first / (largest_side * cell_size_), velocity_.second / (largest_side * cell_size_));
    //{ return std::make_pair(velocity_.first / max_speed_.first, velocity_.second / max_speed_.second); }
}

std::pair<double, double> DroneState::GetOrientationToGoal() const {
    double x = goal_position_.first - position_.first;
    double y = goal_position_.second - position_.second;
    double magnitude = sqrt(x * x + y * y);
    return std::make_pair(x / magnitude, y / magnitude);
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

double DroneState::GetDistanceToNearestBoundaryNorm() const {
    auto largest_side = std::max(map_dimensions_.first, map_dimensions_.second);
    auto position = GetGridPositionDouble();
    double x = position.first;
    double y = position.second;

    if (x < 0) {
        x = std::abs(x);
    }
    if (y < 0) {
        y = std::abs(y);
    }
    auto distance_x = map_dimensions_.first - x;
    auto distance_y = map_dimensions_.second - y;
    auto distance = std::min({x, y, distance_x, distance_y});

    return distance / largest_side;
}

std::vector<std::vector<double>> DroneState::GetExplorationMapNorm() const {
    std::vector<std::vector<double>> exploration_map_norm(exploration_map_.size(), std::vector<double>(exploration_map_[0].size()));
    double max_value = map_dimensions_.first * map_dimensions_.second;
    for (size_t i = 0; i < exploration_map_.size(); ++i) {
        for (size_t j = 0; j < exploration_map_[i].size(); ++j) {
            exploration_map_norm[i][j] = static_cast<double>(exploration_map_[i][j]) / max_value;
        }
    }
    return exploration_map_norm;
}

double DroneState::GetExplorationMapScalar() const {
    auto explore_norm_ = this->GetExplorationMapNorm();
    double scalar = 0;
    int max = explore_norm_.size() * explore_norm_[0].size();
    for (size_t i = 0; i < explore_norm_.size(); i++) {
        for(size_t j = 0; j < explore_norm_[i].size(); j++) {
            scalar += explore_norm_[i][j];
        }
    }
    return scalar / max;
}


