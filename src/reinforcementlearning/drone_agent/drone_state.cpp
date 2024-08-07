//
// Created by nex on 15.07.23.
//

#include "drone_state.h"

DroneState::DroneState(double speed_x, double speed_y, std::pair<double, double> max_speed, std::vector<std::vector<int>> terrain, std::vector<std::vector<int>> fire_status,
                       std::vector<std::vector<int>> map, std::pair<double, double> map_dimensions, std::pair<double, double> position, double cell_size) {
    velocity_.first = speed_x;
    velocity_.second = speed_y;
    terrain_ = std::move(terrain);
    fire_status_ = std::move(fire_status);
    max_speed_ = max_speed;
    map_ = std::move(map);
    position_ = position;
    map_dimensions_ = map_dimensions;
    cell_size_ = cell_size;
}

std::pair<double, double> DroneState::GetNewVelocity(double netout_speed_x, double netout_speed_y) {
    // Netout determines the velocity CHANGE
    double new_speed_x = velocity_.first + netout_speed_x * max_speed_.first;
    double new_speed_y = velocity_.second + netout_speed_y * max_speed_.second;

    // Netout determines the velocity DIRECTLY
    // double new_speed_x = netout_speed_x * max_speed_.first;
    // double new_speed_y = netout_speed_y * max_speed_.second;

    new_speed_x = (new_speed_x < max_speed_.first) ? new_speed_x : max_speed_.first;
    new_speed_x = (new_speed_x > -max_speed_.first) ? new_speed_x : -max_speed_.first;
    // Not by Angle
    new_speed_y = (new_speed_y < max_speed_.second) ? new_speed_y : max_speed_.second;
    new_speed_y = (new_speed_y > -max_speed_.second) ? new_speed_y : -max_speed_.second;

    // By Angle
    // double angle = fmod(new_speed_y, 2 * M_PI);

    // // This ensures that if the angle is negative, it's converted to a positive value within the range.
    // if (angle < 0) {
    //     angle += 2 * M_PI;
    // }
    // new_speed_y = angle;

    return std::make_pair(new_speed_x, new_speed_y);
}

std::vector<std::vector<double>> DroneState::GetTerrainNorm() {
    std::vector<std::vector<double>> terrain_norm(terrain_.size(), std::vector<double>(terrain_[0].size()));

    for (size_t i = 0; i < terrain_.size(); ++i) {
        for (size_t j = 0; j < terrain_[i].size(); ++j) {
            terrain_norm[i][j] = static_cast<double>(terrain_[i][j]) / int(static_cast<CellState>(CELL_STATE_COUNT - 1));
        }
    }

    return terrain_norm;
}

int DroneState::CountOutsideArea() {
    int outside_area = 0;
    for (size_t i = 0; i < terrain_.size(); ++i) {
        for (size_t j = 0; j < terrain_[i].size(); ++j) {
            if (terrain_[i][j] == OUTSIDE_GRID) {
                outside_area++;
            }
        }
    }
    return outside_area;
}

std::vector<std::vector<double>> DroneState::GetFireStatusNorm() {
    return std::vector<std::vector<double>>();
}

std::vector<std::vector<double>> DroneState::GetMapNorm() {
    return std::vector<std::vector<double>>();
}

std::pair<double, double> DroneState::GetPositionNorm() const {
    double x = (2.0 * position_.first / (map_dimensions_.first * cell_size_)) - 1;
    double y = (2.0 * position_.second / (map_dimensions_.second * cell_size_)) - 1;
    return std::make_pair(x, y);
}


