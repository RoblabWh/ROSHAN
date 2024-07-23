//
// Created by nex on 15.07.23.
//

#ifndef ROSHAN_DRONE_STATE_H
#define ROSHAN_DRONE_STATE_H

#include <utility>
#include <vector>
#include <cmath>
#include <array>
#include <iostream>
#include <numeric>
#include <algorithm>
#include "models/firespin/utils.h"
#include "state.h"


class DroneState : public State{
public:
    explicit DroneState(double speed_x, double speed_y, std::pair<double, double> max_speed, std::vector<std::vector<int>> terrain,
                        std::vector<std::vector<int>> fire_status, std::vector<std::vector<int>> map, std::pair<double, double> map_dimensions,
                        std::pair<double, double> position, double cell_size);
//    void SetOrientation() { orientation_vector_.first = cos(velocity_.first); orientation_vector_.second = sin(velocity_.first); }
//    std::pair<double, double> GetOrientation() { return orientation_vector_; }
//    std::pair<double, double> GetNewOrientation(double angular) { return std::make_pair(cos(velocity_.first + angular), sin(velocity_.first + angular)); }
    void SetVelocity(double speed_x, double speed_y) { velocity_.first = speed_x; velocity_.second = speed_y; }
    std::pair<double, double> GetNewVelocity(double speed_x, double speed_y);
    // These are for the states
    std::pair<double, double> GetVelocity() { return velocity_; }
    std::pair<double, double> GetVelocityNorm() { return std::make_pair(velocity_.first / max_speed_.first, velocity_.second / max_speed_.second); }
    std::vector<std::vector<int>> GetTerrain() { return terrain_; }
    std::vector<std::vector<double>> GetTerrainNorm();
    std::vector<std::vector<int>> GetFireStatus() { return fire_status_; }
    std::vector<std::vector<double>> GetFireStatusNorm();
    int CountOutsideArea();
    std::vector<std::vector<int>> GetMap() { return map_; }
    static std::vector<std::vector<double>> GetMapNorm();
    std::pair<double, double> GetPositionNorm() const;

    // For python visibility
    std::pair<double, double> get_velocity() const { return velocity_; }
    std::vector<std::vector<int>> get_terrain() const { return terrain_; }
    std::vector<std::vector<int>> get_fire_status() const { return fire_status_; }
    std::vector<std::vector<int>> get_map() const { return map_; }
    std::pair<int, int> get_position() const { return position_; }
    std::pair<double, double> get_orientation_vector() const { return orientation_vector_; }
private:
    std::pair<double, double> velocity_; // speed_x, speed_y
    std::pair<double, double> max_speed_;
    double max_angle_ = M_PI_2;
    std::pair<double, double> map_dimensions_;
    double cell_size_;
    std::pair<double, double> position_; // x, y
    std::vector<std::vector<int>> terrain_;
    std::vector<std::vector<int>> fire_status_;
    std::vector<std::vector<int>> map_;
    std::pair<double, double> orientation_vector_; // x, y
};

#endif //ROSHAN_DRONE_STATE_H
