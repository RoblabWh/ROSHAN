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
    explicit DroneState(std::pair<double, double> velocity_vector,
                        std::pair<double, double> max_speed,
                        std::vector<std::vector<std::vector<int>>> drone_view,
                        std::vector<std::vector<int>> exploration_map,
                        std::vector<std::vector<double>> fire_map,
                        std::pair<double, double> map_dimensions,
                        std::pair<double, double> position,
                        std::pair<double, double> goal_position,
                        int water_dispense,
                        double cell_size);
//    void SetOrientation() { orientation_vector_.first = cos(velocity_.first); orientation_vector_.second = sin(velocity_.first); }
//    std::pair<double, double> GetOrientation() { return orientation_vector_; }
//    std::pair<double, double> GetNewOrientation(double angular) { return std::make_pair(cos(velocity_.first + angular), sin(velocity_.first + angular)); }
    void SetVelocity(double speed_x, double speed_y) { velocity_.first = speed_x; velocity_.second = speed_y; }
    std::pair<double, double> GetNewVelocity(double speed_x, double speed_y);

    // These functions are for the states
    std::pair<double, double> GetVelocity() { return velocity_; }
    std::pair<double, double> GetVelocityNorm() { return std::make_pair(velocity_.first / max_speed_.first, velocity_.second / max_speed_.second); }
    std::vector<std::vector<int>> GetTerrain() { return drone_view_[0]; }
    std::vector<std::vector<int>> GetFireStatus() { return drone_view_[1]; }
    std::vector<std::vector<int>> GetExplorationMap() { return exploration_map_; }
    std::vector<std::vector<double>> GetFireMap() { return fire_map_; }
    int GetWaterDispense() { return water_dispense_; }
    std::vector<std::vector<double>> GetExplorationMapNorm() const;
    static std::vector<std::vector<double>> GetMapNorm();
    std::pair<double, double> GetPositionNorm() const;
    std::pair<double, double> GetGridPositionDouble() const;
    std::pair<double, double> GetGridPositionDoubleNorm() const;
    std::pair<double, double> GetGoalPosition() const { return goal_position_; }
    std::pair<double, double> GetGoalPositionNorm() const;
    std::pair<double, double> GetDeltaGoal() const;
    std::pair<double, double> GetOrientationToGoal() const;
    std::vector<std::vector<std::vector<int>>> GetDroneViewNorm();

    // Used for Reward Calculation
    int CountOutsideArea();

    // For python visibility
    std::pair<double, double> get_velocity() const { return velocity_; }
    std::vector<std::vector<std::vector<int>>> get_drone_view() const { return drone_view_; }
    std::vector<std::vector<int>> get_map() const { return exploration_map_; }
    std::vector<std::vector<double>> get_fire_map() const { return fire_map_; }
    std::pair<int, int> get_position() const { return position_; }
    std::pair<double, double> get_orientation_vector() const { return orientation_vector_; }
private:
    std::pair<double, double> velocity_; // speed_x, speed_y
    std::pair<double, double> max_speed_;
    int water_dispense_;
    double max_angle_ = M_PI_2;
    std::pair<double, double> map_dimensions_;
    double cell_size_;
    std::pair<double, double> position_; // x, y
    std::pair<double, double> goal_position_;
    std::vector<std::vector<std::vector<int>>> drone_view_; // terrain, fire_status split at the first dimension
    std::vector<std::vector<int>> exploration_map_;
    std::vector<std::vector<double>> fire_map_;
    std::pair<double, double> orientation_vector_; // x, y
    double DiscretizeOutput(double netout);
};

#endif //ROSHAN_DRONE_STATE_H
