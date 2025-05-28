//
// Created by nex on 15.07.23.
//

#ifndef ROSHAN_AGENT_STATE_H
#define ROSHAN_AGENT_STATE_H

#include <utility>
#include <vector>
#include <cmath>
#include <array>
#include <iostream>
#include <numeric>
#include <algorithm>
#include "firespin/utils.h"
#include "state.h"


class AgentState : public State{
public:
    //* Constructor for the DroneState
    //* @param velocity_vector The velocity of the Agent
    //* @param max_speed The maximum speed of the Agent
    //* @param drone_view The view of the Agent
    //* @param exploration_map The exploration map of the Agent
    //* @param fire_map The fire map of the Agent
    //* @param map_dimensions The dimensions of the map (used for normalization)
    //* @param position The position of the Agent
    //* @param goal_position The goal position of the Agent
    //* @param water_dispense The water dispense Action of the Agent
    //* @param cell_size The size of the cells in the simulation (used for normalization)
    explicit AgentState() = default;

    //** Setter Functions for the states (better readability, than a massive Constructor) **//
    void SetVelocity(std::pair<double, double> velocity) { velocity_ = velocity; }
    void SetDroneView(std::shared_ptr<const std::vector<std::vector<std::vector<int>>>> drone_view) { drone_view_ = std::move(drone_view); }
    void SetTotalDroneView(std::shared_ptr<const std::vector<std::vector<double>>> total_drone_view) { total_drone_view_ = std::move(total_drone_view); }
    void SetExplorationMap(std::shared_ptr<const std::vector<std::vector<int>>> exploration_map) { exploration_map_ = std::move(exploration_map); }
    void SetFireMap(std::shared_ptr<const std::vector<std::vector<double>>> fire_map) { fire_map_ = std::move(fire_map); }
    void SetPosition(std::pair<double, double> position) { position_ = position; }
    void SetGoalPosition(std::pair<double, double> goal_position) { goal_position_ = goal_position; }
    void SetWaterDispense(int water_dispense) { water_dispense_ = water_dispense; }
    void SetMapDimensions(std::pair<double, double> map_dimensions) { map_dimensions_ = map_dimensions; }
    void SetCellSize(double cell_size) { cell_size_ = cell_size; }
    void SetMultipleTotalDroneView(std::vector<std::shared_ptr<const std::vector<std::vector<double>>>> total_drone_view) { multiple_total_drone_views_ = std::move(total_drone_view); }

    //** These functions are for the states **//
    [[nodiscard]] std::vector<std::vector<std::vector<double>>> GetMultipleTotalDroneView() const {
        std::vector<std::vector<std::vector<double>>> drone_views;
        drone_views.reserve(multiple_total_drone_views_.size());
        for (const auto& view : multiple_total_drone_views_) {
            drone_views.push_back(*view);
        }
        return drone_views;
    }


    //* Returns the velocity of the drone
    //* @return std::pair<double, double> The velocity of the drone
    [[nodiscard]] std::pair<double, double> GetVelocity() const { return velocity_; }

    //* Returns the normalized velocity of the drone. Normalized by the max speed.
    //* @return std::pair<double, double> The normalized velocity of the drone
    [[nodiscard]] std::pair<double, double> GetVelocityNorm() const;

    //* Returns the Terrain of this State
    //* @return std::vector<std::vector<int>> TerrainView around the Agent.
    [[nodiscard]] std::vector<std::vector<int>> GetTerrainView() const { return (*drone_view_)[0]; }

    //* Returns the Fire Status of this State. 1 indicates fire, 0 indicates no fire.
    //* @return std::vector<std::vector<int>> FireView around the Agent.
    [[nodiscard]] std::vector<std::vector<int>> GetFireView() const { return (*drone_view_)[0]; }
    //* The Fire Map is a 2D vector with all discovered fires on the whole map.
    //* @return std::vector<std::vector<double>> The FireMap of the whole Environment of the Agent.
    [[nodiscard]] std::vector<std::vector<double>> GetFireMap() const { return *fire_map_; }

    //* Returns the Exploration Map of this State
    //* The Exploration Map is a 2D vector with all discovered cells on the whole map.
    //* @return std::vector<std::vector<int>> The ExplorationMap of the whole Environment of the Agent.
    [[nodiscard]] std::vector<std::vector<int>> GetExplorationMap() const { return *exploration_map_; }

    //* Returns the WaterDispense Action of this State
    //* @return int The WaterDispense Action of the Agent.
    [[nodiscard]] int GetWaterDispense() const { return water_dispense_; }

    //* Returns the Exploration Map of this State.
    //* The Map is normalized by the maximum exploration time.
    //* @return std::vector<std::vector<double>> The ExplorationMap of the whole Environment of the Agent.
    [[nodiscard]] std::vector<std::vector<double>> GetExplorationMapNorm() const;

    //* Returns the Exploration Map Scalar of this State.
    //* The Exploration Map Scalar is the sum of all values in the Exploration Map.
    //* @return double The ExplorationMap Scalar of the whole Environment of the Agent.
    [[nodiscard]] double GetExplorationMapScalar() const;

    //* Returns the Position of this State
    //* The Position is normalized by the map dimensions and cell_size. It is in the range of [-1, 1]
    //* @return std::pair<double, double> The normalized Position of the Agent.
    [[nodiscard]] std::pair<double, double> GetPositionNormAroundCenter() const;

    //* Returns the Grid Position of this State
    //* The Grid Position is the position of the Agent in the grid.
    //* The Grid Position is a continous value between the grid cells.
    //* @return std::pair<double, double> The continous Grid Position of the Agent.
    [[nodiscard]] std::pair<double, double> GetGridPositionDouble() const;

    //* Returns the Grid Position of this State
    //* The Grid Position is the position of the Agent in the grid.
    //* The Grid Position is a continous value between the grid cells.
    //* The Grid Position is normalized by the map dimensions. It is in the range of [0, 1]
    //* @return std::pair<double, double> The normalized continous Grid Position of the Agent.
    [[nodiscard]] std::pair<double, double> GetGridPositionDoubleNorm() const;

    //* Returns the Goal Position of this State
    //* The Goal Position is the position the Agent wants to reach.
    //* @return std::pair<double, double> The Goal Position of the Agent.
    [[nodiscard]] std::pair<double, double> GetGoalPosition() const { return goal_position_; }

    //* Returns the Goal Position of this State
    //* The Goal Position is the position the Agent wants to reach.
    //* The Goal Position is normalized by the map dimensions. It is in the range of [0, 1]
    //* @return std::pair<double, double> The normalized Goal Position of the Agent.
    [[nodiscard]] std::pair<double, double> GetGoalPositionNorm() const;

    //* Returns the Delta Goal of this State
    //* The Delta Goal is the difference between the Goal Position and the current Position.
    //* @return std::pair<double, double> The Delta Goal of the Agent.
    [[nodiscard]] std::pair<double, double> GetDeltaGoal() const;

    //* Returns the Orientation to the Goal of this State
    //* The Orientation to the Goal is the normalized vector from the Agent to the Goal.
    //* @return std::pair<double, double> The Orientation to the Goal of the Agent.
    [[nodiscard]] std::pair<double, double> GetOrientationToGoal() const;

    //* Returns the Outside Area Counter of this State
    //* The Outside Area Counter is the number of cells the Agent is seeing in his drone view that are outside the map.
    //* @return int The Outside Area Counter of the Agent.
    [[nodiscard]] int CountOutsideArea();

    //* Returns the normalized Drone View of this State
    //* The Drone View is the view the Agent sees around him. It is split into Terrain and Fire Status.
    //* The Terrain is the first element of the vector, the Fire Status is the second element of the vector.
    //* The Drone View is normalized by the maximum value of the Terrain and Fire Status
    //* @return std::vector<std::vector<std::vector<int>>> The normalized Drone View of the Agent.
    [[nodiscard]] std::vector<std::vector<std::vector<int>>> GetDroneViewNorm();

    //* Returns the normalized Distance to the next Boundary of this State
    //* The Distance to the next Boundary is the distance of the Agent to the next boundary of the map.
    //* The Distance is normalized by the maximum distance of the map.
    //* @return double The normalized Distance to the next Boundary of the Agent.
    [[nodiscard]] double GetDistanceToNearestBoundaryNorm() const;

    //* Returns the Position of the Agent in the Exploration Map
    //* The Position is the position of the Agent in the actual Map rescaled to the dimensions of the Exploration Map.
    //* @return std::pair<double, double> The Position of the Agent in the Exploration Map.
    [[nodiscard]] std::pair<double, double> GetPositionInExplorationMap() const;

    [[nodiscard]] std::vector<std::vector<double>> GetTotalDroneView() const { return *total_drone_view_; }
    [[nodiscard]] std::shared_ptr<const std::vector<std::vector<double>>> GetTotalDroneViewPtr() const { return total_drone_view_; }

    //** These functions are only for Python Debugger Visibility **//
    [[nodiscard]] std::pair<double, double> get_velocity() const { return velocity_; }
    [[nodiscard]] std::vector<std::vector<std::vector<int>>> get_drone_view() const { return *drone_view_; }
    [[nodiscard]] std::vector<std::vector<double>> get_total_drone_view() const { return *total_drone_view_; }
    [[nodiscard]] std::vector<std::vector<int>> get_map() const { return *exploration_map_; }
    [[nodiscard]] std::vector<std::vector<double>> get_fire_map() const { return *fire_map_; }
    [[nodiscard]] std::pair<int, int> get_position() const { return position_; }
    [[nodiscard]] std::pair<double, double> get_orientation_vector() const { return orientation_vector_; }
private:
    //* State Value for the velocity of an Agent in x and y direction
    std::pair<double, double> velocity_;
    //* State Value weather the Agent dispensed water
    int water_dispense_;
    std::pair<double, double> map_dimensions_;
    double cell_size_;
    std::pair<double, double> position_; // x, y
    std::pair<double, double> goal_position_;
    std::pair<double, double> orientation_vector_; // x, y
    std::shared_ptr<const std::vector<std::vector<std::vector<int>>>> drone_view_;
    std::vector<std::shared_ptr<const std::vector<std::vector<double>>>> multiple_total_drone_views_;
    std::shared_ptr<const std::vector<std::vector<double>>> total_drone_view_;
    std::shared_ptr<const std::vector<std::vector<int>>> exploration_map_;
    std::shared_ptr<const std::vector<std::vector<double>>> fire_map_;
};

#endif //ROSHAN_AGENT_STATE_H
