//
// Created by nex on 13.07.23.
//

#ifndef ROSHAN_DRONE_H
#define ROSHAN_DRONE_H

#include <utility>
#include <SDL.h>
#include <deque>
#include <memory>
#include "reinforcementlearning/drone_agent/rendering/DroneRenderer.h"
#include "src/models/firespin/model_parameters.h"
#include "src/models/firespin/firemodel_gridmap.h"
#include "drone_state.h"

// TODO Remove circular dependency
class GridMap;

class DroneAgent: public std::enable_shared_from_this<DroneAgent> {
public:
    explicit DroneAgent(std::pair<int, int> point, FireModelParameters &parameters, int id);
    ~DroneAgent() = default;
    std::deque<DroneState> GetStates() { return drone_states_; }
    void SetRenderer(std::shared_ptr<SDL_Renderer> renderer) { renderer_ = DroneRenderer(std::move(renderer)); }
    void UpdateStates(GridMap &grid_map, std::pair<double, double> velocity_vector, const std::vector<std::vector<std::vector<int>>>& drone_view, int water_dispense);
    std::pair<double, double> Step(double netout_x, double netout_y);
    bool DispenseWaterCertain(GridMap &grid_map);
    void DispenseWater(GridMap &grid_map, int water_dispense);
    std::pair<int, int> GetGridPosition();
    std::pair<double, double> GetGridPositionDouble();
    std::pair<double, double> GetRealPosition();
    std::pair<double, double> GetGoalPosition() { return goal_position_; }
    double GetDistanceToGoal();
    void SetReachedGoal(bool reached_goal) { reached_goal_ = reached_goal; }
    bool GetReachedGoal() { return reached_goal_; }
    std::pair<int, int> GetGoalPositionInt() { return std::make_pair((int)goal_position_.first, (int)goal_position_.second); }
    void IncrementOutOfAreaCounter() { out_of_area_counter_++; }
    void ResetOutOfAreaCounter() { out_of_area_counter_ = 0; }
    int GetOutOfAreaCounter() { return out_of_area_counter_; }
    DroneState GetLastState() { return drone_states_[0]; }
    double GetMaxDistanceFromMap() { return max_distance_from_map_; }
    void SetLastDistanceToFire(double distance) { last_distance_to_fire_ = distance; }
    void SetLastNearFires(int near_fires) { last_near_fires_ = near_fires; }
    void SetGoalPosition(std::pair<double, double> goal_position) { goal_position_ = goal_position; }
    void SetLastDistanceToGoal(double distance) { last_distance_to_goal_ = distance; }
    double GetLastDistanceToGoal() { return last_distance_to_goal_; }
    bool GetDroneInGrid() { return drone_in_grid_; }
    void SetDispenedWater(bool dispensed) { dispensed_water_ = dispensed; }
    bool GetDispensedWater() { return dispensed_water_; }
    void SetExtinguishedFire(bool extinguished) { extinguished_fire_ = extinguished; }
    bool GetExtinguishedFire() { return extinguished_fire_; }
    bool GetExtinguishedLastFire() {return extinguished_last_fire_; }
    void SetExploreDifference(int explore_difference) { explore_difference_ = explore_difference; }
    int GetExploreDifference() { return explore_difference_; }
    double GetLastDistanceToFire() { return last_distance_to_fire_; }
    int GetLastNearFires() { return last_near_fires_; }
    int DroneSeesFire();
    int GetId() const { return id_; }
    int GetViewRange() const { return view_range_; }
    void Render(std::pair<int, int> position, int size);
    void Initialize(GridMap &grid_map);
    double FindNearestFireDistance();
private:
    std::pair<double, double> MoveByXYVel(double speed_x, double speed_y);
    std::pair<double, double> MoveByAngle(double netout_speed, double netout_angle);
    void CalcMaxDistanceFromMap();
    int id_;
    FireModelParameters &parameters_;
    std::deque<DroneState> drone_states_;
    std::pair<int, int> map_dimensions_;
    std::pair<double, double> position_; // x, y in (m)
    std::pair<double, double> goal_position_;
    int explore_difference_;
    int view_range_;
    int time_steps_;
    bool drone_in_grid_;
    double max_distance_from_map_;
    int out_of_area_counter_;
    bool reached_goal_ = false;
    bool dispensed_water_;
    bool extinguished_fire_;
    bool extinguished_last_fire_ = false;
    double last_distance_to_fire_{};
    double last_distance_to_goal_{};
    int last_near_fires_{};
    std::pair<double, double> velocity_; // angular & linear
    DroneRenderer renderer_;
};


#endif //ROSHAN_DRONE_H
