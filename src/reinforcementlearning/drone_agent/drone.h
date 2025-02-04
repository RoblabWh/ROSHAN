//
// Created by nex on 13.07.23.
//

#ifndef ROSHAN_DRONE_H
#define ROSHAN_DRONE_H

#define DEBUG_REWARD_NO

#include <utility>
#include <SDL.h>
#include <deque>
#include <memory>
#include "reinforcementlearning/texturerenderer.h"
#include "firespin/model_parameters.h"
#include "firespin/firemodel_gridmap.h"
#include "src/utils.h"
#include "agent.h"
#include "drone_state.h"
#include "fly_action.h"
#include "explore_action.h"

// TODO Remove circular dependency
class GridMap;

class DroneAgent: public Agent {
public:
    explicit DroneAgent(const std::shared_ptr<GridMap>& grid_map, std::string agent_type, FireModelParameters &parameters, int id);
    ~DroneAgent() override = default;
    std::deque<DroneState> GetStates() { return drone_states_; }
    void SetRenderer(std::shared_ptr<SDL_Renderer> renderer) { renderer_ = TextureRenderer(std::move(renderer), "../assets/drone.png"); }
    void SetRenderer2(std::shared_ptr<SDL_Renderer> renderer) { renderer2_ = TextureRenderer(std::move(renderer), "../assets/Xmarksthespot.png"); }
    void UpdateStates(const std::shared_ptr<GridMap>& grid_map, std::pair<double, double> velocity_vector, const std::vector<std::vector<std::vector<int>>>& drone_view, int water_dispense);
    std::pair<double, double> MovementStep(double netout_x, double netout_y);
    bool DispenseWaterCertain(const std::shared_ptr<GridMap>& grid_map);
    void DispenseWater(const std::shared_ptr<GridMap>& grid_map, int water_dispense);
    void IncrementOutOfAreaCounter() { out_of_area_counter_++; }
    void ResetOutOfAreaCounter() { out_of_area_counter_ = 0; }
    int DroneSeesFire();
    int GetId() const { return id_; }
    void SetActive(bool active) { active_ = active; }
    void Render(std::pair<int, int> position, std::pair<int, int> goal_position_screen, int size);
    void Initialize(const std::shared_ptr<GridMap>& grid_map);
    double FindNearestFireDistance();
    void Step(double speed_x, double speed_y, const std::shared_ptr<GridMap>& gridmap);
    void OnFlyAction(std::shared_ptr<FlyAction> action, std::shared_ptr<GridMap> gridMap) override;
    void OnExploreAction(std::shared_ptr<ExploreAction> action, std::shared_ptr<GridMap> gridMap) override;

    [[maybe_unused]] bool IsAlive() const { return is_alive_; }

    //** Getter-Setter **//
    DroneState GetLastState() { return drone_states_[0]; }
    int GetViewRange() const { return view_range_; }
    int GetOutOfAreaCounter() const { return out_of_area_counter_; }
    double GetMaxDistanceFromMap() const { return max_distance_from_map_; }
    double GetLastDistanceToGoal() const { return last_distance_to_goal_; }
    bool GetDroneInGrid() const { return drone_in_grid_; }
    bool GetDispensedWater() const { return dispensed_water_; }
    void SetExtinguishedFire(bool extinguished) { extinguished_fire_ = extinguished; }
    bool GetExtinguishedFire() const { return extinguished_fire_; }
    bool GetExtinguishedLastFire() const {return extinguished_last_fire_; }
    int GetExploreDifference() const { return explore_difference_; }
    double GetLastDistanceToFire() const { return last_distance_to_fire_; }
    int GetLastNearFires() const { return last_near_fires_; }
    CircularBuffer<float> GetEpisodeRewards() { return rewards_; }
    std::pair<int, int> GetGridPosition();
    std::pair<double, double> GetGridPositionDouble();
    std::pair<double, double> GetRealPosition();
    std::pair<double, double> GetGoalPosition() { return goal_position_; }
    double GetDistanceToGoal();
    bool GetReachedGoal() const { return reached_goal_; }
    std::pair<int, int> GetGoalPositionInt() const { return std::make_pair((int)goal_position_.first, (int)goal_position_.second); }
    double GetWaterCapacity() const { return water_capacity_; }
    int GetPolicyType() const { return policy_type_; }
    void SetPolicyType(int policy_type) { policy_type_ = policy_type; }
    void SetReachedGoal(bool reached_goal) { reached_goal_ = reached_goal; }
    void SetWaterCapacity(double water_capacity) { water_capacity_ = water_capacity; }
    void SetLastDistanceToFire(double distance) { last_distance_to_fire_ = distance; }
    void SetLastNearFires(int near_fires) { last_near_fires_ = near_fires; }
    void SetGoalPosition(std::pair<double, double> goal_position) { goal_position_ = goal_position; }
    void SetLastDistanceToGoal(double distance) { last_distance_to_goal_ = distance; }
    void SetDispenedWater(bool dispensed) { dispensed_water_ = dispensed; }
    void SetExploreDifference(int explore_difference) { explore_difference_ = explore_difference; }
    void SetReward(double reward) { rewards_.put(static_cast<float>(reward));}
    void PushEpisodeReward();
    std::pair<bool, bool> IsTerminal(bool eval_mode, const std::shared_ptr<GridMap>& grid_map, int total_env_steps) const;
    double CalculateReward(bool terminal_state, int total_env_steps);
    std::unordered_map<std::string, double> GetRewardComponents() { return reward_components_; }
private:
    std::pair<double, double> MoveByXYVel(double speed_x, double speed_y);
    void CalcMaxDistanceFromMap();
    static void LogRewards(const std::unordered_map<std::string, double>& rewards) ;
    static double ComputeTotalReward(const std::unordered_map<std::string, double>& rewards);
    int id_;
    FireModelParameters &parameters_;
    std::deque<DroneState> drone_states_;
    std::pair<int, int> map_dimensions_;
    std::pair<double, double> position_; // x, y in (m)
    std::pair<double, double> goal_position_;
    bool is_alive_;
    int explore_difference_{};
    int view_range_;
    int time_steps_;
    bool drone_in_grid_ = true;
    double max_distance_from_map_{};
    int out_of_area_counter_;
    double water_capacity_;
    int policy_type_{}; // 0 = extinguish fire, 1 = fly to groundstation, 2 = recharge, 3 = explore
    bool reached_goal_ = false;
    bool dispensed_water_{};
    bool extinguished_fire_{};
    bool extinguished_last_fire_ = false;
    bool active_ = false;
    double last_distance_to_fire_{};
    double last_distance_to_goal_{};
    int last_near_fires_{};
    std::pair<double, double> velocity_; // angular & linear
    TextureRenderer renderer_;
    TextureRenderer renderer2_;

    // Rewards Collection for Debugging!
    std::string agent_type_;
    CircularBuffer<float> rewards_;
    std::unordered_map<std::string, double> reward_components_;

    double CalculateFlyReward(bool terminal_state, int total_env_steps);
    double CalculateExploreReward(bool terminal_state, int total_env_steps);
};


#endif //ROSHAN_DRONE_H
