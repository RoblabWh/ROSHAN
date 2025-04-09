//
// Created by nex on 08.04.25.
//

#ifndef ROSHAN_FLY_AGENT_H
#define ROSHAN_FLY_AGENT_H


#include "agent.h"
#include "reinforcementlearning/actions/fly_action.h"

#include <utility>
#include <SDL.h>
#include <deque>
#include <memory>
#include "reinforcementlearning/texturerenderer.h"
#include "firespin/model_parameters.h"
#include "firespin/firemodel_gridmap.h"
#include "firespin/rendering/firemodel_renderer.h"
#include "src/utils.h"
#include "agent.h"
#include "drone_state.h"
#include "reinforcementlearning/actions/fly_action.h"
#include "reinforcementlearning/actions/explore_action.h"

class GridMap;
class FireModelRenderer;

class FlyAgent : public Agent {
public:
    explicit FlyAgent(const std::shared_ptr<GridMap>& grid_map, FireModelParameters &parameters, int id, int timesteps);
    void ExecuteAction(std::shared_ptr<Action> action, std::string hierarchy_type, std::shared_ptr<GridMap> gridMap) override {
        action->ExecuteOn(shared_from_this(), hierarchy_type, gridMap);
    }
    std::vector<bool> GetTerminalStates(bool eval_mode, const std::shared_ptr<GridMap>& grid_map, int total_env_steps);


    void Initialize(int mode, const std::shared_ptr<GridMap>& grid_map, const std::shared_ptr<FireModelRenderer>& model_renderer, const std::string& rl_mode, double rng_number);
    void InitializeGridMap(const std::shared_ptr<GridMap>& grid_map);

    void PerformFly(FlyAction* action, const std::string& hierarchy_type, const std::shared_ptr<GridMap>& gridMap);

    void StepReset() override {
        this->SetExploreDifference(0);
        this->SetPerformedHierarchyAction(false);
    }

    std::deque<std::shared_ptr<State>> GetObservations();

    // TODO Tidy UP !!!
    std::deque<DroneState> GetStates() { return drone_states_; }
    void SetRenderer(SDL_Renderer* renderer) { renderer_ = TextureRenderer(renderer, "../assets/drone.png"); }
    void SetRenderer2(SDL_Renderer* renderer) { renderer2_ = TextureRenderer(renderer, "../assets/Xmarksthespot.png"); }
    void UpdateStates(const std::shared_ptr<GridMap>& grid_map, std::pair<double, double> velocity_vector, int water_dispense);
    std::pair<double, double> MovementStep(double netout_x, double netout_y);
    bool DispenseWaterCertain(const std::shared_ptr<GridMap>& grid_map);
    void DispenseWater(const std::shared_ptr<GridMap>& grid_map, int water_dispense);
    void IncrementOutOfAreaCounter() { out_of_area_counter_++; }
    void ResetOutOfAreaCounter() { out_of_area_counter_ = 0; }
    int DroneSeesFire();
    int GetId() const { return id_; }
    void SetActive(bool active) { active_ = active; }
    void Render(std::pair<int, int> position, std::pair<int, int> goal_position_screen, int size);
    double FindNearestFireDistance();
    void Step(double speed_x, double speed_y, const std::shared_ptr<GridMap>& gridmap);
    void FlyPolicy(const std::shared_ptr<GridMap> &gridmap);
//    void OnFlyAction(const std::shared_ptr<FlyAction>& action, const std::shared_ptr<GridMap>& gridMap) override;
//    void OnExploreAction(std::shared_ptr<ExploreAction> action, std::shared_ptr<GridMap> gridMap) override;

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
    void ModifyReward(double reward) {
        reward_components_["Intrinsic Reward"] = reward;
        rewards_.ModifyLast(static_cast<float>(reward)); }
    void SetPerformedHierarchyAction(bool did_hierachy_step) { did_hierarchy_step = did_hierachy_step; }
    bool GetPerformedHierarchyAction() const override { return did_hierarchy_step; }
    void PushEpisodeReward();
    double CalculateReward() override;
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
    bool did_hierarchy_step = false;
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
    bool last_terminal_state_ = false;
    int last_total_env_steps_ = 0;
    std::pair<double, double> velocity_; // angular & linear
    TextureRenderer renderer_;
    TextureRenderer renderer2_;

    // Rewards Collection for Debugging!
    bool explored_fires_equals_actual_fires_ = false;
    std::string agent_type_;
    CircularBuffer<float> rewards_;
    std::unordered_map<std::string, double> reward_components_;

    double CalculateFlyReward(bool terminal_state, int total_env_steps);
    double CalculateExploreReward(bool terminal_state, int total_env_steps);

    std::vector<bool> TerminalFly(bool eval_mode, const std::shared_ptr<GridMap> &grid_map, int total_env_steps) const;
    std::vector<bool> TerminalExplore(bool eval_mode, const std::shared_ptr<GridMap> &grid_map, int total_env_steps);

};


#endif //ROSHAN_FLY_AGENT_H
