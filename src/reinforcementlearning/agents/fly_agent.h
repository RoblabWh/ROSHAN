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
#include <random>
#include "reinforcementlearning/texturerenderer.h"
#include "firespin/model_parameters.h"
#include "firespin/firemodel_gridmap.h"
#include "firespin/rendering/firemodel_renderer.h"
#include "src/utils.h"
#include "agent.h"
#include "agent_state.h"
#include "reinforcementlearning/actions/fly_action.h"
#include "reinforcementlearning/actions/explore_action.h"

class GridMap;
class FireModelRenderer;

class FlyAgent : public Agent {
public:
    explicit FlyAgent(FireModelParameters &parameters, int id, int time_steps);

    void ExecuteAction(std::shared_ptr<Action> action, std::string hierarchy_type, std::shared_ptr<GridMap> gridMap) override {
        action->ExecuteOn(shared_from_this(), hierarchy_type, gridMap);
    }
    AgentTerminal GetTerminalStates(bool eval_mode, const std::shared_ptr<GridMap>& grid_map, int env_steps_remaining) override;

    void Initialize(int mode,
                    double speed,
                    int view_range,
                    const std::shared_ptr<GridMap>& grid_map,
                    const std::shared_ptr<FireModelRenderer>& model_renderer);
    void Reset(Mode mode,
               const std::shared_ptr<GridMap>& grid_map,
               const std::shared_ptr<FireModelRenderer>& model_renderer) override;

    void InitializeFlyAgentStates(const std::shared_ptr<GridMap>& grid_map);

    void PerformFly(FlyAction* action, const std::string& hierarchy_type, const std::shared_ptr<GridMap>& gridMap);

    void StepReset() override {
        did_hierarchy_step = false;
    }

    double CalculateReward(const std::shared_ptr<GridMap>& grid_map) override;

    // TODO Tidy UP !!!

    //Rendering
    void SetDroneTextureRenderer(SDL_Renderer* renderer, const std::string& texture_path) { drone_texture_renderer_ = TextureRenderer(renderer, texture_path.c_str()); }
    void SetGoalTextureRenderer(SDL_Renderer* renderer) { goal_texture_renderer_ = TextureRenderer(renderer, "../assets/goal.png"); }
    //void Render(std::pair<int, int> position, std::pair<int, int> goal_position_screen, int size);
    void Render(const FireModelCamera& camera);
    void SetRender(bool should_render) {
        if (should_render_ && !should_render) {
            trail_.clear(); // Clear the trail when rendering is disabled
        }
        should_render_ = should_render; }
    void SetActive(bool active) { active_ = active; }
    void SetTrailLength(int length) { trail_length_ = length; }
    void AppendTrail(const std::pair<int, int> &position);
    std::deque<std::pair<double, double>> GetCameraTrail(const FireModelCamera &camera) const;

    // Fly Agent functions that other Classes need in some way too
    void Step(double speed_x, double speed_y, const std::shared_ptr<GridMap>& gridmap);

    //Fly Agent specific public
    void DispenseWater(const std::shared_ptr<GridMap>& grid_map, int water_dispense);
    bool DispenseWaterCertain(const std::shared_ptr<GridMap>& grid_map);
    int GetViewRange() const { return view_range_; }
    int GetOutOfAreaCounter() const { return out_of_area_counter_; }
    std::pair<double, double> CalculateLocalGoal(double goal_x, double goal_y);

    //** Getter **//
    bool GetDroneInGrid() const { return drone_in_grid_; }
    std::pair<int, int> GetGridPosition();
    double GetDistanceToGoal();
    std::pair<double, double> GetGridPositionDouble() { return std::make_pair<double, double>(position_.first / parameters_.GetCellSize(), position_.second / parameters_.GetCellSize()); }//{ return this->GetLastState().GetGridPositionDouble(); }
    std::pair<double, double> GetRealPosition() { return position_; }
    std::pair<double, double> GetGoalPosition() { return goal_position_; }
    std::string GetAgentSubType() const { return agent_sub_type_; }
    double GetDroneSize() const { return parameters_.drone_size_ / parameters_.GetCellSize(); } // like GetGridPositionDouble
    bool GetDistanceToNearestBoundaryNorm(int rows, int cols, double view_range_half, std::vector<double>& out_norm);


    std::pair<int, int> GetGoalPositionInt() const { return std::make_pair((int)goal_position_.first, (int)goal_position_.second); }
    int GetNewlyExploredCells() const { return newly_explored_cells_; }
    int GetRevisitedCells() const { return last_step_total_revisited_cells_of_all_agents_; }
    int GetAndResetNewlyExploredCells() {
        auto nec = newly_explored_cells_;
        newly_explored_cells_ = 0;
        return nec;
    }
    bool GetExtinguishedLastFire() const { return extinguished_last_fire_; }
    std::pair<double, double> GetVelocityVecNorm() const { return {vel_vector_.first / max_speed_.first / norm_scale_, vel_vector_.second / max_speed_.second / norm_scale_}; }
    //** Setter **//
    void SetGoalPosition(std::pair<double, double> goal_position) { goal_position_ = goal_position; }
    void SetPosition(std::pair<double, double> point) { position_ = std::make_pair((point.first) * parameters_.GetCellSize(), (point.second) * parameters_.GetCellSize()); }
    void SetRevisitedCells(int revisited_cells) { last_step_total_revisited_cells_of_all_agents_ = revisited_cells; }
    void SetAgentSubType(const std::string& agent_type) { agent_sub_type_ = agent_type; }
    void SetSpeed(const std::pair<double, double> &speed) { max_speed_ = speed; }
    void SetCollision(bool collision) { collision_occurred_ = collision; }
    void ClearDistances() { distance_to_other_agents_.clear(); }
    void ClearMask() { distance_mask_.clear(); }
    void AppendDistance(const std::vector<double> &dist) {distance_to_other_agents_.push_back(dist);}
    void AppendMask(bool mask) {distance_mask_.push_back(mask);}
    std::vector<std::vector<double>> GetDistancesToOtherAgents() const { return distance_to_other_agents_; }
    std::vector<bool> GetDistanceMask() const { return distance_mask_; }
private:
    //Fly Agent specific
    void FlyPolicy(const std::shared_ptr<GridMap> &gridmap);
    std::pair<double, double> MovementStep(double netout_x, double netout_y);

    void CalcMaxDistanceFromMap();
    static bool almostEqual(const std::pair<double, double>& p1, const std::pair<double, double>& p2, double epsilon = 0.25) {
        return std::fabs(p1.first - p2.first) < epsilon && std::fabs(p1.second - p2.second) < epsilon;
    }

    double last_distance_to_goal_{};
    bool drone_in_grid_ = true; //* Is the Drone in the Grid?
    int view_range_{}; //* View Range of the Agent in Grid Cells (10m each)
    std::pair<double, double> max_speed_{}; //* Value for the maximum velocity of an Agent in x and y direction

    // Possibly Deprecated
    double FindNearestFireDistance();
    int DroneSeesFire();
    bool dispensed_water_{};

    std::pair<double, double> position_; // x, y in (m)
    std::pair<double, double> vel_vector_;
    std::pair<double, double> goal_position_;
    std::vector<std::vector<double>> distance_to_other_agents_;
    std::vector<bool> distance_mask_;
    int newly_explored_cells_{};
    int out_of_area_counter_;
    bool collision_occurred_ = false;
    double water_capacity_;
    bool extinguished_fire_ = false;
    enum policy_types {EXTINGUISH_FIRE, FLY_TO_GROUNDSTATION, RECHARGE, EXPLORE};
    int policy_type_ = EXTINGUISH_FIRE;
    bool extinguished_last_fire_ = false;
    bool active_ = false;

    TextureRenderer drone_texture_renderer_;
    TextureRenderer goal_texture_renderer_;
    std::deque<std::pair<double, double>> trail_;
    int trail_length_ = 50;
    double norm_scale_{};

    // Possibly Deprecated
    double max_distance_from_map_{};
    int last_near_fires_{}; // Needs to be populated with this->DroneSeesFire() possible future use
    double last_distance_to_fire_{}; // Needs to be populated with this->FindNearestFireDistance() possible future use
    bool explored_fires_equals_actual_fires_ = false;

    // Fly Agent Specific
    //* Computes the new velocity based on the netout of the neural network
    //* and the internal logic how the velocity is updated, this function ensures max speed is not exceeded
    //* @param speed_x The netout of the neural network for the x velocity
    //* @param speed_y The netout of the neural network for the y velocity
    //* @return std::pair<double, double> New Velocity
    std::pair<double, double> GetNewVelocity(double next_speed_x, double next_speed_y) const;

    std::shared_ptr<AgentState> BuildAgentState(const std::shared_ptr<GridMap> &grid_map) override;

    // Currently just for Debugging
    int last_step_total_revisited_cells_of_all_agents_{};
    bool should_render_ = true; // If false, the agent will not render anything
};

#endif //ROSHAN_FLY_AGENT_H
