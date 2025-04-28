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
    std::vector<bool> GetTerminalStates(bool eval_mode, const std::shared_ptr<GridMap>& grid_map, int env_steps_remaining) override;

    void Initialize(int mode, const std::shared_ptr<GridMap>& grid_map, const std::shared_ptr<FireModelRenderer>& model_renderer, const std::string& rl_mode);
    void InitializeFlyAgentStates(const std::shared_ptr<GridMap>& grid_map);

    void PerformFly(FlyAction* action, const std::string& hierarchy_type, const std::shared_ptr<GridMap>& gridMap);

    void StepReset() override {
        did_hierarchy_step = false;
    }

    double CalculateReward() override;

    // TODO Tidy UP !!!

    //Rendering
    void SetDroneTextureRenderer(SDL_Renderer* renderer) { drone_texture_renderer_ = TextureRenderer(renderer, "../assets/drone.png"); }
    void SetGoalTextureRenderer(SDL_Renderer* renderer) { goal_texture_renderer_ = TextureRenderer(renderer, "../assets/Xmarksthespot.png"); }
    void Render(std::pair<int, int> position, std::pair<int, int> goal_position_screen, int size);
    void SetActive(bool active) { active_ = active; }

    // Fly Agent functions that other Classes need in some way too
    void Step(double speed_x, double speed_y, const std::shared_ptr<GridMap>& gridmap);

    //Fly Agent specific public
    void DispenseWater(const std::shared_ptr<GridMap>& grid_map, int water_dispense);
    int GetViewRange() const { return view_range_; }
    int GetOutOfAreaCounter() const { return out_of_area_counter_; }
    std::pair<double, double> CalculateLocalGoal(double goal_x, double goal_y);

    //** Getter **//
    bool GetDroneInGrid() const { return drone_in_grid_; }
    std::pair<int, int> GetGridPosition();
    double GetDistanceToGoal();
    std::pair<double, double> GetGridPositionDouble() { return this->GetLastState().GetGridPositionDouble(); };
    std::pair<double, double> GetRealPosition() { return position_; };
    std::pair<double, double> GetGoalPosition() { return goal_position_; }

    std::pair<int, int> GetGoalPositionInt() const { return std::make_pair((int)goal_position_.first, (int)goal_position_.second); }
    int GetNewlyExploredCells() const { return newly_explored_cells_; }
    int GetRevisitedCells() const { return last_step_total_revisited_cells_of_all_agents_; }
    int GetAndResetNewlyExploredCells() {
        auto nec = newly_explored_cells_;
        newly_explored_cells_ = 0;
        return nec;
    }
    //** Setter **//
    void SetGoalPosition(std::pair<double, double> goal_position) { goal_position_ = goal_position; }
    void SetPosition(std::pair<int, int> point) { position_ = std::make_pair((point.first + 0.5) * parameters_.GetCellSize(), (point.second + 0.5) * parameters_.GetCellSize()); }
    void SetRevisitedCells(int revisited_cells) { last_step_total_revisited_cells_of_all_agents_ = revisited_cells; }
private:
    void UpdateStates(const std::shared_ptr<GridMap>& grid_map, std::pair<double, double> velocity_vector, int water_dispense);

    //Fly Agent specific
    bool DispenseWaterCertain(const std::shared_ptr<GridMap>& grid_map);
    void FlyPolicy(const std::shared_ptr<GridMap> &gridmap);
    std::pair<double, double> MovementStep(double netout_x, double netout_y);
    void CalcMaxDistanceFromMap();
    bool almostEqual(const std::pair<double, double>& p1, const std::pair<double, double>& p2, double epsilon = 1e-1) {
        return std::fabs(p1.first - p2.first) < epsilon && std::fabs(p1.second - p2.second) < epsilon;
    }

    double last_distance_to_goal_{};
    bool drone_in_grid_ = true; //* Is the Drone in the Grid?
    int view_range_; //* View Range of the Agent in Grid Cells (10m each)
    std::pair<double, double> max_speed_; //* Value for the maximum velocity of an Agent in x and y direction

    // Possibly Deprecated
    double FindNearestFireDistance();
    int DroneSeesFire();
    bool dispensed_water_{};

    std::pair<double, double> position_; // x, y in (m)
    std::pair<double, double> goal_position_;
    int newly_explored_cells_{};
    int out_of_area_counter_;
    double water_capacity_;
    int policy_type_{}; // 0 = extinguish fire, 1 = fly to groundstation, 2 = recharge, 3 = explore
    bool extinguished_last_fire_ = false;
    bool active_ = false;

    TextureRenderer drone_texture_renderer_;
    TextureRenderer goal_texture_renderer_;

    // Possibly Deprecated
    bool extinguished_fire_{};
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

    std::shared_ptr<AgentState>
    BuildAgentState(const std::shared_ptr<GridMap> &grid_map, std::pair<double, double> velocity, int water_dispense);

    // Currently just for Debugging
    int last_step_total_revisited_cells_of_all_agents_{};
};


#endif //ROSHAN_FLY_AGENT_H
