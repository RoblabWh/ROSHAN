//
// Created by nex on 21.06.25.
//

#ifndef ROSHAN_PLANNER_AGENT_H
#define ROSHAN_PLANNER_AGENT_H

#include "agent.h"
#include "reinforcementlearning/actions/plan_action.h"
#include "reinforcementlearning/agents/explore_agent.h"


class GridMap;
class FireModelRenderer;

class PlannerAgent : public Agent {
public:
    explicit PlannerAgent(FireModelParameters &parameters, int total_id, int id, int time_steps);

    void ExecuteAction(std::shared_ptr<Action> action, std::string hierarchy_type, std::shared_ptr<GridMap> gridMap) override {
        action->ExecuteOn(shared_from_this(), hierarchy_type, gridMap);
    }

    void Initialize(std::shared_ptr<ExploreAgent> explore_agent,
                    std::vector<std::shared_ptr<FlyAgent>> fly_agents,
                    const std::shared_ptr<GridMap> &grid_map);

    void Reset(Mode mode,
               const std::shared_ptr<GridMap>& grid_map,
               const std::shared_ptr<FireModelRenderer>& model_renderer) override;

    void PerformPlan(PlanAction* action, const std::string& hierarchy_type, const std::shared_ptr<GridMap>& gridMap);
    bool GetPerformedHierarchyAction() const override { return did_hierarchy_step; }
    double CalculateReward(const std::shared_ptr<GridMap>& grid_map) override;
    void StepReset() override {
        did_hierarchy_step = false;
    }

    AgentTerminal GetTerminalStates(bool eval_mode, const std::shared_ptr<GridMap>& grid_map, int total_env_steps) override;
    void SetEvalMode(bool eval_mode) { eval_mode_ = eval_mode; }
private:
    void InitializePlannerAgentStates(const std::shared_ptr<GridMap> &grid_map);

    // Agents
    std::shared_ptr<ExploreAgent> explore_agent_;
    std::vector<std::shared_ptr<FlyAgent>> fly_agents_;

    //Gridmap
    std::shared_ptr<GridMap> gridmap_;

    bool did_hierarchy_step = false;
    std::vector<std::deque<std::pair<double, double>>> perfect_goals_;
    int goal_idx_ = 0;
    int revisited_cells_{};
    int extinguished_fires_ = 0;

    bool extinguished_last_fire_ = false;
    bool eval_mode_ = false;
    // Sentinel -1.0 skips the first-step reward without biasing later comparisons
    // (old 0.0 gated with '> 0.0' silently dropped any first step where the
    // drone spawned on its goal).
    double prev_mean_distance_ = -1.0;
    double prev_num_burning_ = -1.0;
    // Per-drone water levels from the previous planner step, used to compute the
    // WaterRefill dense reward. Empty vector is the "first call" sentinel.
    std::vector<double> prev_water_levels_;

    std::shared_ptr<AgentState> BuildAgentState(const std::shared_ptr<GridMap> &grid_map) override;
};

#endif //ROSHAN_PLANNER_AGENT_H
