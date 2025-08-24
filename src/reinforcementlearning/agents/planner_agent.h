//
// Created by nex on 21.06.25.
//

#ifndef ROSHAN_PLANNER_AGENT_H
#define ROSHAN_PLANNER_AGENT_H

#include "agent.h"
#include "reinforcementlearning/actions/plan_action.h"
#include "reinforcementlearning/agents/explore_agent.h"


class GridMap;

class PlannerAgent : public Agent {
public:
    explicit PlannerAgent(FireModelParameters &parameters, int id, int time_steps);

    void ExecuteAction(std::shared_ptr<Action> action, std::string hierarchy_type, std::shared_ptr<GridMap> gridMap) override {
        action->ExecuteOn(shared_from_this(), hierarchy_type, gridMap);
    }

    void Initialize(std::shared_ptr<ExploreAgent> explore_agent, std::vector<std::shared_ptr<FlyAgent>> fly_agents, const std::shared_ptr<GridMap> &grid_map, const std::string &rl_mode);

    void PerformPlan(PlanAction* action, const std::string& hierarchy_type, const std::shared_ptr<GridMap>& gridMap);
    bool GetPerformedHierarchyAction() const override { return did_hierarchy_step; }
    double CalculateReward() override;
    void StepReset() override {
        did_hierarchy_step = false;
    }

    AgentTerminal GetTerminalStates(bool eval_mode, const std::shared_ptr<GridMap>& grid_map, int total_env_steps) override;
    void SetGridMap(std::shared_ptr<GridMap> gridmap) { gridmap_ = std::move(gridmap); }
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

    // Rewards Collection for Debugging!
    bool explored_fires_equals_actual_fires_ = false;
    bool extinguished_last_fire_ = false;

    void UpdateStates(const std::shared_ptr<GridMap> &grid_map);
    std::shared_ptr<AgentState> BuildAgentState(const std::shared_ptr<GridMap> &grid_map);
};

#endif //ROSHAN_PLANNER_AGENT_H
