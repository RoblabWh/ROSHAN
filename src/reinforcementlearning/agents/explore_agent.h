//
// Created by nex on 08.04.25.
//

#ifndef ROSHAN_EXPLORE_AGENT_H
#define ROSHAN_EXPLORE_AGENT_H

#include "agent.h"
#include "reinforcementlearning/agents/fly_agent.h"
#include "reinforcementlearning/actions/explore_action.h"

class GridMap;
class FireModelRenderer;

class ExploreAgent : public Agent {
public:
    explicit ExploreAgent(FireModelParameters &parameters, int id, int time_steps);

    void ExecuteAction(std::shared_ptr<Action> action, std::string hierarchy_type, std::shared_ptr<GridMap> gridMap) override {
        action->ExecuteOn(shared_from_this(), hierarchy_type, gridMap);
    }

    void Initialize(std::vector<std::shared_ptr<FlyAgent>> fly_agents, const std::shared_ptr<GridMap> &grid_map);
    void Reset(Mode mode,
               const std::shared_ptr<GridMap>& grid_map,
               const std::shared_ptr<FireModelRenderer>& model_renderer) override;


    void PerformExplore(ExploreAction* action, const std::string& hierarchy_type, const std::shared_ptr<GridMap>& gridMap);
    bool GetPerformedHierarchyAction() const override { return did_hierarchy_step; }
    double CalculateReward(const std::shared_ptr<GridMap>& grid_map) override;
    void StepReset() override {
        did_hierarchy_step = false;
    }

    AgentTerminal GetTerminalStates(bool eval_mode, const std::shared_ptr<GridMap>& grid_map, int total_env_steps) override;

private:
    bool did_hierarchy_step = false;
    std::vector<std::shared_ptr<FlyAgent>> fly_agents_;
    std::vector<std::deque<std::pair<double, double>>> perfect_goals_;
    int goal_idx_ = 0;
    int revisited_cells_{};

    // Rewards Collection for Debugging!
    bool explored_fires_equals_actual_fires_ = false;

    static std::pair<double, double> GetGoalFromAction(const ExploreAction* action, const std::shared_ptr<GridMap> &grid_map);
    std::pair<double, double> GetGoalFromCertain(std::deque<std::pair<double, double>> &goals, const std::shared_ptr<GridMap>& gridMap);
    void InitializeExploreAgentStates(const std::shared_ptr<GridMap> &grid_map);
    std::shared_ptr<AgentState> BuildAgentState(const std::shared_ptr<GridMap> &grid_map) override;
};


#endif //ROSHAN_EXPLORE_AGENT_H
