//
// Created by nex on 08.04.25.
//

#ifndef ROSHAN_EXPLORE_AGENT_H
#define ROSHAN_EXPLORE_AGENT_H

#include "agent.h"
#include "reinforcementlearning/agents/fly_agent.h"
#include "reinforcementlearning/actions/explore_action.h"

class GridMap;

class ExploreAgent : public Agent {
public:
    explicit ExploreAgent(FireModelParameters &parameters, int id, int time_steps);

    void ExecuteAction(std::shared_ptr<Action> action, std::string hierarchy_type, std::shared_ptr<GridMap> gridMap) override {
        action->ExecuteOn(shared_from_this(), hierarchy_type, gridMap);
    }

    void Initialize(std::vector<std::shared_ptr<FlyAgent>> fly_agents, const std::shared_ptr<GridMap> &grid_map, const std::string &rl_mode);

    void PerformExplore(ExploreAction* action, const std::string& hierarchy_type, const std::shared_ptr<GridMap>& gridMap);
    bool GetPerformedHierarchyAction() const override { return did_hierarchy_step; }
    double CalculateReward() override;
    void StepReset() override {
        did_hierarchy_step = false;
    }

    std::vector<bool> GetTerminalStates(bool eval_mode, const std::shared_ptr<GridMap>& grid_map, int total_env_steps) override;

private:
    void InitializeExploreAgentStates();

    bool did_hierarchy_step = false;
    std::vector<std::shared_ptr<FlyAgent>> fly_agents_;
    int revisited_cells_{};

    // Rewards Collection for Debugging!
    bool explored_fires_equals_actual_fires_ = false;

    void UpdateStates(const std::shared_ptr<GridMap> &grid_map);
    void InitializeExploreAgentStates(const std::shared_ptr<GridMap> &grid_map);
    std::shared_ptr<AgentState> BuildAgentState(const std::shared_ptr<GridMap> &grid_map);
};


#endif //ROSHAN_EXPLORE_AGENT_H
