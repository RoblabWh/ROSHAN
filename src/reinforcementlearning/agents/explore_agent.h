//
// Created by nex on 08.04.25.
//

#ifndef ROSHAN_EXPLORE_AGENT_H
#define ROSHAN_EXPLORE_AGENT_H

#include "agent.h"
#include "reinforcementlearning/actions/explore_action.h"

class GridMap;

class ExploreAgent : public Agent {
public:
    void ExecuteAction(std::shared_ptr<Action> action, std::string hierarchy_type, std::shared_ptr<GridMap> gridMap) override {
        action->ExecuteOn(shared_from_this(), hierarchy_type, gridMap);
    }

    void PerformExplore(ExploreAction* action, std::string hierarchy_type, std::shared_ptr<GridMap> gridMap);
    bool GetPerformedHierarchyAction() const override { return did_hierarchy_step; }
    double CalculateReward() override;
    void StepReset() override {

    }

private:
    bool did_hierarchy_step = false;
};


#endif //ROSHAN_EXPLORE_AGENT_H
