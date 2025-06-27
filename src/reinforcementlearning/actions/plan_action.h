//
// Created by nex on 21.06.25.
//

#ifndef ROSHAN_PLAN_ACTION_H
#define ROSHAN_PLAN_ACTION_H


#include <string>
#include <utility>
#include <vector>
#include <iostream>
#include "action.h"

class PlanAction : public Action {
public:
    PlanAction() : goals_(std::vector<std::pair<double, double>>()) {}
    PlanAction(std::vector<std::pair<double, double>> goals) : goals_(std::move(goals)) {}
    void ExecuteOn(std::shared_ptr<Agent> agent, std::string hierarchy_type, std::shared_ptr<GridMap> gridMap) override;
    // additional PlanAction-specific members
    std::pair<double, double> GetGoalFromAction(int agent_id);
private:
    std::vector<std::pair<double, double>> goals_; // List of goals to be planned
};


#endif //ROSHAN_PLAN_ACTION_H
