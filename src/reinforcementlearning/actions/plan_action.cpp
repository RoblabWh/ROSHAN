//
// Created by nex on 21.06.25.
//

#include "plan_action.h"
#include "reinforcementlearning/agents/planner_agent.h"

void PlanAction::ExecuteOn(std::shared_ptr<Agent> agent, std::string hierarchy_type, std::shared_ptr<GridMap> gridMap) {
    if (auto plan_agent = std::dynamic_pointer_cast<PlannerAgent>(agent)) {
        plan_agent->PerformPlan(this, hierarchy_type, gridMap);
    } else {
        std::cerr << "PlanAction executed on non-PlannerAgent!" << std::endl;
    }
}

std::pair<double, double> PlanAction::GetGoalFromAction(int agent_id) {
    return goals_[agent_id]; // Return the goal for the specific agent
}
