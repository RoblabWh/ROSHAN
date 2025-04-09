//
// Created by nex on 08.04.25.
//

#include "fly_action.h"
#include "reinforcementlearning/agents/fly_agent.h"
#include <iostream>

void FlyAction::ExecuteOn(std::shared_ptr<Agent> agent, std::string hierarchy_type, std::shared_ptr<GridMap> gridMap) {
    if (auto fly_agent = std::dynamic_pointer_cast<FlyAgent>(agent)) {
        fly_agent->PerformFly(this, hierarchy_type, gridMap);
    } else {
        std::cerr << "FlyAction executed on non-FlyAgent!" << std::endl;
    }
}