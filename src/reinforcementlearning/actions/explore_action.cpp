//
// Created by nex on 08.04.25.
//

#include "explore_action.h"
#include "reinforcementlearning/agents/explore_agent.h"
#include <iostream>

void ExploreAction::ExecuteOn(std::shared_ptr<Agent> agent, std::string hierarchy_type, std::shared_ptr<GridMap> gridMap) {
    if (auto explore_agent = std::dynamic_pointer_cast<ExploreAgent>(agent)) {
        explore_agent->PerformExplore(this, hierarchy_type, gridMap);
    } else {
        std::cerr << "ExploreAction executed on non-ExploreAgent!" << std::endl;
    }
}