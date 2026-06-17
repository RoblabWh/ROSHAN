//
// Created by nex on 12.04.25.
//

#include "agent.h"

double Agent::ComputeTotalReward(const std::unordered_map<std::string, double>& rewards) {
    // Keys starting with '_' are diagnostics-only — emitted for TensorBoard but
    // excluded from the summed reward. Used e.g. by PlannerAgent to log Φ
    // components alongside the PBRS differential without double-counting them.
    double total = 0;
    for (const auto& [key, value] : rewards) {
        if (!key.empty() && key.front() == '_') continue;
        total += value;
    }
    return total;
}

void Agent::LogRewards(const std::unordered_map<std::string, double>& rewards) {
}

void Agent::UpdateStates(const std::shared_ptr<GridMap> &grid_map) {
    agent_states_.push_front(BuildAgentState(grid_map));

    // Maximum number of states i.e. memory
    if (agent_states_.size() > time_steps_) {
        agent_states_.pop_back();
    }
}
