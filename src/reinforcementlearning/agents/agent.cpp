//
// Created by nex on 12.04.25.
//

#include "agent.h"

double Agent::ComputeTotalReward(const std::unordered_map<std::string, double>& rewards) {
    double total = 0;
    for (const auto& [key, value] : rewards) {
        total += value;
    }
    return total;
}

std::deque<std::shared_ptr<State>> Agent::GetObservations() {
    std::deque<std::shared_ptr<State>> states;
    for (const auto& state : agent_states_) {
        states.push_back(state); // Already a shared_ptr
    }
    return states;
}

void Agent::LogRewards(const std::unordered_map<std::string, double>& rewards) {
#ifdef DEBUG_REWARD_YES
    for (const auto& [key, value] : rewards) {
            std::cout << key << " Reward: " << value << "\n";
        }
        std::cout << "\n";
#endif
}

void Agent::UpdateStates(const std::shared_ptr<GridMap> &grid_map) {
    agent_states_.push_front(BuildAgentState(grid_map));

    // Maximum number of states i.e. memory
    if (agent_states_.size() > time_steps_) {
        agent_states_.pop_back();
    }
}
