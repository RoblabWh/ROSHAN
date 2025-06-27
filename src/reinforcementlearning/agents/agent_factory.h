//
// Created by nex on 08.04.25.
//

#ifndef ROSHAN_AGENT_FACTORY_H
#define ROSHAN_AGENT_FACTORY_H


#include "agent.h"
#include "explore_agent.h"
#include "fly_agent.h"
#include "planner_agent.h"
#include "firespin/model_parameters.h"
#include "firespin/firemodel_gridmap.h"

#include <memory>
#include <unordered_map>
#include <functional>
#include <string>
#include <utility>

class AgentFactory {
public:
    using CreatorFunction = std::function<std::shared_ptr<Agent>(FireModelParameters& parameters, int drone_id, int time_steps)>;

    static AgentFactory& GetInstance() {
        static AgentFactory instance;
        return instance;
    }

    void RegisterAgent(const std::string& agent_type, CreatorFunction creator) {
        creators_[agent_type] = std::move(creator);
    }

    std::shared_ptr<Agent> CreateAgent(const std::string& agent_type,
                                       FireModelParameters& parameters,
                                       int id,
                                       int time_steps) {
        auto it = creators_.find(agent_type);
        if (it != creators_.end()) {
            return (it->second)(parameters, id, time_steps);
        }
        throw std::runtime_error("Unknown agent type: " + agent_type);
    }

private:
    AgentFactory() = default;
    std::unordered_map<std::string, CreatorFunction> creators_;
};


#endif //ROSHAN_AGENT_FACTORY_H
