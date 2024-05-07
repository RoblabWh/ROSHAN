//
// Created by nex on 25.07.23.
//

#ifndef ROSHAN_AGENT_H
#define ROSHAN_AGENT_H

#include <string>
#include <vector>
#include <memory>
#include <deque>
#include "action.h"
#include "state.h"

class Agent {
public:
    virtual ~Agent() = default;
    virtual std::vector<std::shared_ptr<Action>> SelectActions(std::vector<std::deque<std::shared_ptr<State>>> states) = 0;
    virtual void Update(std::vector<std::vector<std::shared_ptr<State>>> observations,
                        std::vector<std::shared_ptr<Action>> actions,
                        std::vector<double> rewards,
                        std::vector<std::vector<std::shared_ptr<State>>> next_observations,
                        std::vector<bool> dones) = 0;
};

#endif //ROSHAN_AGENT_H
