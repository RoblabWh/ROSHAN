//
// Created by nex on 25.07.23.
//

#ifndef ROSHAN_AGENT_H
#define ROSHAN_AGENT_H

#include <string>
#include <vector>
#include <memory>
#include <deque>
#include "state.h"
#include "action.h"

class FlyAction;
class ExploreAction;
class GridMap;

class Agent : public std::enable_shared_from_this<Agent> {
public:
    virtual ~Agent() = default;
    virtual void OnFlyAction(std::shared_ptr<FlyAction> action, std::shared_ptr<GridMap> gridMap) = 0;
    virtual void OnExploreAction(std::shared_ptr<ExploreAction> action, std::shared_ptr<GridMap> gridMap) = 0;
};

#endif //ROSHAN_AGENT_H
