//
// Created by nex on 25.07.23.
//

#ifndef ROSHAN_ACTION_H
#define ROSHAN_ACTION_H

#include <memory>

//class Agent;
//class GridMap;
//
//class Action {
//public:
//    virtual ~Action() = default;
//    virtual void Apply(const std::shared_ptr<Agent> &agent, std::shared_ptr<GridMap> gridMap) const = 0;
//};

#include <memory>

class Agent;
class GridMap;

class Action {
public:
    virtual ~Action() = default;
    virtual void ExecuteOn(std::shared_ptr<Agent> agent, std::string hierarchy_type, std::shared_ptr<GridMap> gridMap) = 0;
};

#endif //ROSHAN_ACTION_H
