//
// Created by nex on 14.01.25.
//

#ifndef ROSHAN_EXPLORE_ACTION_H
#define ROSHAN_EXPLORE_ACTION_H

#include "action.h"
#include "agent.h"

class ExploreAction : public Action{
public:
    ExploreAction() : goal_x_(0), goal_y_(0) {}
    ExploreAction(double goal_x, double goal_y) : goal_x_(goal_x), goal_y_(goal_y) {}
    [[nodiscard]] double GetGoalX() const { return goal_x_; }
    [[nodiscard]] double GetGoalY() const { return goal_y_; }
    void Apply(const std::shared_ptr<Agent> &agent, const std::shared_ptr<GridMap> gridMap) const override {
        agent->OnExploreAction(std::make_shared<ExploreAction>(*this), gridMap);
    }
private:
    double goal_x_;
    double goal_y_;
    int water_dispense_{};
};


#endif //ROSHAN_EXPLORE_ACTION_H
