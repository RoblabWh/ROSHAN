//
// Created by nex on 14.01.25.
//

#ifndef ROSHAN_EXPLORE_ACTION_H
#define ROSHAN_EXPLORE_ACTION_H

#include "action.h"

class ExploreAction : public Action {
public:
    ExploreAction() : goal_x_(0), goal_y_(0) {}
    ExploreAction(double goal_x, double goal_y) : goal_x_(goal_x), goal_y_(goal_y) {}
    [[nodiscard]] double GetGoalX() const { return goal_x_; }
    [[nodiscard]] double GetGoalY() const { return goal_y_; }
    void ExecuteOn(std::shared_ptr<Agent> agent, std::string hierarchy_type, std::shared_ptr<GridMap> gridMap) override;
    // additional FlyAction-specific members
private:
    double goal_x_;
    double goal_y_;
};


#endif //ROSHAN_EXPLORE_ACTION_H
