//
// Created by nex on 25.07.23.
//

#ifndef ROSHAN_FLY_ACTION_H
#define ROSHAN_FLY_ACTION_H

//#include "reinforcementlearning/actions/action.h"
//#include "reinforcementlearning/agents/agent.h"
//
//class FlyAction : public Action{
//public:
//    FlyAction() : speed_x_(0), speed_y_(0) {}
//    FlyAction(double linear, double angular) : speed_x_(linear), speed_y_(angular) {}
//    [[nodiscard]] double GetSpeedX() const { return speed_x_; }
//    [[nodiscard]] double GetSpeedY() const { return speed_y_; }
//    [[nodiscard]] int GetWaterDispense() const { return water_dispense_; }
//    void Apply(const std::shared_ptr<Agent> &agent, const std::shared_ptr<GridMap> gridMap) const override {
//        agent->OnFlyAction(std::make_shared<FlyAction>(*this), gridMap);
//    }
//private:
//    double speed_x_;
//    double speed_y_;
//    int water_dispense_{};
//};

#include <string>
#include "action.h"

class FlyAction : public Action {
public:
    FlyAction() : speed_x_(0), speed_y_(0) {}
    FlyAction(double linear, double angular) : speed_x_(linear), speed_y_(angular) {}
    [[nodiscard]] double GetSpeedX() const { return speed_x_; }
    [[nodiscard]] double GetSpeedY() const { return speed_y_; }
    [[nodiscard]] int GetWaterDispense() const { return water_dispense_; }
    void ExecuteOn(std::shared_ptr<Agent> agent, std::string hierarchy_type, std::shared_ptr<GridMap> gridMap) override;
    // additional FlyAction-specific members
private:
    double speed_x_;
    double speed_y_;
    int water_dispense_{};
};

#endif //ROSHAN_FLY_ACTION_H
