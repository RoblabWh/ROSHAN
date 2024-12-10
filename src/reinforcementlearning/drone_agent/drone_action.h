//
// Created by nex on 25.07.23.
//

#ifndef ROSHAN_DRONE_ACTION_H
#define ROSHAN_DRONE_ACTION_H

#include "action.h"
#include "agent.h"

class DroneAction : public Action{
public:
    DroneAction() : speed_x_(0), speed_y_(0) {}
    DroneAction(double linear, double angular) : speed_x_(linear), speed_y_(angular) {}
    [[nodiscard]] double GetSpeedX() const { return speed_x_; }
    [[nodiscard]] double GetSpeedY() const { return speed_y_; }
    [[nodiscard]] int GetWaterDispense() const { return water_dispense_; }
    void Apply(const std::shared_ptr<Agent> &agent, const std::shared_ptr<GridMap> gridMap) const override {
        agent->OnDroneAction(std::make_shared<DroneAction>(*this), gridMap);
    }
private:
    double speed_x_;
    double speed_y_;
    int water_dispense_{};
};

#endif //ROSHAN_DRONE_ACTION_H
