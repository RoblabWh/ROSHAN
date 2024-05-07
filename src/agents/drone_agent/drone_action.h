//
// Created by nex on 25.07.23.
//

#ifndef ROSHAN_DRONE_ACTION_H
#define ROSHAN_DRONE_ACTION_H

#include "action.h"

class DroneAction : public Action{
public:
    DroneAction() : speed_x_(0), speed_y_(0), water_dispense_(0) {}
    DroneAction(double linear, double angular, int water_dispense) : speed_x_(linear), speed_y_(angular), water_dispense_(water_dispense) {}
    double GetSpeedX() { return speed_x_; }
    double GetSpeedY() { return speed_y_; }
    int GetWaterDispense() { return water_dispense_; }
private:
    double speed_x_;
    double speed_y_;
    int water_dispense_;
};

#endif //ROSHAN_DRONE_ACTION_H
