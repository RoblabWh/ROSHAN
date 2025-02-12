//
// Created by nex on 11.06.23.
//

#include "wind.h"

void Wind::UpdateWind() {
    Uw_ = parameters_.GetWindSpeed();
    angle_ = parameters_.GetAngle();
    A_ = parameters_.GetA();
    u_prime_ = A_ * Uw_;
    CalculateComponents();
}

Wind::Wind(FireModelParameters &parameters) : parameters_(parameters) {
    UpdateWind();
}

void Wind::CalculateComponents() {
    Uw_i1_ = Uw_ * cos(angle_);
    Uw_i2_ = Uw_ * sin(angle_);
}

void Wind::SetRandomAngle() {
    double wind_angle_ = random() * 2 * M_PI / RAND_MAX;
    parameters_.SetWindAngle(wind_angle_);
    this->UpdateWind();
}
