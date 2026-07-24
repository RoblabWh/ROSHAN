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
    // Draw from the seeded engine, not libc random() (which is never seeded)
    std::uniform_real_distribution<double> dist(0.0, 2.0 * M_PI);
    parameters_.SetWindAngle(dist(parameters_.gen_));
    this->UpdateWind();
}
