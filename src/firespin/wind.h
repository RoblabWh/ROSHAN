//
// Created by nex on 11.06.23.
//

#ifndef ROSHAN_WIND_H
#define ROSHAN_WIND_H

#include <cmath>
#include "model_parameters.h"

class Wind {

public:

    explicit Wind(FireModelParameters &parameters);

    [[maybe_unused]] [[nodiscard]] double GetCurrentWindSpeed() const {return Uw_;}
    [[nodiscard]] double getWindSpeedComponent1() const { return Uw_i1_; }
    [[nodiscard]] double getWindSpeedComponent2() const { return Uw_i2_; }

    [[maybe_unused]] [[nodiscard]] double GetCurrentA() const {return A_;}
    [[nodiscard]] double GetCurrentTurbulence() const {return u_prime_;}
    [[nodiscard]] double GetCurrentAngle() const {return angle_;}
    void SetRandomAngle();

    void UpdateWind();

private:
    void CalculateComponents();
    FireModelParameters &parameters_;
    double Uw_;    // The 10-m wind speed
    double angle_; // The angle of the wind direction
    double Uw_i1_; // The component of the wind speed in the 1st direction
    double Uw_i2_; // The component of the wind speed in the 2nd direction
    // Atmospheric boundary layer parameters
    double A_; // Parameter related to turbulent velocity fluctuations
    double u_prime_; // Turbulent velocity fluctuations
};


#endif //ROSHAN_WIND_H
