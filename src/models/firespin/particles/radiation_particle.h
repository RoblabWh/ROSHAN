//
// Created by nex on 11.06.23.
//

#ifndef ROSHAN_RADIATION_PARTICLE_H
#define ROSHAN_RADIATION_PARTICLE_H

#include <cmath>
#include <random>

class RadiationParticle {

public:
    RadiationParticle(double x, double y, double Lr_min, double Lr_max, double Sf_0_mean, double Sf_0_std, double Y_st, double Y_lim, std::mt19937 gen);
    void UpdateState(double dt);
    void GetPosition(double &x1, double &x2) const { x1 = X_[0]; x2 = X_[1];}
    double GetIntensity() const { return Y_st_; }
    bool IsCapableOfIgnition() const { return Y_st_ >= Y_lim_; }

private:
    double X_[2];         // Position
    double X_mc_[2];      // Position of the mother cell
    double r_p_;          // Radius
    double phi_p_;        // Angle
    double Y_st_;         // Burning status
    double Y_lim_;        // Ignition limit
    double tau_mem_;      // Decay timescale
    double Lr_;           // A const Scaling factor
    double Sf_0_mean_;
    double Sf_0_std_;
    double Sf_0_;         // A normal distribution

    double N_phi_;
    double N_r_;
    double M_PI_2_ = 2 * M_PI;

    // Generate a normally-distributed random number for phi_r
    std::mt19937 gen_;
    std::normal_distribution<> normal_dist_;
};


#endif //ROSHAN_RADIATION_PARTICLE_H
