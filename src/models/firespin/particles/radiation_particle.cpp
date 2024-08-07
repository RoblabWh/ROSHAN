//
// Created by nex on 11.06.23.
//

#include <iostream>
#include "radiation_particle.h"

RadiationParticle::RadiationParticle(double x, double y, double Lr_min, double Lr_max, double Sf_0_mean, double Sf_0_std, double Y_st, double Y_lim, std::mt19937& gen)
: gen_(gen), uniform_dist_(Lr_min, Lr_max), normal_dist_sf_(Sf_0_mean, Sf_0_std) {
    X_[0] = x;
    X_[1] = y;
    X_mc_[0] = x;
    X_mc_[1] = y;

    phi_p_ = 0.0;
    r_p_ = 0.0;

    Sf_0_mean_ = Sf_0_mean;
    Sf_0_std_ = Sf_0_std;

    Sf_0_ = normal_dist_sf_(gen_);

    if (Sf_0_ < 0) {
        Sf_0_ = -Sf_0_;
    } // Wind Speed can't be negative

    Lr_ = uniform_dist_(gen_); // Characteristic radiation length in meters (m)
    if (Lr_ < 0) {
        throw std::runtime_error("Radiation Length Lr can't be negative");
    }
    // Make sure the particles don't exist too long, setting the value 0.000925 makes sure the particles exist for at most 3 hours
    // if the radiation length is set to 10 m. And for at most 0.75 hours if the radiation length is set to 25 m.
    if (Sf_0_ > 0.00925) {
        tau_mem_ = Lr_ / Sf_0_; // in seconds (s)
    } else {
        tau_mem_ = 2703;
    }
    Y_st_ = Y_st;
    Y_lim_ = Y_lim;
}

void RadiationParticle::UpdateState(double dt, RandomBuffer& buffer) {

    double N_phi = buffer.getNext();
    double N_r = buffer.getNext();

    // Update phi_p
    phi_p_ += M_PI_2_ * N_phi;
    r_p_ += Sf_0_mean_ * dt + Sf_0_std_ * sqrt(dt) * N_r;

    // Update position
    X_[0] = X_mc_[0] + r_p_ * cos(phi_p_);
    X_[1] = X_mc_[1] + r_p_ * sin(phi_p_);

    // Update burning status
    Y_st_ -= Y_st_ / tau_mem_ * dt;
}