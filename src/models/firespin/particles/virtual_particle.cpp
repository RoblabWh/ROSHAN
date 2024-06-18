//
// Created by nex on 11.06.23.
//

#include "virtual_particle.h"

VirtualParticle::VirtualParticle(int x, int y, double tau_mem, double Y_st,
                                 double Y_lim, double Fl, double C0, double Lt, std::mt19937& gen)
                                 : gen_(gen), normal_dist_(0.0, 1.0) {
    for (double & i : U_) {
        i = 0.0;
    }
    X_[0] = x;
    X_[1] = y;
    U_[0] = 0.0;
    U_[1] = 0.0;
    tau_mem_ = tau_mem;
    Y_st_ = Y_st;
    Y_lim_ = Y_lim;
    Fl_ = Fl;
    C0_ = C0;
    Lt_ = Lt;

//    std::random_device rd;
//    gen_.seed(rd());

}

void VirtualParticle::UpdateState(Wind wind, double dt) {
    // Update Y_st
    Y_st_ -= Y_st_ / tau_mem_ * dt;
    u_prime_ = wind.GetCurrentTurbulece();
    Uw_i_ = {wind.getWindSpeedComponent1(), wind.getWindSpeedComponent2()};

    fac1 = ((2.0 + 3.0 * C0_) / 4.0) * (u_prime_ / Lt_);
    fac2 = pow((C0_ * pow(u_prime_, 3) / Lt_ * dt), 0.5);

    // Update X and U for each direction
    for (int i = 0; i < 2; ++i) {
        // Generate a normally-distributed random number for N_i
        N_i_ = normal_dist_(gen_);

        // Update U
        U_[i] += -(fac1 * (U_[i] - Uw_i_[i]) * dt) + fac2 * N_i_;

        // Update X
        X_[i] += Fl_ * U_[i] * dt;
    }
}
