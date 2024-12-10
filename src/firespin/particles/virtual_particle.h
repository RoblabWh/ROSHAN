//
// Created by nex on 11.06.23.
//

#ifndef ROSHAN_VIRTUAL_PARTICLE_H
#define ROSHAN_VIRTUAL_PARTICLE_H

#include <cmath>
#include <vector>
#include <random>
#include "firespin/wind.h"
#include "firespin/model_parameters.h"
#include "firespin/utils.h"

class VirtualParticle {

public:

    VirtualParticle(int x, int y, double tau_mem, double Y_st,
                    double Y_lim, double Fl, double C0, double Lt);
    void UpdateState(Wind& wind, double dt, RandomBuffer& buffer);
    void GetPosition(double& x1, double& x2) const { x1 = X_[0]; x2 = X_[1];}
    double GetIntensity() const { return Y_st_; }
    bool IsCapableOfIgnition() const { return Y_st_ >= Y_lim_; }

    // Delete copy constructor and copy assignment operator
    VirtualParticle(const VirtualParticle&) = delete;
    VirtualParticle& operator=(const VirtualParticle&) = delete;

    // Define move constructor
    VirtualParticle(VirtualParticle&& other) noexcept {
        *this = std::move(other);
    }

    // Define move assignment operator
    VirtualParticle& operator=(VirtualParticle&& other) noexcept {
        if (this != &other) {
            X_[0] = other.X_[0];
            X_[1] = other.X_[1];
            U_[0] = other.U_[0];
            U_[1] = other.U_[1];
            Y_st_ = other.Y_st_;
            tau_mem_ = other.tau_mem_;
            Y_lim_ = other.Y_lim_;
            Fl_ = other.Fl_;
            C0_ = other.C0_;
            Lt_ = other.Lt_;
            Uw_i_ = std::move(other.Uw_i_);
            u_prime_ = other.u_prime_;
            N_i_ = other.N_i_;
            fac1 = other.fac1;
            fac2 = other.fac2;
        }
        return *this;
    }

private:
    double X_[2]{};      //Position
    double U_[2]{};      //Belocity
    double Y_st_{};        //Burning status
    double tau_mem_{};     // Memory timescale
    double Y_lim_{};       // Ignition limit
    double Fl_{};          // Scaling factor for new position
    double C0_{};          // A constant close to 2
    double Lt_{};          // I dont really know what this is
    std::vector<double> Uw_i_; // Wind velocity
    double u_prime_{};
    double N_i_{};
    double fac1{};
    double fac2{};
};


#endif //ROSHAN_VIRTUAL_PARTICLE_H
