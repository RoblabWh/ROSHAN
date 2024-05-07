//
// Created by nex on 11.06.23.
//

#ifndef ROSHAN_VIRTUAL_PARTICLE_H
#define ROSHAN_VIRTUAL_PARTICLE_H

#include <cmath>
#include <vector>
#include <random>
#include "src/models/firespin/wind.h"
#include "src/models/firespin/model_parameters.h"

class VirtualParticle {

public:

    VirtualParticle(int x, int y, double tau_mem, double Y_st,
                    double Y_lim, double Fl, double C0, double Lt, std::mt19937 gen);
    void UpdateState(Wind wind, double dt);
    void GetPosition(double& x1, double& x2) const { x1 = X_[0]; x2 = X_[1];}
    double GetIntensity() const { return Y_st_; }
    bool IsCapableOfIgnition() const { return Y_st_ >= Y_lim_; }

private:
    double X_[2]{};      //Position
    double U_[2]{};      //Belocity
    double Y_st_;        //Burning status
    double tau_mem_;     // Memory timescale
    double Y_lim_;       // Ignition limit
    double Fl_;          // Scaling factor for new position
    double C0_;          // A constant close to 2
    double Lt_;          // I dont really know what this is
    std::vector<double> Uw_i_; // Wind velocity
    double u_prime_{};
    double N_i_{};
    double fac1{};
    double fac2{};

    std::mt19937 gen_;
    std::normal_distribution<> normal_dist_;
};


#endif //ROSHAN_VIRTUAL_PARTICLE_H
