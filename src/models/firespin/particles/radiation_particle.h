//
// Created by nex on 11.06.23.
//

#ifndef ROSHAN_RADIATION_PARTICLE_H
#define ROSHAN_RADIATION_PARTICLE_H

#include <cmath>
#include <random>

class RadiationParticle {

public:
    RadiationParticle(double x, double y, double Lr_min, double Lr_max, double Sf_0_mean, double Sf_0_std, double Y_st, double Y_lim, std::mt19937& gen);
    void UpdateState(double dt);
    void GetPosition(double &x1, double &x2) const { x1 = X_[0]; x2 = X_[1];}
    double GetIntensity() const { return Y_st_; }
    bool IsCapableOfIgnition() const { return Y_st_ >= Y_lim_; }
    // Delete copy constructor and copy assignment operator
    RadiationParticle(const RadiationParticle&) = delete;
    RadiationParticle& operator=(const RadiationParticle&) = delete;

    // Define move constructor
    RadiationParticle(RadiationParticle&& other) noexcept
            : gen_(other.gen_), normal_dist_(std::move(other.normal_dist_)),
              uniform_dist_(std::move(other.uniform_dist_)), normal_dist_sf_(std::move(other.normal_dist_sf_)) {
        *this = std::move(other);
    }

    // Define move assignment operator
    RadiationParticle& operator=(RadiationParticle&& other) noexcept {
        if (this != &other) {
            X_[0] = other.X_[0];
            X_[1] = other.X_[1];
            X_mc_[0] = other.X_mc_[0];
            X_mc_[1] = other.X_mc_[1];
            phi_p_ = other.phi_p_;
            r_p_ = other.r_p_;
            Sf_0_mean_ = other.Sf_0_mean_;
            Sf_0_std_ = other.Sf_0_std_;
            Sf_0_ = other.Sf_0_;
            Lr_ = other.Lr_;
            tau_mem_ = other.tau_mem_;
            Y_st_ = other.Y_st_;
            Y_lim_ = other.Y_lim_;
            N_phi_ = other.N_phi_;
            N_r_ = other.N_r_;
        }
        return *this;
    }

private:
    double X_[2]{};         // Position
    double X_mc_[2]{};      // Position of the mother cell
    double r_p_{};          // Radius
    double phi_p_{};        // Angle
    double Y_st_{};         // Burning status
    double Y_lim_{};        // Ignition limit
    double tau_mem_{};      // Decay timescale
    double Lr_{};           // A const Scaling factor
    double Sf_0_mean_{};
    double Sf_0_std_{};
    double Sf_0_{};         // A normal distribution

    double N_phi_{};
    double N_r_{};
    double M_PI_2_ = 2 * M_PI;

    // Generate a normally-distributed random number for phi_r
    std::mt19937& gen_;
    std::normal_distribution<> normal_dist_;
    std::uniform_real_distribution<> uniform_dist_;
    std::normal_distribution<> normal_dist_sf_;
};


#endif //ROSHAN_RADIATION_PARTICLE_H
