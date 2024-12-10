//
// Created by nex on 27.06.23.
//

#ifndef ROSHAN_FIREMODEL_CELL_INTERFACE_H
#define ROSHAN_FIREMODEL_CELL_INTERFACE_H

#include <SDL.h>
#include "imgui.h"
#include "utils.h"

class ICell {
public:
    virtual ~ICell() = default;
    SDL_Color color_;

    ImVec4 GetImVecColor() const {
        return ImVec4(color_.r, color_.g, color_.b, color_.a);
    }
    Uint32 GetMappedColor() const { return mapped_color_; }
    bool HasNoise() const { return has_noise_; }
    double GetCellBurningDuration() const { return cell_burning_duration_; }
    double GetIgnitionDelayTime() const { return ignition_delay_time_; }
    double GetSf0Mean() const { return radiation_sf0_[0]; }
    double GetSf0Std() const { return radiation_sf0_[1]; }
    int GetNumConvectionParticles() const { return num_convection_particles_; }
    int GetNumRadiationParticles() const { return num_radiation_particles_; }
    std::pair<double, double> GetRadiationLength() const { return std::make_pair(radiation_length_min_, radiation_length_max_); }
    void SetCellBurningDuration(double cell_burning_duration) { cell_burning_duration_ = cell_burning_duration; }

    // Rendering Only
    virtual int GetNoiseLevel() const { return noise_level_; }
    virtual int GetNoiseSize() const { return noise_size_; }

protected:
    Uint32 mapped_color_;
    bool has_noise_;
    double cell_burning_duration_;
    double ignition_delay_time_;
    double radiation_length_max_; //Lr [m]
    double radiation_length_min_; //Lr [m]
    double radiation_sf0_[2]; // No wind propagation speed of radiation particles [m/s] (mean and std)
    int num_convection_particles_;
    int num_radiation_particles_;
    // Rendering Only
    int noise_level_ = 20;
    int noise_size_ = 2;
};

#endif //ROSHAN_FIREMODEL_CELL_INTERFACE_H
