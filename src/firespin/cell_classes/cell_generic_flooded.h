//
// Created by nex on 02.07.24.
//

#ifndef ROSHAN_CELL_GENERIC_FLOODED_H
#define ROSHAN_CELL_GENERIC_FLOODED_H

#include "firespin/firemodel_cell_interface.h"
#include <SDL.h>

class CellGenericFlooded : public ICell {
public:
    static void SetDefaultNoiseLevel(int noise_level) {
        default_noise_level_ = noise_level;
    }

    static void SetDefaultNoiseSize(int noise_size) {
        default_noise_size_ = noise_size;
    }

    CellGenericFlooded(SDL_PixelFormat* format) {
        color_ = {77, 187, 230, 255};
        mapped_color_ = SDL_MapRGBA(format, color_.r, color_.g, color_.b, color_.a);
        cell_burning_duration_ = 160; // currently overwritten by model parameters
        ignition_delay_time_ = 10; // currently overwritten by model parameters
        radiation_sf0_[0] = 0.1;
        radiation_sf0_[1] = 0.025;
        num_convection_particles_ = 10;
        radiation_length_min_ = 9;
        radiation_length_max_ = 10;
        num_radiation_particles_ = 4;
        has_noise_ = false;
        noise_level_ = default_noise_level_;
        noise_size_ = default_noise_size_;
    }

private:
    static int default_noise_level_;
    static int default_noise_size_;
};

#endif //ROSHAN_CELL_GENERIC_FLOODED_H
