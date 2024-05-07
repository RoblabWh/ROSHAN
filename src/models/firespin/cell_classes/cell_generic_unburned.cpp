//
// Created by nex on 27.06.23.
//

#ifndef ROSHAN_FIREMODEL_GENERIC_UNBURNED_
#define ROSHAN_FIREMODEL_GENERIC_UNBURNED_

#include "src/models/firespin/firemodel_cell_interface.h"
#include <SDL.h>

class CellGenericUnburned : public ICell {
public:
    CellGenericUnburned(SDL_PixelFormat* format) {
        color_ = {50, 190, 75, 255};
        mapped_color_ = SDL_MapRGBA(format, color_.r, color_.g, color_.b, color_.a);
        cell_burning_duration_ = 160; // currently overwritten by model parameters
        ignition_delay_time_ = 10; // currently overwritten by model parameters
        radiation_sf0_[0] = 0.1;
        radiation_sf0_[1] = 0.025;
        num_convection_particles_ = 10;
        radiation_length_min_ = 5;
        radiation_length_max_ = 50;
        num_radiation_particles_ = 4;
    }
};

#endif //ROSHAN_FIREMODEL_GENERIC_UNBURNED_