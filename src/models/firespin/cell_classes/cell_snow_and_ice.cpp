//
// Created by nex on 27.06.23.
//

#ifndef ROSHAN_FIREMODEL_SNOW_AND_ICE_
#define ROSHAN_FIREMODEL_SNOW_AND_ICE_

#include "src/models/firespin/firemodel_cell_interface.h"
#include <SDL.h>

class CellSnowAndIce : public ICell {
public:
    CellSnowAndIce(SDL_PixelFormat* format) {
        color_ = {0, 255, 255, 255};
        mapped_color_ = SDL_MapRGBA(format, color_.r, color_.g, color_.b, color_.a);
        cell_burning_duration_ = 0;
        ignition_delay_time_ = -1;
        radiation_sf0_[0] = 0;
        radiation_sf0_[1] = 0;
        num_convection_particles_ = 0;
        radiation_length_min_ = 0;
        radiation_length_max_ = 0;
        num_radiation_particles_ = 0;
    }
};

#endif //ROSHAN_FIREMODEL_SNOW_AND_ICE_