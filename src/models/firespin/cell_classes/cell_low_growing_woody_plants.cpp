//
// Created by nex on 27.06.23.
//

#ifndef ROSHAN_FIREMODEL_LOW_GROWING_WOODY_PLANTS_
#define ROSHAN_FIREMODEL_LOW_GROWING_WOODY_PLANTS_

#include "src/models/firespin/firemodel_cell_interface.h"
#include <SDL.h>

class CellLowGrowingWoodyPlants : public ICell {
public:
    CellLowGrowingWoodyPlants(SDL_PixelFormat* format) {
        color_ = {105, 76, 51, 255};
        mapped_color_ = SDL_MapRGBA(format, color_.r, color_.g, color_.b, color_.a);
        cell_burning_duration_ = 720;
        ignition_delay_time_ = 120;
        radiation_sf0_[0] = 0.1;
        radiation_sf0_[1] = 0.025;
        num_convection_particles_ = 40;
        radiation_length_min_ = 9;
        radiation_length_max_ = 10;
        num_radiation_particles_ = 4;
    }
};

#endif //ROSHAN_FIREMODEL_LOW_GROWING_WOODY_PLANTS_