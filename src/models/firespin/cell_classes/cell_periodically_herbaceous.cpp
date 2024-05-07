//
// Created by nex on 27.06.23.
//

#ifndef ROSHAN_FIREMODEL_PERIODICALLY_HERBACEOUS_
#define ROSHAN_FIREMODEL_PERIODICALLY_HERBACEOUS_

#include "src/models/firespin/firemodel_cell_interface.h"
#include <SDL.h>

class CellPeriodicallyHerbaceous : public ICell {
public:
    CellPeriodicallyHerbaceous(SDL_PixelFormat* format) {
        color_ = {240, 230, 140, 255};
        mapped_color_ = SDL_MapRGBA(format, color_.r, color_.g, color_.b, color_.a);
        cell_burning_duration_ = 360;
        ignition_delay_time_ = 100;
        radiation_sf0_[0] = 0.1;
        radiation_sf0_[1] = 0.025;
        num_convection_particles_ = 20;
        radiation_length_min_ = 9;
        radiation_length_max_ = 10;
        num_radiation_particles_ = 2;
    }
};

#endif //ROSHAN_FIREMODEL_PERIODICALLY_HERBACEOUS_