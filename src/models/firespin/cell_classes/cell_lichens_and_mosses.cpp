//
// Created by nex on 27.06.23.
//

#ifndef ROSHAN_FIREMODEL_LICHENS_AND_MOSSES_
#define ROSHAN_FIREMODEL_LICHENS_AND_MOSSES_

#include "src/models/firespin/firemodel_cell_interface.h"
#include <SDL.h>

class CellLichensAndMosses : public ICell {
public:
    CellLichensAndMosses(SDL_PixelFormat* format) {
        color_ = {255, 153, 204, 255};
        mapped_color_ = SDL_MapRGBA(format, color_.r, color_.g, color_.b, color_.a);
        cell_burning_duration_ = 240;
        ignition_delay_time_ = 100;
        radiation_sf0_[0] = 0.1;
        radiation_sf0_[1] = 0.025;
        num_convection_particles_ = 10;
        radiation_length_min_ = 9;
        radiation_length_max_ = 10;
        num_radiation_particles_ = 3;
    }
};

#endif //ROSHAN_FIREMODEL_LICHENS_AND_MOSSES_