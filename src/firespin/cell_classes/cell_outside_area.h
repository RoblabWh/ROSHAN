//
// Created by nex on 02.07.24.
//

#ifndef ROSHAN_CELL_OUTSIDE_AREA_H
#define ROSHAN_CELL_OUTSIDE_AREA_H

#include "firespin/firemodel_cell_interface.h"
#include <SDL.h>

class CellOutsideArea : public ICell {
public:
    static void SetDefaultNoiseLevel(int noise_level) {
        default_noise_level_ = noise_level;
    }

    static void SetDefaultNoiseSize(int noise_size) {
        default_noise_size_ = noise_size;
    }

    CellOutsideArea() {
        color_ = {25, 25, 25, 255};
        InitMappedColor();
        cell_burning_duration_ = 0;
        ignition_delay_time_ = 0;
        radiation_sf0_[0] = 0;
        radiation_sf0_[1] = 0;
        num_convection_particles_ = 0;
        radiation_length_min_ = 0;
        radiation_length_max_ = 0;
        num_radiation_particles_ = 0;
        has_noise_ = false;
        noise_level_ = default_noise_level_;
        noise_size_ = default_noise_size_;
    }

private:
    static int default_noise_level_;
    static int default_noise_size_;
};
#endif //ROSHAN_CELL_OUTSIDE_AREA_H
