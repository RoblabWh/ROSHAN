//
// Created by nex on 02.07.24.
//

#ifndef ROSHAN_CELL_PERIODICALLY_HERBACEOUS_H
#define ROSHAN_CELL_PERIODICALLY_HERBACEOUS_H

#include "firespin/firemodel_cell_interface.h"
#include <SDL.h>

class CellPeriodicallyHerbaceous : public ICell {
public:
    static void SetDefaultNoiseLevel(int noise_level) {
        default_noise_level_ = noise_level;
    }

    static void SetDefaultNoiseSize(int noise_size) {
        default_noise_size_ = noise_size;
    }

    CellPeriodicallyHerbaceous() {
        color_ = {240, 230, 140, 255};
        InitMappedColor();
        cell_burning_duration_ = 360;
        ignition_delay_time_ = 100;
        radiation_sf0_[0] = 0.1;
        radiation_sf0_[1] = 0.025;
        num_convection_particles_ = 20;
        radiation_length_min_ = 9;
        radiation_length_max_ = 10;
        num_radiation_particles_ = 2;
        has_noise_ = true;
        noise_level_ = default_noise_level_;
        noise_size_ = default_noise_size_;
    }

private:
    static int default_noise_level_;
    static int default_noise_size_;
};


#endif //ROSHAN_CELL_PERIODICALLY_HERBACEOUS_H
