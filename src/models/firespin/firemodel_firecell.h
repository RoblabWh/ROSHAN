//
// Created by nex on 10.06.23.
//

#ifndef ROSHAN_FIREMODEL_FIRECELL_H
#define ROSHAN_FIREMODEL_FIRECELL_H

#include "point.h"
#include "src/models/firespin/particles/virtual_particle.h"
#include "src/models/firespin/particles/radiation_particle.h"
#include "model_parameters.h"
#include "firemodel_cell_interface.h"
#include "imgui.h"
#include "wind.h"
#include "utils.h"
#include <vector>
#include <random>
#include <chrono>

#include "src/models/firespin/cell_classes/cell_generic_burned.cpp"
#include "src/models/firespin/cell_classes/cell_generic_unburned.cpp"
#include "src/models/firespin/cell_classes/cell_generic_burning.cpp"
#include "src/models/firespin/cell_classes/cell_lichens_and_mosses.cpp"
#include "src/models/firespin/cell_classes/cell_low_growing_woody_plants.cpp"
#include "src/models/firespin/cell_classes/cell_non_and_sparsley_vegetated.cpp"
#include "src/models/firespin/cell_classes/cell_outside_area.cpp"
#include "src/models/firespin/cell_classes/cell_periodically_herbaceous.cpp"
#include "src/models/firespin/cell_classes/cell_permanent_herbaceous.cpp"
#include "src/models/firespin/cell_classes/cell_sealed.cpp"
#include "src/models/firespin/cell_classes/cell_snow_and_ice.cpp"
#include "src/models/firespin/cell_classes/cell_water.cpp"
#include "src/models/firespin/cell_classes/cell_woody_breadleaved_deciduous_trees.cpp"
#include "src/models/firespin/cell_classes/cell_woody_broadleaved_evergreen_trees.cpp"
#include "src/models/firespin/cell_classes/cell_woody_needle_leaved_trees.cpp"
#include "src/models/firespin/cell_classes/cell_generic_flooded.cpp"


class FireCell {

public:
    FireCell(int x, int y, std::mt19937& gen, FireModelParameters &parameters, int raster_value = 0);
    ~FireCell();

    CellState GetIgnitionState();
    CellState GetCellState() { return cell_state_; }
    CellState GetCellInitialState() { return cell_initial_state_; }
    bool IsBurning() { return GetIgnitionState() == CellState::GENERIC_BURNING; }
    bool FloodTick();
    bool IsFlooded();

    bool CanIgnite();
    void Ignite();
    std::pair<bool, bool> ShouldEmitNextParticles();
    VirtualParticle EmitConvectionParticle();
    RadiationParticle EmitRadiationParticle();
    Uint32 GetMappedColor();

    void Tick();
    void burn();
    bool ShouldIgnite();
    void Flood();
    void Extinguish();
    void ShowInfo(int rows, int cols);
private:
    FireModelParameters &parameters_;

    double burning_duration_;
    double ticking_duration_;
    double burning_tick_;
    bool has_burned_down_;
    int last_burning_duration_;
    double flood_duration_;
    double flood_timer_;
    double tau_ign_;
    double tau_ign_start_;
    double tau_ign_tmp_;
    int x_; // Start of the x coordinate in meters (m)
    int y_; // Start of the y coordinate in meters (m)
    int num_convection_particles_;
    int num_radiation_particles;
    int convection_particle_emission_threshold_;
    int radiation_particle_emission_threshold_;
    SDL_Surface* surface_;
    ICell* cell_;
    ICell* mother_cell_;
    CellState cell_state_;
    CellState cell_initial_state_;

    // Random Generator for the particles
    std::mt19937 gen_;
    std::uniform_real_distribution<> real_dis_;

    ICell *GetCell();

    void SetCellState(CellState cell_state);
    void ResetFloodedCell();
};


#endif //ROSHAN_FIREMODEL_FIRECELL_H
