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

#include "src/models/firespin/cell_classes/cell_generic_burned.h"
#include "src/models/firespin/cell_classes/cell_generic_unburned.h"
#include "src/models/firespin/cell_classes/cell_generic_burning.h"
#include "src/models/firespin/cell_classes/cell_lichens_and_mosses.h"
#include "src/models/firespin/cell_classes/cell_low_growing_woody_plants.h"
#include "src/models/firespin/cell_classes/cell_non_and_sparsley_vegetated.h"
#include "src/models/firespin/cell_classes/cell_outside_area.h"
#include "src/models/firespin/cell_classes/cell_periodically_herbaceous.h"
#include "src/models/firespin/cell_classes/cell_permanent_herbaceous.h"
#include "src/models/firespin/cell_classes/cell_sealed.h"
#include "src/models/firespin/cell_classes/cell_snow_and_ice.h"
#include "src/models/firespin/cell_classes/cell_water.h"
#include "src/models/firespin/cell_classes/cell_woody_breadleaved_deciduous_trees.h"
#include "src/models/firespin/cell_classes/cell_woody_broadleaved_evergreen_trees.h"
#include "src/models/firespin/cell_classes/cell_woody_needle_leaved_trees.h"
#include "src/models/firespin/cell_classes/cell_generic_flooded.h"


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
    bool IsBurning() const { return cell_state_ == CellState::GENERIC_BURNING; }
    void Ignite();
    std::pair<bool, bool> ShouldEmitNextParticles();
    VirtualParticle EmitConvectionParticle();
    RadiationParticle EmitRadiationParticle();
    Uint32 GetMappedColor();
    std::vector<std::vector<int>>& GetNoiseMap();

    void Tick();
    void burn();
    bool ShouldIgnite();
    void Flood();
    bool WasFlooded() { return was_flooded_; };
    void Extinguish();
    void ShowInfo(int rows, int cols);

    //Used for Rendering Only
    void GenerateNoiseMap();
    bool HasNoise();
    int GetNoiseLevel();
    int GetNoiseSize();
    void SetDefaultNoise(int noise_level, int noise_size);

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

    //Texture Test
    std::vector<std::vector<int>> noise_map_;

    // Random Generator for the particles
    std::mt19937 gen_;
    std::uniform_real_distribution<> real_dis_;

    ICell *GetCell();

    bool was_flooded_ = false;
    void SetCellState(CellState cell_state);
    void ResetFloodedCell();
};


#endif //ROSHAN_FIREMODEL_FIRECELL_H
