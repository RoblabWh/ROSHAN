//
// Created by nex on 10.06.23.
//

#include <iostream>
#include <array>
#include "firemodel_firecell.h"

// Singleton ICell instances — one per CellState, lazily initialized.
// All ICell subclasses are effectively immutable after construction,
// so sharing a single instance per type eliminates heap alloc/dealloc
// on every SetCellState call during fire propagation.
static constexpr int kNumCellStates = CELL_STATE_COUNT;  // 17
static std::array<ICell*, kNumCellStates> cell_singletons_{};
static bool singletons_valid_ = false;

static ICell* CreateCellForState(CellState state) {
    switch (state) {
        case OUTSIDE_GRID:                    return nullptr;  // No cell class for grid boundary
        case GENERIC_UNBURNED:                return new CellGenericUnburned();
        case GENERIC_BURNING:                 return new CellGenericBurning();
        case GENERIC_BURNED:                  return new CellGenericBurned();
        case LICHENS_AND_MOSSES:              return new CellLichensAndMosses();
        case LOW_GROWING_WOODY_PLANTS:        return new CellLowGrowingWoodyPlants();
        case NON_AND_SPARSLEY_VEGETATED:      return new CellNonAndSparsleyVegetated();
        case OUTSIDE_AREA:                    return new CellOutsideArea();
        case PERIODICALLY_HERBACEOUS:         return new CellPeriodicallyHerbaceous();
        case PERMANENT_HERBACEOUS:            return new CellPermanentHerbaceous();
        case SEALED:                          return new CellSealed();
        case SNOW_AND_ICE:                    return new CellSnowAndIce();
        case WATER:                           return new CellWater();
        case WOODY_BROADLEAVED_DECIDUOUS_TREES: return new CellWoodyBroadleavedDeciduousTrees();
        case WOODY_BROADLEAVED_EVERGREEN_TREES: return new CellWoodyBroadleavedEvergreenTrees();
        case WOODY_NEEDLE_LEAVED_TREES:       return new CellWoodyNeedleLeavedTrees();
        case GENERIC_FLOODED:                 return new CellGenericFlooded();
        default: throw std::runtime_error("Unknown CellState in CreateCellForState");
    }
}

static void EnsureCellSingletons() {
    if (singletons_valid_) return;
    for (int i = 0; i < kNumCellStates; ++i) {
        delete cell_singletons_[i];
        cell_singletons_[i] = CreateCellForState(static_cast<CellState>(i));
    }
    singletons_valid_ = true;
}

void InvalidateCellSingletons() {
    // Called when noise defaults change (SetCellNoise from UI).
    // Next GetCell() call will recreate them with updated defaults.
    singletons_valid_ = false;
}

FireCell::FireCell(int x, int y, FireModelParameters &parameters, int raster_value)
: parameters_(parameters){
    //Cell State
    cell_initial_state_ = CellState(raster_value);
    cell_state_ = CellState(raster_value);
    cell_ = GetCell();
    mother_cell_ = GetCell();
    has_burned_down_ = false;

    // Cell Parameters
    x_ = x * parameters_.GetCellSize();
    y_ = y * parameters_.GetCellSize();
    ticking_duration_ = 0;
    burning_tick_ = 0;

    num_convection_particles_ = mother_cell_->GetNumConvectionParticles();
    num_radiation_particles = mother_cell_->GetNumRadiationParticles();

    if (parameters_.map_is_uniform_) {
        burning_duration_ = parameters_.GetCellBurningDuration();
        tau_ign_ = parameters_.GetIgnitionDelayTime();
    } else {
        burning_duration_ = cell_->GetCellBurningDuration();
        tau_ign_ = cell_->GetIgnitionDelayTime();
    }

    // Initialize random number generator
    real_dis_ = std::uniform_real_distribution<>(0.0, 1.0);
    std::uniform_real_distribution<> dis(0.1, 0.2);
    std::uniform_int_distribution<> sign_dis(-1, 1);
    burning_duration_ += sign_dis(parameters_.gen_) * burning_duration_ * dis(parameters_.gen_);
    tau_ign_ += sign_dis(parameters_.gen_) * tau_ign_ * dis(parameters_.gen_);
    tau_ign_start_ = tau_ign_;

    convection_particle_emission_threshold_ = (burning_duration_ - 1) / num_convection_particles_;
    if (convection_particle_emission_threshold_ < 1)
        convection_particle_emission_threshold_ = 1;

    radiation_particle_emission_threshold_ = (burning_duration_ - 1) / num_radiation_particles;
    if (radiation_particle_emission_threshold_ < 1)
        radiation_particle_emission_threshold_ = 1;

    flood_duration_ = parameters_.GetFloodDuration();
    flood_timer_ = flood_duration_;
    tau_ign_tmp_ = 0;
}

CellState FireCell::GetIgnitionState() {
    return cell_state_;
}

ICell *FireCell::GetCell() {
    EnsureCellSingletons();
    return cell_singletons_[static_cast<int>(cell_state_)];
}

void FireCell::Ignite() {
    if (cell_state_ == GENERIC_BURNING || cell_state_ == GENERIC_BURNED) {
        throw std::runtime_error("FireCell::Ignite() called on a cell that is not unburned");
    }
    SetCellState(GENERIC_BURNING);
    ticking_duration_ = tau_ign_;
}

VirtualParticle FireCell::EmitConvectionParticle() {
    double x_pos_rnd = real_dis_(parameters_.gen_);
    double y_pos_rnd = real_dis_(parameters_.gen_);
    double cell_size = (parameters_.GetCellSize());
    double x_pos = x_ + (cell_size * x_pos_rnd);
    double y_pos = y_ + (cell_size * y_pos_rnd);
    VirtualParticle particle(x_pos, y_pos, parameters_.GetTauMemVirt(), parameters_.GetYStVirt(),
                             parameters_.GetYLimVirt(), parameters_.GetFlVirt(), parameters_.GetC0Virt(),
                             parameters_.GetLt());

    return particle;
}

RadiationParticle FireCell::EmitRadiationParticle() {
    double x_pos_rnd = real_dis_(parameters_.gen_);
    double y_pos_rnd = real_dis_(parameters_.gen_);
    double cell_size = (parameters_.GetCellSize());
    double x_pos = x_ + (cell_size * x_pos_rnd);
    double y_pos = y_ + (cell_size * y_pos_rnd);
    auto radiation_length = mother_cell_->GetRadiationLength();
    RadiationParticle radiation_particle(x_pos, y_pos, radiation_length.first, radiation_length.second, mother_cell_->GetSf0Mean(), mother_cell_->GetSf0Std(),
                                         parameters_.GetYStRad(), parameters_.GetYLimRad(), parameters_.gen_);

    return radiation_particle;
}

void FireCell::Tick() {
    //throw an error if cellState ignited cells do not tick
    if (cell_state_ == GENERIC_BURNING || cell_state_ == GENERIC_BURNED || cell_state_ == GENERIC_FLOODED) {
        throw std::runtime_error("FireCell::Tick() called on a cell that has CELL_STATE: " + CellStateToString(cell_state_));
    }
    if (tau_ign_ != -1) {
        // The cell ticks until it is ignited
        ticking_duration_ += parameters_.GetDt();
    }
}

std::pair<bool, bool> FireCell::ShouldEmitNextParticles() const {
    // Converting the burning_tick_ to an int before the modulo operation
    int burning_tick_int = static_cast<int>(burning_tick_);

    // Check if the last burning duration is not equal to the current burning tick
    bool is_new_burning_tick = ceil(last_burning_duration_) != floor(burning_tick_);

    // It's a new burning tick & the burning tick is a multiple of the particle emission threshold
    bool emit_condition_convection = is_new_burning_tick && (burning_tick_int % convection_particle_emission_threshold_ == 0);
    bool emit_condition_radiation = is_new_burning_tick && (burning_tick_int % radiation_particle_emission_threshold_ == 0);

    // First is for the convection particles, second is for the radiation particles
    return std::make_pair(emit_condition_convection, emit_condition_radiation);
}

void FireCell::burn() {
    last_burning_duration_ = burning_tick_;
    burning_duration_ -= parameters_.GetDt();
    burning_tick_ += parameters_.GetDt();
    if (burning_duration_ <= 0) {
        SetCellState(GENERIC_BURNED);
        has_burned_down_ = true;
    }
}

bool FireCell::CanIgnite() {
    if (cell_state_ == GENERIC_BURNING || cell_state_ == GENERIC_BURNED  ||
        cell_state_ == SNOW_AND_ICE || cell_state_ == OUTSIDE_AREA || cell_state_ == WATER ||
        cell_state_ == NON_AND_SPARSLEY_VEGETATED || cell_state_ == GENERIC_FLOODED) {
        return false;
    }
    return true;
}

bool FireCell::ShouldIgnite() {
    if (ticking_duration_ >= tau_ign_) {
        return true;
    }
    return false;
}

void FireCell::Extinguish() {
    if (has_burned_down_) {
        SetCellState(GENERIC_BURNED);
    } else {
        SetCellState(cell_initial_state_);
    }
}

bool FireCell::FloodTick(){
    if (cell_state_ != GENERIC_FLOODED)
        throw std::runtime_error("FireCell::FloodTick() called on a cell that has CELL_STATE: " + CellStateToString(cell_state_));
    flood_timer_ += parameters_.GetDt();
    if (!IsFlooded()) {
        Extinguish();
        ResetFloodedCell();
        return true;
    }
    return false;
}

void FireCell::Flood() {
    if (cell_state_ != GENERIC_FLOODED) {
        tau_ign_tmp_ = tau_ign_start_;
        tau_ign_ = -1;
        if (cell_state_ == GENERIC_BURNING) {
            Extinguish();
            ticking_duration_ = 0;
        }
        SetCellState(GENERIC_FLOODED);
        was_flooded_ = true;
    }
    flood_timer_ = 0;
}

void FireCell::ResetFloodedCell() {
    tau_ign_ = tau_ign_tmp_;
    flood_timer_ = flood_duration_;
}

bool FireCell::IsFlooded() {
    // returns true if the cell is flooded
    if(flood_timer_ >= flood_duration_) {
        return false;
    }
    return true;
}

void FireCell::SetCellState(CellState cell_state) {
    cell_state_ = cell_state;
    cell_ = GetCell();
    // Cache the blended color for deterministic states
    if (cell_state_ == GENERIC_BURNED || cell_state_ == GENERIC_FLOODED) {
        ComputeAndCacheBlendedColor();
    } else {
        has_cached_color_ = false;
    }
}

void FireCell::ComputeAndCacheBlendedColor() {
    Uint32 mapped_cell_color = cell_->GetMappedColor();
    Uint32 mapped_mother_cell_color = mother_cell_->GetMappedColor();

    // Decompose using ARGB8888 bit layout
    Uint8 burned_a = (mapped_cell_color >> 24) & 0xFF;
    Uint8 burned_r = (mapped_cell_color >> 16) & 0xFF;
    Uint8 burned_g = (mapped_cell_color >> 8) & 0xFF;
    Uint8 burned_b = mapped_cell_color & 0xFF;

    Uint8 mother_r = (mapped_mother_cell_color >> 16) & 0xFF;
    Uint8 mother_g = (mapped_mother_cell_color >> 8) & 0xFF;
    Uint8 mother_b = mapped_mother_cell_color & 0xFF;

    int weight_a = (cell_state_ == GENERIC_BURNED) ? 20 : 40;
    int mother_weight = 1;
    int total_weight = weight_a + mother_weight;

    Uint8 blended_r = (burned_r * weight_a + mother_r * mother_weight) / total_weight;
    Uint8 blended_g = (burned_g * weight_a + mother_g * mother_weight) / total_weight;
    Uint8 blended_b = (burned_b * weight_a + mother_b * mother_weight) / total_weight;

    cached_mapped_color_ = (static_cast<Uint32>(255) << 24) |
                           (static_cast<Uint32>(blended_r) << 16) |
                           (static_cast<Uint32>(blended_g) << 8) |
                           static_cast<Uint32>(blended_b);
    has_cached_color_ = true;
}

void FireCell::ShowInfo(int rows, int cols) {
    ImGui::Text("Current Cell State");
    ImVec4 color = cell_->GetImVecColor();
    double cellsize = parameters_.GetCellSize();
    double total_width = cols * cellsize;
    double total_height = rows * cellsize;
    double relative_x = (2.0 * (x_ + 5) / total_width) - 1; // +0.5 to get the center of the cell
    double relative_y = (2.0 * (y_ + 5) / total_height) - 1;
    ImGui::Text("Relative Position(Center): %f, %f", relative_x, relative_y);
    ImGui::ColorButton("MyColor##3", {color.x / 255, color.y / 255, color.z / 255, color.w / 255}, ImGuiColorEditFlags_NoTooltip | ImGuiColorEditFlags_NoPicker);
    ImGui::SameLine();
    ImGui::TextUnformatted(CellStateToString(cell_state_).c_str());
    ImGui::Text("Mother Cell State");
    color = mother_cell_->GetImVecColor();
    ImGui::ColorButton("MyColor##4", {color.x / 255, color.y / 255, color.z / 255, color.w / 255}, ImGuiColorEditFlags_NoTooltip | ImGuiColorEditFlags_NoPicker);
    ImGui::SameLine();
    ImGui::TextUnformatted(CellStateToString(cell_initial_state_).c_str());
    ImGui::Text("%0.f m x %0.f m", parameters_.GetCellSize(), parameters_.GetCellSize());
    ImGui::Text("Burning duration: %.2f", burning_duration_);
    ImGui::Text("Ticking duration: %.2f", ticking_duration_);
    ImGui::Text("Tau ign: %.2f", tau_ign_);
    ImGui::Text("Noise Level: %d", GetNoiseLevel());
    ImGui::Text("Noise Size: %d", GetNoiseSize());
}

Uint32 FireCell::GetMappedColor() {
    if (has_cached_color_) {
        return cached_mapped_color_;
    }
    return cell_->GetMappedColor();
}

bool FireCell::HasNoise() {
    return cell_->HasNoise();
}

int FireCell::GetNoiseLevel() {
    return cell_->GetNoiseLevel();
}

int FireCell::GetNoiseSize() {
    return cell_->GetNoiseSize();
}

void FireCell::GenerateNoiseMap() {
    int noise_level = cell_->GetNoiseLevel();
    int size = cell_->GetNoiseSize();
    noise_map_.resize(size, std::vector<int>(size));
    std::uniform_int_distribution<> dist(-noise_level, noise_level);

    std::seed_seq seq{static_cast<unsigned int>(parameters_.seed_),
                      static_cast<unsigned int>(x_),
                      static_cast<unsigned int>(y_)};
    std::mt19937 gen(seq);
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            noise_map_[y][x] = dist(gen);
        }
    }
}

std::vector<std::vector<int>>& FireCell::GetNoiseMap() {
    return noise_map_;
}

FireCell::~FireCell() {
    // cell_ and mother_cell_ point to shared singletons — do not delete
}

void FireCell::Reset(int raster_value) {
    cell_initial_state_ = CellState(raster_value);
    cell_state_ = cell_initial_state_;

    cell_ = GetCell();
    mother_cell_ = GetCell();

    has_burned_down_ = false;
    ticking_duration_ = 0;
    burning_tick_ = 0;

    num_convection_particles_ = mother_cell_->GetNumConvectionParticles();
    num_radiation_particles = mother_cell_->GetNumRadiationParticles();

    if (parameters_.map_is_uniform_) {
        burning_duration_ = parameters_.GetCellBurningDuration();
        tau_ign_ = parameters_.GetIgnitionDelayTime();
    } else {
        burning_duration_ = cell_->GetCellBurningDuration();
        tau_ign_ = cell_->GetIgnitionDelayTime();
    }

    real_dis_ = std::uniform_real_distribution<>(0.0, 1.0);
    std::uniform_real_distribution<> dis(0.1, 0.2);
    std::uniform_int_distribution<> sign_dis(-1, 1);
    burning_duration_ += sign_dis(parameters_.gen_) * burning_duration_ * dis(parameters_.gen_);
    tau_ign_ += sign_dis(parameters_.gen_) * tau_ign_ * dis(parameters_.gen_);
    tau_ign_start_ = tau_ign_;

    convection_particle_emission_threshold_ = (burning_duration_ - 1) / num_convection_particles_;
    if (convection_particle_emission_threshold_ < 1)
        convection_particle_emission_threshold_ = 1;

    radiation_particle_emission_threshold_ = (burning_duration_ - 1) / num_radiation_particles;
    if (radiation_particle_emission_threshold_ < 1)
        radiation_particle_emission_threshold_ = 1;

    flood_duration_ = parameters_.GetFloodDuration();
    flood_timer_ = flood_duration_;
    tau_ign_tmp_ = 0;
    was_flooded_ = false;
    has_cached_color_ = false;
}
