//
// Created by nex on 08.06.23.
//

#include "firemodel_gridmap.h"

GridMap::GridMap(std::shared_ptr<Wind> wind, FireModelParameters &parameters,
                 std::vector<std::vector<int>>* rasterData) :
                 parameters_(parameters),
                 buffer_( rasterData->size() * ((rasterData->empty()) ? 0 : (*rasterData)[0].size()) * 30){
    cols_ = rasterData->size(); //x
    rows_ = (rasterData->empty()) ? 0 : (*rasterData)[0].size(); //y
    // Cols and Rows are swapped in Renderer to match GEO representation
    wind_ = wind;

    // Generate a normally-distributed random number for phi_r
    gen_ = std::mt19937(rd_());

    cells_ = std::vector<std::vector<std::shared_ptr<FireCell>>>(cols_, std::vector<std::shared_ptr<FireCell>>(rows_));
    explored_map_ = std::vector<std::vector<int>>(cols_, std::vector<int>(rows_, 0));
    fire_map_ = std::vector<std::vector<int>>(cols_, std::vector<int>(rows_, 0));

    for (int x = 0; x < cols_; ++x) {
        for (int y = 0; y < rows_; ++y) {
            cells_[x][y] = std::make_shared<FireCell>(x, y, gen_, parameters_, (*rasterData)[x][y]);
        }
    }

    num_cells_ = cols_ * rows_;
    parameters_.SetGridNxNy(cols_, rows_);
    num_burned_cells_ = 0;
    num_unburnable_ = this->GetNumUnburnableCells();
    virtual_particles_.reserve(100000);
    radiation_particles_.reserve(100000);
    ticking_cells_.reserve(100000);
    burning_cells_.reserve(100000);
    changed_cells_.reserve(100000);
    flooded_cells_.reserve(1000);

    noise_generated_ = false;
}

template <typename ParticleType>
void GridMap::UpdateVirtualParticles(std::vector<ParticleType>& particles, std::vector<std::vector<bool>>& visited_cells) {
    std::vector<ParticleType*> particles_to_remove;

    for (size_t part = 0; part < particles.size(); ++part) {
        auto& particle = particles[part];

        if constexpr (std::is_same<ParticleType, VirtualParticle>::value) {
            particle.UpdateState(*wind_, parameters_.GetDt(), buffer_);
        } else {
            particle.UpdateState(parameters_.GetDt(), buffer_);
        }

        double x, y;
        particle.GetPosition(x, y);
        int i, j;
        parameters_.ConvertRealToGridCoordinates(x, y, i, j);

        if (!IsPointInGrid(i, j) || !particle.IsCapableOfIgnition()) {
            {
                particles_to_remove.push_back(&particle);
            }
            continue;
        }

        Point p = Point(i, j);

        {
            visited_cells[p.x_][p.y_] = true;
            if (CellCanIgnite(p.x_, p.y_)) {
                ticking_cells_.insert(p);
            }
        }
    }

    particles.erase(
            std::remove_if(particles.begin(), particles.end(),
                           [&](ParticleType& p) {
                               return std::find(particles_to_remove.begin(), particles_to_remove.end(), &p) != particles_to_remove.end();
                           }),
            particles.end()
    );
}

void GridMap::UpdateParticles() {
    std::vector<std::vector<bool>> visited_cells(cols_, std::vector<bool>(rows_, false));
    UpdateVirtualParticles(virtual_particles_, visited_cells);
    UpdateVirtualParticles(radiation_particles_, visited_cells);

    for (auto it = ticking_cells_.begin(); it != ticking_cells_.end(); ) {
        // If cell is not visited, remove it from ticking cells
        if (!visited_cells[it->x_][it->y_]) {
            it = ticking_cells_.erase(it);
        } else {
            ++it;
        }
    }
}

GridMap::~GridMap() {
}

void GridMap::IgniteCell(int x, int y) {
    cells_[x][y]->Ignite();
    burning_cells_.insert(Point(x, y));
    changed_cells_.emplace_back(x, y);
    ticking_cells_.erase(Point(x, y));
}

void GridMap::UpdateCells() {
    // Iterate over flooded cells and extinguish them
    for (auto it = flooded_cells_.begin(); it != flooded_cells_.end(); ) {
        int x = it->x_;
        int y = it->y_;
        auto& cell = cells_[x][y];
        // Let the water fade
        if (cell->FloodTick()) {
            // The cell is no longer flooded
            it = flooded_cells_.erase(it);
        } else {
            ++it;
        }
        changed_cells_.emplace_back(x, y);
    }
    // Iterate over ticking cells and ignite them if their ignition time has come
    for (auto it = ticking_cells_.begin(); it != ticking_cells_.end(); ) {
        int x = it->x_;
        int y = it->y_;
        auto& cell = cells_[x][y];
        cell->Tick();
        if (cell->ShouldIgnite()) {
            // The cell has ignited, so it is no longer ticking
            it = ticking_cells_.erase(it);
            // Ignite the cell
            IgniteCell(x, y);
        } else {
            ++it;
        }
    }
    // Iterate over burning cells and let them burn
    for (auto it = burning_cells_.begin(); it != burning_cells_.end(); ) {
        int x = it->x_;
        int y = it->y_;
        auto& cell = cells_[x][y];
        cell->burn();
        bool cell_has_changed = false;
        auto should_emit_particle = cell->ShouldEmitNextParticles();
        if (parameters_.emit_convective_ && should_emit_particle.first) {
            virtual_particles_.push_back(std::move(cell->EmitConvectionParticle()));
            cell_has_changed = true;
        }
        if (parameters_.emit_radiation_ && should_emit_particle.second) {
            radiation_particles_.push_back(std::move(cell->EmitRadiationParticle()));
            cell_has_changed = true;
        }
        if (cell_has_changed)
            changed_cells_.emplace_back(x, y);

        if (cell->GetIgnitionState() == CellState::GENERIC_BURNED) {
            // The cell has burned out, so it is no longer burning
            num_burned_cells_++;
            it = burning_cells_.erase(it);
            changed_cells_.emplace_back(x, y);
        } else {
            ++it;
        }
    }
}

bool GridMap::WaterDispension(int x, int y) {
    if (!IsPointInGrid(x, y))
        // Drone is outside the grid so nothing happens
        return false;
    if (GetCellState(x, y) == CellState::GENERIC_BURNING) {
        burning_cells_.erase(Point(x, y));
        EraseParticles(x, y);
        changed_cells_.push_back(Point(x, y));
        
        flooded_cells_.insert(Point(x, y));
        cells_[x][y]->Flood();
        flooded_cells_.insert(Point(x, y));
        changed_cells_.emplace_back(x, y);
        // Drone is inside the grid and the cell is burning, so extinguish the cell and return true
        return true;
    } else {
        if (!cells_[x][y]->IsFlooded()) {
            flooded_cells_.insert(Point(x, y));
        }
        cells_[x][y]->Flood();
        EraseParticles(x, y);
        changed_cells_.emplace_back(Point(x, y));
        // There was no fire in the cell so flood the cell and return false
        return false;
    }
}

void GridMap::EraseParticles(int x, int y) {
    ticking_cells_.erase(Point(x, y));
    //Erase all particles that are on that cell
    std::function<void(double&, double&, int&, int&)> convertr =
            std::bind(&FireModelParameters::ConvertRealToGridCoordinates, &parameters_, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4);
    virtual_particles_.erase(
            std::remove_if(virtual_particles_.begin(), virtual_particles_.end(),
                           [x, y, &convertr](const VirtualParticle& particle) {
                               double particle_x, particle_y;
                               particle.GetPosition(particle_x, particle_y);
                               int i, j;
                               convertr(particle_x, particle_y, i, j);
                               return i == x && j == y;
                           }),
            virtual_particles_.end());
    radiation_particles_.erase(
            std::remove_if(radiation_particles_.begin(), radiation_particles_.end(),
                           [x, y, &convertr](const RadiationParticle& particle) {
                               double particle_x, particle_y;
                               particle.GetPosition(particle_x, particle_y);
                               int i, j;
                               convertr(particle_x, particle_y, i, j);
                               return i == x && j == y;
                           }),
            radiation_particles_.end());
}

void GridMap::ExtinguishCell(int x, int y) {
    cells_[x][y]->Extinguish();
    burning_cells_.erase(Point(x, y));

    EraseParticles(x, y);

    changed_cells_.push_back(Point(x, y));
}

std::vector<std::vector<std::vector<int>>> GridMap::GetDroneView(std::shared_ptr<DroneAgent> drone) {
    int drone_view_radius = drone->GetViewRange();
    std::pair<int, int> drone_position = drone->GetGridPosition();
    int size = drone_view_radius + 1;

    // Initialisiere eine 3D-Matrix: [status_type][x][y]
    std::vector<std::vector<std::vector<int>>> view(2, std::vector<std::vector<int>>(size, std::vector<int>(size, -1)));

    int drone_view_radius_2 = drone_view_radius / 2;
    for (int y = drone_position.second - drone_view_radius_2; y <= drone_position.second + drone_view_radius_2; ++y) {
        for (int x = drone_position.first - drone_view_radius_2; x <= drone_position.first + drone_view_radius_2; ++x) {
            int new_i = y - drone_position.second + drone_view_radius_2;
            int new_j = x - drone_position.first + drone_view_radius_2;
            if (IsPointInGrid(x, y)) {
                // Setze Zellstatus (Status 0)
                view[0][new_i][new_j] = cells_[x][y]->GetCellState();

                // Setze Feuerstatus (Status 1)
                if (cells_[x][y]->IsBurning())
                    view[1][new_i][new_j] = 1;
                else
                    view[1][new_i][new_j] = 0; // oder ein anderer Wert, der "kein Feuer" darstellt
            }
        }
    }

    return view;
}

int GridMap::UpdateLastSeenTime(int x, int y) {
    int difference = parameters_.GetExplorationTime() - explored_map_[x][y];
    explored_map_[x][y] = parameters_.GetExplorationTime();
    return difference;
}

void GridMap::UpdateCellDiminishing() {
    for (int x = 0; x < cols_; ++x) {
        for (int y = 0; y < rows_; ++y) {
            if (explored_map_[x][y] > 0) {
                explored_map_[x][y]--;
            }
        }
    }
}

void GridMap::UpdateExploredAreaFromDrone(std::shared_ptr<DroneAgent> drone) {
    std::pair<int, int> drone_position = drone->GetGridPosition();
    int drone_view_radius = drone->GetViewRange();
    int drone_view_radius_2 = drone_view_radius / 2;
    int combined_difference = 0;

    for (int x = drone_position.first - drone_view_radius_2; x <= drone_position.first + drone_view_radius_2; ++x) {
        for (int y = drone_position.second - drone_view_radius_2; y <= drone_position.second + drone_view_radius_2; ++y) {
            if (IsPointInGrid(x, y)) {
                combined_difference += UpdateLastSeenTime(x, y);
                fire_map_[x][y] = cells_[x][y]->IsBurning() ? 1 : 0;
            }
        }
    }
    drone->SetExploreDifference(combined_difference);
}

// Calculates the number of unburnable cells
int GridMap::GetNumUnburnableCells() const {
    int num_unburnable_cells = 0;
    for (int x = 0; x < cols_; ++x) {
        for (int y = 0; y < rows_; ++y) {
            if (!cells_[x][y]->CanIgnite()) {
                num_unburnable_cells++;
            }
        }
    }
    return num_unburnable_cells;
}

double GridMap::PercentageUnburnable() const {
    return (double)num_unburnable_ / (double)num_cells_;
}

// Calculates the percentage of burned cells
double GridMap::PercentageBurned() const {
    return (double)num_burned_cells_ / (double)num_cells_;
}

//Returns whether there are still burning fires on the map
bool GridMap::IsBurning() const {
    return !(burning_cells_.empty() && virtual_particles_.empty() && radiation_particles_.empty());
}

// Calculates the percentage of burning cells
double GridMap::PercentageBurning() const {
    return (double)(burning_cells_.size()) / (double)num_cells_;
}

bool GridMap::CanStartFires(int num_fires) const {
    return (num_cells_ - (num_unburnable_ + burning_cells_.size() + num_burned_cells_)) >= num_fires;
}

void GridMap::SetCellNoise(CellState state, int noise_level, int noise_size) {
    switch (state) {
        case GENERIC_UNBURNED:
            CellGenericUnburned::SetDefaultNoiseLevel(noise_level);
            CellGenericUnburned::SetDefaultNoiseSize(noise_size);
            break;
        case GENERIC_BURNING:
            CellGenericBurning::SetDefaultNoiseLevel(noise_level);
            CellGenericBurning::SetDefaultNoiseSize(noise_size);
            break;
        case GENERIC_BURNED:
            CellGenericBurned::SetDefaultNoiseLevel(noise_level);
            CellGenericBurned::SetDefaultNoiseSize(noise_size);
            break;
        case LICHENS_AND_MOSSES:
            CellLichensAndMosses::SetDefaultNoiseLevel(noise_level);
            CellLichensAndMosses::SetDefaultNoiseSize(noise_size);
            break;
        case LOW_GROWING_WOODY_PLANTS:
            CellLowGrowingWoodyPlants::SetDefaultNoiseLevel(noise_level);
            CellLowGrowingWoodyPlants::SetDefaultNoiseSize(noise_size);
            break;
        case NON_AND_SPARSLEY_VEGETATED:
            CellNonAndSparsleyVegetated::SetDefaultNoiseLevel(noise_level);
            CellNonAndSparsleyVegetated::SetDefaultNoiseSize(noise_size);
            break;
        case OUTSIDE_AREA:
            CellOutsideArea::SetDefaultNoiseLevel(noise_level);
            CellOutsideArea::SetDefaultNoiseSize(noise_size);
            break;
        case PERIODICALLY_HERBACEOUS:
            CellPeriodicallyHerbaceous::SetDefaultNoiseLevel(noise_level);
            CellPeriodicallyHerbaceous::SetDefaultNoiseSize(noise_size);
            break;
        case PERMANENT_HERBACEOUS:
            CellPermanentHerbaceous::SetDefaultNoiseLevel(noise_level);
            CellPermanentHerbaceous::SetDefaultNoiseSize(noise_size);
            break;
        case SEALED:
            CellSealed::SetDefaultNoiseLevel(noise_level);
            CellSealed::SetDefaultNoiseSize(noise_size);
            break;
        case SNOW_AND_ICE:
            CellSnowAndIce::SetDefaultNoiseLevel(noise_level);
            CellSnowAndIce::SetDefaultNoiseSize(noise_size);
            break;
        case WATER:
            CellWater::SetDefaultNoiseLevel(noise_level);
            CellWater::SetDefaultNoiseSize(noise_size);
            break;
        case WOODY_BROADLEAVED_DECIDUOUS_TREES:
            CellWoodyBroadleavedDeciduousTrees::SetDefaultNoiseLevel(noise_level);
            CellWoodyBroadleavedDeciduousTrees::SetDefaultNoiseSize(noise_size);
            break;
        case WOODY_BROADLEAVED_EVERGREEN_TREES:
            CellWoodyBroadleavedEvergreenTrees::SetDefaultNoiseLevel(noise_level);
            CellWoodyBroadleavedEvergreenTrees::SetDefaultNoiseSize(noise_size);
            break;
        case WOODY_NEEDLE_LEAVED_TREES:
            CellWoodyNeedleLeavedTrees::SetDefaultNoiseLevel(noise_level);
            CellWoodyNeedleLeavedTrees::SetDefaultNoiseSize(noise_size);
            break;
        case GENERIC_FLOODED:
            CellGenericFlooded::SetDefaultNoiseLevel(noise_level);
            CellGenericFlooded::SetDefaultNoiseSize(noise_size);
            break;
        default:
            throw std::runtime_error("FireCell::GetCell() called on a celltype that is not defined");
    }
}

void GridMap::GenerateNoiseMap() {
    if (!this->noise_generated_ && parameters_.has_noise_){
        for(auto cell_row : cells_) {
            for(auto cell : cell_row) {
                if(cell->HasNoise()){
                    cell->GenerateNoiseMap();
                }
            }
        }
        this->noise_generated_ = true;
    }
}

std::vector<std::pair<int, int>> GridMap::GetMooreNeighborhood(int x, int y) const {
    std::vector<std::pair<int, int>> neighborhood;
    for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
            if (i == 0 && j == 0) {
                continue;
            }
            int new_x = x + i;
            int new_y = y + j;
            if (IsPointInGrid(new_x, new_y)) {
                neighborhood.emplace_back(new_x, new_y);
            }
        }
    }
    return neighborhood;
}

std::vector<std::vector<int>> GridMap::GetExploredMap(int size) {
    if(size == 0) {
        size = parameters_.GetExplorationMapSize();
    }
    return InterpolationResize(explored_map_, size, size);
}

std::vector<std::vector<int>> GridMap::GetFireMap(int size) {
    if(size == 0) {
        size = parameters_.GetFireMapSize();
    }
    return PoolingResize(fire_map_, size, size);
}
