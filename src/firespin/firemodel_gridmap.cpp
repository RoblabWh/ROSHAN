//
// Created by nex on 08.06.23.
//

#include "firemodel_gridmap.h"

#include <algorithm>
#include <utility>

GridMap::GridMap(std::shared_ptr<Wind> wind, FireModelParameters &parameters,
                 std::vector<std::vector<int>>* rasterData) :
                 parameters_(parameters),
                 buffer_(rasterData->size() * ((rasterData->empty()) ? 0 : (*rasterData)[0].size()) * 30, parameters) {
    int cols = static_cast<int>(rasterData->size()); //x NO IT'S Y
    int rows = (rasterData->empty()) ? 0 : static_cast<int>((*rasterData)[0].size()); //y NO IT'S X (this confusion stems from the map being transposed in memory)
    // Cols and Rows are swapped in Renderer to match GEO representation
    wind_ = std::move(wind);
    cells_ = std::vector<std::vector<std::shared_ptr<FireCell>>>(rows, std::vector<std::shared_ptr<FireCell>>(cols));
    explored_map_ = std::vector<std::vector<int>>(rows, std::vector<int>(cols, 0));
    step_explored_map_ = std::vector<std::vector<int>>(rows, std::vector<int>(cols, 0));
    fire_map_ = std::vector<std::vector<int>>(rows, std::vector<int>(cols, 0));
    visited_cells_ = std::vector<std::vector<bool>>(rows, std::vector<bool>(cols, false));

//#pragma omp parallel for collapse(2)
    for (int x = 0; x < rows; ++x) {
        for (int y = 0; y < cols; ++y) {
            cells_[x][y] = std::make_shared<FireCell>(x, y, parameters_, (*rasterData)[y][x]);
        }
    }
    cols_ = cols;
    rows_ = rows;
    num_cells_ = cols_ * rows_;
    // Precalculate ref for max fires;
    n_ref_fires_ = static_cast<int>(std::ceil(static_cast<float>(this->num_cells_) * parameters_.GetFirePercentage()));
    parameters_.SetGridNxNy(rows_, cols_);
    y_off_ = 2 * (1 - ((cols_ - 0.5) / cols_));
    x_off_ = 2 * (1 - ((rows_ - 0.5) / rows_));
    num_burned_cells_ = 0;
    num_unburnable_ = this->GetNumUnburnableCells();
    virtual_particles_.reserve(100000);
    radiation_particles_.reserve(100000);
    ticking_cells_.reserve(100000);
    burning_cells_.reserve(100000);
    changed_cells_.reserve(100000);
    flooded_cells_.reserve(1000);

    noise_generated_ = false;
    last_has_noise_ = parameters_.has_noise_;
}

void GridMap::Reset(std::vector<std::vector<int>>* rasterData) {
    int cols = static_cast<int>(rasterData->size());
    int rows = (rasterData->empty()) ? 0 : static_cast<int>((*rasterData)[0].size());

    if (cols != cols_ || rows != rows_ || cells_.empty()) {
        cells_.assign(rows, std::vector<std::shared_ptr<FireCell>>(cols));
        explored_map_.assign(rows, std::vector<int>(cols, 0));
        step_explored_map_.assign(rows, std::vector<int>(cols, 0));
        fire_map_.assign(rows, std::vector<int>(cols, 0));
        visited_cells_.assign(rows, std::vector<bool>(cols, false));
        cols_ = cols;
        rows_ = rows;
    }

//#pragma omp parallel for collapse(2)
    for (int x = 0; x < rows_; ++x) {
        for (int y = 0; y < cols_; ++y) {
            if (cells_[x][y]) {
                cells_[x][y]->Reset((*rasterData)[y][x]);
            } else {
                cells_[x][y] = std::make_shared<FireCell>(x, y, parameters_, (*rasterData)[y][x]);
            }
            explored_map_[x][y] = 0;
            step_explored_map_[x][y] = 0;
            fire_map_[x][y] = 0;
            visited_cells_[x][y] = false;
        }
    }

    num_cells_ = cols_ * rows_;
    parameters_.SetGridNxNy(rows_, cols_);
    y_off_ = 2 * (1 - ((cols_ - 0.5) / cols_));
    x_off_ = 2 * (1 - ((rows_ - 0.5) / rows_));
    num_burned_cells_ = 0;
    num_unburnable_ = this->GetNumUnburnableCells();

    virtual_particles_.clear();
    radiation_particles_.clear();
    ticking_cells_.clear();
    burning_cells_.clear();
    flooded_cells_.clear();
    changed_cells_.clear();
    reserved_positions_.clear();
    buffer_.fillBuffer();
}


template <typename ParticleType>
void GridMap::UpdateVirtualParticles(std::vector<ParticleType> &particles, std::vector<std::vector<bool>> &visited_cells) {
    size_t part = 0;
    while (part < particles.size()) {
        auto &particle = particles[part];

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
            particles[part] = std::move(particles.back());
            particles.pop_back();
            continue;
        }

        Point p = Point(i, j);
        visited_cells[p.x_][p.y_] = true;
        if (CellCanIgnite(p.x_, p.y_)) {
            ticking_cells_.insert(p);
        }

        ++part;
    }
}

void GridMap::UpdateParticles() {
    for (auto &row : visited_cells_) {
        std::fill(row.begin(), row.end(), false);
    }
    UpdateVirtualParticles(virtual_particles_, visited_cells_);
    UpdateVirtualParticles(radiation_particles_, visited_cells_);

    for (auto it = ticking_cells_.begin(); it != ticking_cells_.end(); ) {
        // If cell is not visited, remove it from ticking cells
        if (!visited_cells_[it->x_][it->y_]) {
            it = ticking_cells_.erase(it);
        } else {
            ++it;
        }
    }
}

GridMap::~GridMap(){
    cells_.clear();
    virtual_particles_.clear();
    radiation_particles_.clear();
    burning_cells_.clear();
    ticking_cells_.clear();
    flooded_cells_.clear();
    changed_cells_.clear();
    explored_map_.clear();
    step_explored_map_.clear();
    fire_map_.clear();
    visited_cells_.clear();
}

void GridMap::pruneReservations() {
    // Keep only reservations that are still burning
    std::unordered_set<int64_t> still_burning;
    still_burning.reserve(burning_cells_.size());
    for (const auto& c : burning_cells_) still_burning.insert(idx(c.x_, c.y_));

    for (auto it = reserved_positions_.begin(); it != reserved_positions_.end(); ) {
        if (still_burning.find(*it) == still_burning.end()) it = reserved_positions_.erase(it);
        else ++it;
    }
}

void GridMap::RemoveReservation(std::pair<int, int> cell) {
    const int64_t id = idx(cell.first, cell.second);
    reserved_positions_.erase(id);
}

std::pair<double, double> GridMap::GetNextFire(std::pair<int, int> drone_position) {
    pruneReservations();
    auto possible_fires = parameters_.eval_fly_policy_ ? this->GetRawFirePositionsFromFireMap() : burning_cells_;

    if (possible_fires.empty()) {
        auto st = this->GetGroundstation()->GetGridPositionDouble();
        return {st.first, st.second};
    }

    double min_distance = std::numeric_limits<double>::max();
    int best_x = -1, best_y = -1;

    for (auto cell : possible_fires) {
        const int64_t id = idx(cell.x_, cell.y_);
        if (reserved_positions_.count(id)) continue;

        double distance = sqrt(
                pow(cell.x_ - drone_position.first, 2) +
                pow(cell.y_ - drone_position.second, 2)
        );
        if (distance < min_distance) { min_distance = distance; best_x = cell.x_; best_y = cell.y_; }
    }

    if (best_x != -1) {
        reserved_positions_.insert(idx(best_x, best_y));
        return {best_x + 0.5, best_y + 0.5};
    }

    // all fires are reserved
    auto st = this->GetGroundstation()->GetGridPositionDouble(); //this->GetRandomPointInGrid();
    return {st.first, st.second};
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
        auto neighbors = GetMooreNeighborhood(x, y);
        for (const auto& neighbor : neighbors) {
            changed_cells_.emplace_back(neighbor.first, neighbor.second);
        }
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
            auto neighbors = GetMooreNeighborhood(x, y);
            for (const auto& neighbor : neighbors) {
                changed_cells_.emplace_back(neighbor.first, neighbor.second);
            }
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
        changed_cells_.emplace_back(x, y);
        auto neighbors = GetMooreNeighborhood(x, y);
        for (const auto& neighbor : neighbors) {
            changed_cells_.emplace_back(neighbor.first, neighbor.second);
        }
        
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
        changed_cells_.emplace_back(x, y);
        auto neighbors = GetMooreNeighborhood(x, y);
        for (const auto& neighbor : neighbors) {
            changed_cells_.emplace_back(neighbor.first, neighbor.second);
        }
        // There was no fire in the cell so flood the cell and return false
        return false;
    }
}

void GridMap::EraseParticles(int x, int y) {
    auto should_erase = [&](const auto& particle) {
        double px, py;
        particle.GetPosition(px, py);
        int i, j;
        parameters_.ConvertRealToGridCoordinates(px, py, i, j);
        return i == x && j == y;
    };

    auto erase_swap_pop = [&](auto& particles) {
        for (size_t i = 0; i < particles.size(); ) {
            if (should_erase(particles[i])) {
                std::swap(particles[i], particles.back());
                particles.pop_back();
            } else {
                ++i;
            }
        }
    };

    erase_swap_pop(virtual_particles_);
    erase_swap_pop(radiation_particles_);
    ticking_cells_.erase(Point(x, y));
}

void GridMap::ExtinguishCell(int x, int y) {
    cells_[x][y]->Extinguish();
    burning_cells_.erase(Point(x, y));

    EraseParticles(x, y);

    changed_cells_.emplace_back(x, y);
    auto neighbors = GetMooreNeighborhood(x, y);
    for (const auto& neighbor : neighbors) {
        changed_cells_.emplace_back(neighbor.first, neighbor.second);
    }
}

std::shared_ptr<const std::vector<std::vector<std::vector<int>>>> GridMap::GetDroneView(std::pair<int, int> drone_position, int drone_view_radius) {
    int size = drone_view_radius + 1;

    // Initialisiere eine 3D-Matrix: [status_type][x][y]
    std::vector<std::vector<std::vector<int>>> view(2, std::vector<std::vector<int>>(size, std::vector<int>(size, 0)));

    int drone_view_radius_2 = drone_view_radius / 2;
    for (int x = drone_position.first - drone_view_radius_2; x <= drone_position.first + drone_view_radius_2; ++x) {
        for (int y = drone_position.second - drone_view_radius_2; y <= drone_position.second + drone_view_radius_2; ++y) {
            int new_x = x - drone_position.first + drone_view_radius_2;
            int new_y = y - drone_position.second + drone_view_radius_2;
            if (IsPointInGrid(x, y)) {
                // Setze Zellstatus (Status 0)
                view[0][new_x][new_y] = cells_[x][y]->GetCellState();

                // Setze Feuerstatus (Status 1)
                if (cells_[x][y]->IsBurning()){
                    view[1][new_x][new_y] = 1;
                } else {
                    view[1][new_x][new_y] = 0; // oder ein anderer Wert, der "kein Feuer" darstellt
                }
            }
        }
    }

    return std::make_shared<const std::vector<std::vector<std::vector<int>>>>(view);
}

//int GridMap::UpdateExplorationMap(int x, int y) {
////    int difference = parameters_.GetExplorationTime() - explored_map_[x][y];
////    explored_map_[x][y] = parameters_.GetExplorationTime();
//    // If this part of the map was previously not seen by the drone, return 1
//    auto newly_visited = explored_map_[x][y] > 0 ? 0 : 1;
//    explored_map_[x][y] = 1; //parameters_.GetExplorationTime();
//    return newly_visited;
//}

[[maybe_unused]] void GridMap::UpdateCellDiminishing() {
    for (int x = 0; x < rows_; ++x) {
        for (int y = 0; y < cols_; ++y) {
            if (explored_map_[x][y] > 0) {
                explored_map_[x][y]--;
            }
        }
    }
}

int GridMap::UpdateExploredAreaFromDrone(std::pair<int, int> drone_position, int drone_view_radius) {
    int drone_view_radius_2 = drone_view_radius / 2;
    int newly_visited_cells = 0;

    int start_x = std::max(0, drone_position.first - drone_view_radius_2);
    int end_x = std::min(rows_ - 1, drone_position.first + drone_view_radius_2);
    int start_y = std::max(0, drone_position.second - drone_view_radius_2);
    int end_y = std::min(cols_ - 1, drone_position.second + drone_view_radius_2);

    for (int x = start_x; x <= end_x; ++x) {
        for (int y = start_y; y <= end_y; ++y) {
            fire_map_[x][y] = cells_[x][y]->IsBurning() ? 1 : 0;
            if (step_explored_map_[x][y] == 0) {
                newly_visited_cells++;
                step_explored_map_[x][y] = 1;
            }
        }
    }
    return newly_visited_cells;
}

// Calculates the number of unburnable cells
int GridMap::GetNumUnburnableCells() const {
    int num_unburnable_cells = 0;
    for (int x = 0; x < rows_; ++x) {
        for (int y = 0; y < cols_; ++y) {
            if (!cells_[x][y]->CanIgnite()) {
                num_unburnable_cells++;
            }
        }
    }
    return num_unburnable_cells;
}

[[maybe_unused]] double GridMap::PercentageUnburnable() const {
    return (double)num_unburnable_ / (double)num_cells_;
}

// Calculates the percentage of burned cells
double GridMap::PercentageBurned() const {
    return (double)num_burned_cells_ / (double)num_cells_;
}

//Returns whether there are still burning fires on the map or particles!
bool GridMap::IsBurning() const {
    return !(burning_cells_.empty() && virtual_particles_.empty() && radiation_particles_.empty());
}

bool GridMap::HasBurningFires() const {
    return !burning_cells_.empty();
}

// Calculates the percentage of burning cells
[[maybe_unused]] double GridMap::PercentageBurning() const {
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
    if (!noise_generated_ || parameters_.has_noise_ != last_has_noise_) {
        if (parameters_.has_noise_) {
            int rows = static_cast<int>(cells_.size());
//#pragma omp parallel for
            for(int i = 0; i < rows; ++i) {
                const auto& cell_row = cells_[i];
//        for(const auto& cell_row : cells_) {
                for(const auto& cell : cell_row) {
                    if(cell->HasNoise()) {
                        cell->GenerateNoiseMap();
                    }
                }
            }
            this->noise_generated_ = true;
        } else {
            this->noise_generated_ = false;
        }
        last_has_noise_ = parameters_.has_noise_;
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

std::vector<std::vector<int>> GridMap::GetTotalDroneView(std::pair<int, int> drone_position, int view_radius) const {
    std::vector<std::vector<int>> view = std::vector<std::vector<int>>(rows_, std::vector<int>(cols_, 0));

    int drone_view_radius_2 = view_radius / 2;
    for (int x = drone_position.first - drone_view_radius_2; x <= drone_position.first + drone_view_radius_2; ++x) {
        for (int y = drone_position.second - drone_view_radius_2;
             y <= drone_position.second + drone_view_radius_2; ++y) {
            if (IsPointInGrid(x, y)) {
                view[x][y] = 255;
            }
        }
    }
    return view;
}

std::shared_ptr<const std::vector<std::vector<double>>> GridMap::GetInterpolatedDroneView(std::pair<int, int> drone_position, int view_radius, int size, bool interpolated) {
    auto total_drone_view = this->GetTotalDroneView(drone_position, view_radius);
    if(size == 0) {
        size = parameters_.GetExplorationMapSize();
    }
    if (!interpolated) {
        std::vector<std::vector<double>> doubleMatrix(total_drone_view.size(), std::vector<double>(total_drone_view[0].size()));
        for (size_t i = 0; i < total_drone_view.size(); ++i) {
            for (size_t j = 0; j < total_drone_view[0].size(); ++j) {
                doubleMatrix[i][j] = static_cast<double>(total_drone_view[i][j]);
            }
        }
        return std::make_shared<const std::vector<std::vector<double>>>(doubleMatrix);
    }
    return std::make_shared<const std::vector<std::vector<double>>>(BilinearInterpolation(total_drone_view, size, size));
}

std::shared_ptr<const std::vector<std::vector<int>>> GridMap::GetExploredMap(int size, bool interpolated) {
    if(size == 0) {
        size = parameters_.GetExplorationMapSize();
    }
    if (!interpolated) {
        return std::make_shared<const std::vector<std::vector<int>>>(explored_map_);
    }
    return std::make_shared<const std::vector<std::vector<int>>>(InterpolationResize(explored_map_, size, size));
}

std::shared_ptr<const std::vector<std::vector<int>>> GridMap::GetStepExploredMap(int size, bool interpolated) {
    if(size == 0) {
        size = parameters_.GetExplorationMapSize();
    }
    if (!interpolated) {
        return std::make_shared<const std::vector<std::vector<int>>>(step_explored_map_);
    }
    return std::make_shared<const std::vector<std::vector<int>>>(InterpolationResize(step_explored_map_, size, size));
}

std::shared_ptr<const std::vector<std::vector<double>>> GridMap::GetFireMap(int size, bool interpolated) {
    if(size == 0) {
        size = parameters_.GetFireMapSize();
    }
    if (!interpolated) {
        std::vector<std::vector<double>> doubleMatrix(fire_map_.size(), std::vector<double>(fire_map_[0].size()));
        for (size_t i = 0; i < fire_map_.size(); ++i) {
            for (size_t j = 0; j < fire_map_[0].size(); ++j) {
                doubleMatrix[i][j] = static_cast<double>(fire_map_[i][j]);
            }
        }
        return std::make_shared<const std::vector<std::vector<double>>>(doubleMatrix);
    }
    return std::make_shared<const std::vector<std::vector<double>>>(BilinearInterpolation(fire_map_, size, size));
}

std::pair<int, int> GridMap::GetRandomCorner() {
    // Returns a random corner of the grid with offset of 1 cell
    std::uniform_int_distribution<> dis_x(0, 1);
    std::uniform_int_distribution<> dis_y(0, 1);
    int x = dis_x(parameters_.gen_);
    int y = dis_y(parameters_.gen_);
    return std::make_pair(((1 - x) * (rows_ - 2)) + x, ((1 - y) * (cols_ - 2)) + y);
}

void GridMap::SetGroundstation() {
    auto corner = this->GetRandomCorner();
    groundstation_ = std::make_shared<Groundstation>(corner, parameters_);
}

[[maybe_unused]] int GridMap::GetNumExploredFires() const {
    // Returns the number of fires that have been seen by the drone
    int num_fires = 0;
    for (auto& row : fire_map_) {
        num_fires += static_cast<int>(std::count(row.begin(), row.end(), 1));
    }
    return num_fires;
}

[[maybe_unused]] bool GridMap::ExploredFiresEqualsActualFires() const {
    // Returns true if the fires seen by the drone are the same as the actual fires. This is needed because the explored
    // map may contain fires that are burned out

    bool explored_fires_equal_actual_fires = true;
    for (auto& fire : burning_cells_) {
        if (fire_map_[fire.x_][fire.y_] == 0) {
            explored_fires_equal_actual_fires = false;
            break;
        }
    }

    return explored_fires_equal_actual_fires;
}

int GridMap::GetRevisitedCells() {
    int revisited_cells = 0;
    for (int x = 0; x < rows_; ++x) {
        for (int y = 0; y < cols_; ++y) {
            if (step_explored_map_[x][y] > 0) {
                // TODO maybe change the logic here? seems kinda random
                // If the cell was not visited in the last 300 steps, count as newly explored
                if (explored_map_[x][y] <= parameters_.GetExplorationTime() - 300) {
                    explored_map_[x][y] = parameters_.GetExplorationTime(); //1;
                } else {
                    revisited_cells++;
                }
            }
        }
    }
    this->ResetStepExploreMap();
    return revisited_cells;
}

std::shared_ptr<const std::vector<std::pair<int, int>>> GridMap::GetExploredFires() {
    std::vector<std::pair<int, int>> explored_fires;
    for (int x = 0; x < rows_; ++x) {
        for (int y = 0; y < cols_; ++y) {
            if (fire_map_[x][y] == 1) {
                explored_fires.emplace_back(x, y);
            }
        }
    }
    return std::make_shared<const std::vector<std::pair<int, int>>>(explored_fires);
}

std::shared_ptr<std::vector<std::pair<double, double>>> GridMap::GetFirePositionsFromBurningCells() {
    std::shared_ptr<std::vector<std::pair<double, double>>> fire_positions = std::make_shared<std::vector<std::pair<double, double>>>();
    auto groundstation_position = std::make_pair(-1.0, -1.0);//groundstation_->GetGridPositionDouble();
    fire_positions->emplace_back(groundstation_position); // Add a dummy value to indicate no fire
    for (const auto& cell : burning_cells_) {
        fire_positions->emplace_back(cell.x_ + 0.5, cell.y_ + 0.5);
    }
    return fire_positions;
}

std::shared_ptr<std::vector<std::pair<double, double>>> GridMap::GetFirePositionsFromFireMap() const {
    std::vector<std::pair<double, double>> fire_positions;
    // Append Wait Token for the Network at (-1, -1)
    fire_positions.emplace_back(-1.0, -1.0);
    for (size_t i = 0; i < fire_map_.size(); ++i) {
        for (size_t j = 0; j < fire_map_[i].size(); ++j) {
            if (fire_map_[i][j] > 0) {
                fire_positions.emplace_back(static_cast<double>(i) + 0.5, static_cast<double>(j) + 0.5);
            }
        }
    }
    return std::make_shared<std::vector<std::pair<double, double>>>(fire_positions);
}

std::unordered_set<Point> GridMap::GetRawFirePositionsFromFireMap() const {
    std::unordered_set<Point> fire_positions;
    for (int x = 0; x < rows_; ++x) {
        for (int y = 0; y < cols_; ++y) {
            if (fire_map_[x][y] == 1) {
                fire_positions.insert(Point(x, y));
            }
        }
    }
    return fire_positions;
}
