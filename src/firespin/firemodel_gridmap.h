//
// Created by nex on 08.06.23.
//

#ifndef ROSHAN_FIREMODEL_GRIDMAP_H
#define ROSHAN_FIREMODEL_GRIDMAP_H

#include <utility>
#include <vector>
#include <unordered_set>
#include "utils.h"
#include "src/utils.h"
#include "firemodel_firecell.h"
#include "model_parameters.h"
#include "wind.h"
#include <random>
#include <iostream>
#include <mutex>
#include <omp.h>
#include <memory>
#include "src/reinforcementlearning/groundstation.h"

class GridMap {
public:
    GridMap(std::shared_ptr<Wind> wind, FireModelParameters &parameters, std::vector<std::vector<int>>* rasterData = nullptr);
    ~GridMap();

    int GetRows() const { return rows_; } // (x)
    int GetCols() const { return cols_; } // (y)
    void Reset(std::vector<std::vector<int>> *rasterData);
    void IgniteCell(int x, int y);
    bool WaterDispension(int x, int y);
    void ExtinguishCell(int x, int y);
    CellState GetCellState(int x, int y) { return cells_[x][y]->GetIgnitionState(); }

    void UpdateParticles();
    void UpdateCells();
    double PercentageBurned() const;
    std::shared_ptr<const std::vector<std::vector<double>>> GetInterpolatedDroneView(std::pair<int, int> drone_position, int view_radius, int size=0, bool interpolated=true);
    std::shared_ptr<const std::vector<std::vector<int>>> GetExploredMap(int size=0, bool interpolated=true);
    std::shared_ptr<const std::vector<std::vector<int>>> GetStepExploredMap(int size=0, bool interpolated=true);
    std::shared_ptr<const std::vector<std::vector<double>>> GetFireMap(int size=0, bool interpolated=true);
    std::shared_ptr<const std::vector<std::pair<int, int>>> GetExploredFires();
    std::shared_ptr<std::vector<std::pair<double, double>>> GetFirePositionsFromBurningCells();

    [[maybe_unused]] int GetNumExploredFires() const;

    [[maybe_unused]] bool ExploredFiresEqualsActualFires() const;

    [[maybe_unused]] double PercentageBurning() const;

    [[maybe_unused]] double PercentageUnburnable() const;

    [[maybe_unused]] int GetNumOfCells() const { return num_cells_; }

    [[maybe_unused]] std::unordered_set<Point> GetBurningCells() const { return burning_cells_; }
    int GetNumBurningCells() const { return static_cast<int>(burning_cells_.size()); }

    [[maybe_unused]] int GetNumBurnedCells() const { return num_burned_cells_; }
    std::pair<double, double> GetNextFire(std::pair<int, int> drone_position);

    [[maybe_unused]] int GetNumUnburnable() const { return num_unburnable_; }
    bool CanStartFires(int num_fires) const;
    bool IsBurning() const;
    bool HasBurningFires() const;
    int GetNumParticles() { return static_cast<int>(virtual_particles_.size() + radiation_particles_.size());}
    bool CellCanIgnite(int x, int y) const { return cells_[x][y]->CanIgnite(); }
    void ShowCellInfo(int x, int y) { cells_[x][y]->ShowInfo(this->GetRows(), this->GetCols()); }
    int GetNumCells() const { return rows_ * cols_; }
    const std::vector<VirtualParticle>& GetVirtualParticles() const { return virtual_particles_; }
    const std::vector<RadiationParticle>& GetRadiationParticles() const { return radiation_particles_; }
    std::vector<Point> GetChangedCells() const { return changed_cells_; }
    void ResetChangedCells() { changed_cells_.clear(); }
    std::shared_ptr<const std::vector<std::vector<std::vector<int>>>> GetDroneView(std::pair<int, int> drone_position, int drone_view_radius);
    std::vector<std::vector<int>> GetTotalDroneView(std::pair<int, int> drone_position, int view_radius) const;
    void SetGroundstation();
    std::shared_ptr<Groundstation> GetGroundstation() { return groundstation_; }
    void SetGroundstationRenderer(SDL_Renderer* renderer) {groundstation_->SetRenderer(renderer);};

    void SetTerminals(bool terminal) {any_terminal_occured_ = terminal;}
    bool GetTerminalOccured() const {return any_terminal_occured_;}
    int GetNRefFires() const {return n_ref_fires_;}

    std::pair<int, int> GetRandomPointInGrid() {
        std::uniform_int_distribution<> dis_x(0, rows_ - 1);
        std::uniform_int_distribution<> dis_y(0, cols_ - 1);
        return std::make_pair(dis_x(parameters_.gen_), dis_y(parameters_.gen_));
    }

    std::pair<double, double> GetNonGroundStationCorner() {
        std::pair<double, double> corner = GetRandomCorner();
        while (static_cast<int>(corner.first) == groundstation_->GetGridPosition().first &&
               static_cast<int>(corner.second) == groundstation_->GetGridPosition().second) {
            corner = GetRandomCorner();
        }
        corner.first += 0.5;
        corner.second += 0.5;
        return corner;
    }

    bool IsPointInGrid(int x, int y) const {
        return !(x < 0 || x >= rows_ || y < 0 || y >= cols_);
    }

    FireCell& At(int x, int y) {
        return *cells_[x][y];
    }

    bool IsExplored(int x, int y) const {
        return explored_map_[x][y] > 0;
    };

    std::vector<std::pair<int, int>> GetMooreNeighborhood(int x, int y) const;
    int UpdateExploredAreaFromDrone(std::pair<int, int> drone_position, int drone_view_radius);
    int GetRevisitedCells();
    void ResetStepExploreMap() {
        for (int r=0; r<rows_; r++) {
            for (int c=0; c<cols_; c++) {
                step_explored_map_[r][c] = 0;
            }
        }
    }
    void ResetExploredMap() {
        for (int r=0; r<rows_; r++) {
            for (int c=0; c<cols_; c++) {
                explored_map_[r][c] = 0;
            }
        }
    }
    [[maybe_unused]] void UpdateCellDiminishing();
    std::pair<int, int> GetRandomCorner();
    std::pair<double, double> GetPointInGridFromNormalizedCoordinates(double x, double y) const {
        int grid_x = static_cast<int>((x + 1) / 2 * rows_);
        int grid_y = static_cast<int>((y + 1) / 2 * cols_);
        return std::make_pair(grid_x, grid_y);
    }
    std::unordered_set<Point> GetRawFirePositionsFromFireMap() const;
    std::shared_ptr<std::vector<std::pair<double, double>>> GetFirePositionsFromFireMap() const;

    // For Rendering Only
    void GenerateNoiseMap();
    static void SetCellNoise(CellState state, int noise_level, int noise_size);
    void SetNoiseGenerated(bool noise_generated) { noise_generated_ = noise_generated; }
    bool HasNoiseGenerated() const { return noise_generated_; }
    void RemoveReservation(std::pair<int, int> cell);

    [[maybe_unused]] double GetXOff() const { return x_off_; }
    [[maybe_unused]] double GetYOff() const { return y_off_; }

private:
    FireModelParameters &parameters_;
    std::shared_ptr<Wind> wind_;
    int cols_; // Number of columns in the grid (y)
    int rows_; // Number of rows in the grid (x)
    std::vector<std::vector<std::shared_ptr<FireCell>>> cells_;
    std::vector<std::vector<int>> explored_map_;
    std::vector<std::vector<int>> step_explored_map_;
    std::vector<std::vector<int>> fire_map_;
    std::vector<std::vector<bool>> visited_cells_;
    std::unordered_set<Point> ticking_cells_;
    std::unordered_set<Point> burning_cells_;
    std::unordered_set<Point> flooded_cells_;
    std::vector<Point> changed_cells_;
    std::shared_ptr<Groundstation> groundstation_;

    std::vector<VirtualParticle> virtual_particles_;
    std::vector<RadiationParticle> radiation_particles_;
    template <typename ParticleType>
    void UpdateVirtualParticles(std::vector<ParticleType> &particles, std::vector<std::vector<bool>> &visited_cells);

    void EraseParticles(int x, int y);
    void pruneReservations();
    inline int64_t idx(int x, int y) const { return int64_t(y) * cols_ + x;}
    std::unordered_set<int64_t> reserved_positions_;

    //Bad flags for reward calculation
    bool any_terminal_occured_ = false;

    //For calculation of percentage burned
    int num_cells_ = 0;
    int num_burned_cells_ = 0;
    int num_unburnable_ = 0;
    int n_ref_fires_ = 0;

    //Noise handling
    bool noise_generated_;
    bool last_has_noise_;

    //Optimization
    RandomBuffer buffer_;
    double x_off_;
    double y_off_;

    int GetNumUnburnableCells() const;

//    int UpdateExplorationMap(int i, int j);

};


#endif //ROSHAN_FIREMODEL_GRIDMAP_H
