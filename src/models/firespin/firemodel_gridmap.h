//
// Created by nex on 08.06.23.
//

#ifndef ROSHAN_FIREMODEL_GRIDMAP_H
#define ROSHAN_FIREMODEL_GRIDMAP_H

#include <vector>
#include <unordered_set>
#include "utils.h"
#include "point.h"
#include "point_hash.h"
#include "firemodel_firecell.h"
#include "model_parameters.h"
#include "wind.h"
#include <random>
#include <iostream>
#include <mutex>
#include <omp.h>
#include <memory>
#include "src/reinforcementlearning/drone_agent/drone.h"

// TODO Remove circular dependency
class DroneAgent;

class GridMap {
public:
    GridMap(std::shared_ptr<Wind> wind, FireModelParameters &parameters, std::vector<std::vector<int>>* rasterData = nullptr);
    ~GridMap();

    int GetRows() const { return rows_; }
    int GetCols() const { return cols_; }
    FireModelParameters &GetParameters() { return parameters_; }
    void IgniteCell(int x, int y);
    bool WaterDispension(int x, int y);
    void ExtinguishCell(int x, int y);
    CellState GetCellState(int x, int y) { return cells_[x][y]->GetIgnitionState(); }

    void UpdateParticles();
    void UpdateCells();
    double PercentageBurned() const;
    std::vector<std::vector<int>> GetExploredMap(int size=0);
    std::vector<std::vector<int>> GetFireMap(int size=0);

    double PercentageBurning() const;
    double PercentageUnburnable() const;
    int GetNumOfCells() const { return num_cells_; }
    int GetNumBurningCells() const { return burning_cells_.size(); }
    int GetNumBurnedCells() const { return num_burned_cells_; }
    int GetNumUnburnable() const { return num_unburnable_; }
    bool CanStartFires(int num_fires) const;
    bool IsBurning() const;
    int GetNumParticles() { return virtual_particles_.size() + radiation_particles_.size();}
    bool CellCanIgnite(int x, int y) const { return cells_[x][y]->CanIgnite(); }
    void ShowCellInfo(int x, int y) { cells_[x][y]->ShowInfo(this->GetRows(), this->GetCols()); }
    int GetNumCells() const { return rows_ * cols_; }
    const std::vector<VirtualParticle>& GetVirtualParticles() const { return virtual_particles_; }
    const std::vector<RadiationParticle>& GetRadiationParticles() const { return radiation_particles_; }
    std::vector<Point> GetChangedCells() const { return changed_cells_; }
    void ResetChangedCells() { changed_cells_.clear(); }
    std::vector<std::vector<std::vector<int>>> GetDroneView(std::shared_ptr<DroneAgent> drone);
    std::vector<std::vector<int>> GetUpdatedMap(std::shared_ptr<DroneAgent> drone, std::vector<std::vector<int>> fire_status);

    std::pair<int, int> GetRandomPointInGrid() {
        std::uniform_int_distribution<> dis_x(0, cols_ - 1);
        std::uniform_int_distribution<> dis_y(0, rows_ - 1);
        return std::make_pair(dis_x(gen_), dis_y(gen_));
    }

    bool IsPointInGrid(int x, int y) const {
        return !(x < 0 || x >= cols_ || y < 0 || y >= rows_);
    }

    FireCell& At(int x, int y) {
        return *cells_[x][y];
    }

    std::vector<std::pair<int, int>> GetMooreNeighborhood(int x, int y) const;
    void UpdateExploredAreaFromDrone(std::shared_ptr<DroneAgent> drone);
    void UpdateCellDiminishing();

    // For Rendering Only
    void GenerateNoiseMap();
    void SetCellNoise(CellState state, int noise_level, int noise_size);
    void SetNoiseGenerated(bool noise_generated) { noise_generated_ = noise_generated; }
    bool HasNoiseGenerated() const { return noise_generated_; }

private:
    FireModelParameters &parameters_;
    std::shared_ptr<Wind> wind_;
    int rows_; // Number of rows in the grid
    int cols_; // Number of columns in the grid
    std::vector<std::vector<std::shared_ptr<FireCell>>> cells_;
    std::vector<std::vector<int>> explored_map_;
    std::vector<std::vector<int>> fire_map_;
    std::unordered_set<Point> ticking_cells_;
    std::unordered_set<Point> burning_cells_;
    std::unordered_set<Point> flooded_cells_;
    std::vector<Point> changed_cells_;

    // Random decives and Generators for the Cells
    std::random_device rd_;
    std::mt19937 gen_;

    std::vector<VirtualParticle> virtual_particles_;
    std::vector<RadiationParticle> radiation_particles_;
    template <typename ParticleType>
    void UpdateVirtualParticles(std::vector<ParticleType> &particles, std::vector<std::vector<bool>> &visited_cells);

    void EraseParticles(int x, int y);

    //For calculation of percentage burned
    int num_cells_ = 0;
    int num_burned_cells_ = 0;
    int num_unburnable_ = 0;

    //Noise handling
    bool noise_generated_;

    //Optimization
    RandomBuffer buffer_;

    int GetNumUnburnableCells() const;

    int UpdateLastSeenTime(int i, int j);

};


#endif //ROSHAN_FIREMODEL_GRIDMAP_H
