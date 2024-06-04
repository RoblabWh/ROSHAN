//
// Created by nex on 08.06.23.
//

#ifndef ROSHAN_FIREMODEL_GRIDMAP_H
#define ROSHAN_FIREMODEL_GRIDMAP_H

#include <vector>
#include <unordered_set>
#include "point.h"
#include "point_hash.h"
#include "firemodel_firecell.h"
#include "model_parameters.h"
#include "wind.h"
#include <random>
#include <iostream>
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
    bool IsBurning() const;
    int GetNumParticles() { return virtual_particles_.size() + radiation_particles_.size();}
    bool CellCanIgnite(int x, int y) const { return cells_[x][y]->CanIgnite(); }
    void ShowCellInfo(int x, int y) { cells_[x][y]->ShowInfo(this->GetRows(), this->GetCols()); }
    int GetNumCells() const { return rows_ * cols_; }
    std::vector<VirtualParticle> GetVirtualParticles() const { return virtual_particles_; }
    std::vector<RadiationParticle> GetRadiationParticles() const { return radiation_particles_; }
    std::vector<Point> GetChangedCells() const { return changed_cells_; }
    void ResetChangedCells() { changed_cells_.clear(); }
    std::pair<std::vector<std::vector<int>>, std::vector<std::vector<int>>> GetDroneView(std::shared_ptr<DroneAgent> drone);
    std::vector<std::vector<int>> GetUpdatedMap(std::shared_ptr<DroneAgent> drone, std::vector<std::vector<int>> fire_status);

    std::pair<int, int> GetRandomPointInGrid() {
        std::uniform_int_distribution<> dis_x(0, cols_ - 1);
        std::uniform_int_distribution<> dis_y(0, rows_ - 1);
        return std::make_pair(dis_x(gen_), dis_y(gen_));
    }

    bool IsPointInGrid(int i, int j) const {
        return !(i < 0 || i >= cols_ || j < 0 || j >= rows_);
    }

    FireCell& At(int i, int j) {
        return *cells_[i][j];
    }

private:
    FireModelParameters &parameters_;
    std::shared_ptr<Wind> wind_;
    int rows_; // Number of rows in the grid
    int cols_; // Number of columns in the grid
    std::vector<std::vector<std::shared_ptr<FireCell>>> cells_;
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

};


#endif //ROSHAN_FIREMODEL_GRIDMAP_H
