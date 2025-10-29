//
// Created by nex on 06.09.25.
//

#ifndef ROSHAN_START_GOAL_STRATEGY_H
#define ROSHAN_START_GOAL_STRATEGY_H

#include "firespin/firemodel_gridmap.h"
#include "firespin/model_parameters.h"

enum class StartMode {
    MixedByPct,
    GroundStationOnly,
    RandomAnywhere,
    SameCellWithOffsets  // all agents same cell w/ offsets
};

struct SpawnConfig {
    StartMode mode = StartMode::MixedByPct;
    double groundstation_start_percentage_ = 0.5;
    // Same-cell options
    std::string group_key = "default"; // currently unused, but allows for fixed starting cell per group (key must be set at agent creation)
    int group_size = 1;   // total agents expected in this group
    double drone_radius_m = 0.5;
};

struct GoalConfig {
    double fire_goal_pct = 0.5; // when rl_mode != "eval"
};

struct StartAssignment {
    std::pair<double,double> pos_grid_double; // (x,y) in grid coords, doubles
    std::pair<int,int> cell_ij;               // integer cell for reference
};

class StartPlanner {
public:
    StartPlanner(FireModelParameters& params, std::shared_ptr<GridMap> grid_map)
            : parameters_(params), grid_map_(std::move(grid_map)) {}

    // Call once per agent
    StartAssignment assign_start(const SpawnConfig& sc, int idx_in_group) {
        // Choose base cell
        auto cell = choose_cell(sc);
        // Offset (maybe none)
        auto offset_m = compute_offset(sc, idx_in_group);
        // Convert to grid coords (double)
        double cell_size_m = parameters_.GetCellSize();
        auto center = cell_center_grid(cell);
        std::pair<double,double> pos = {
                center.first  + (offset_m.first  / cell_size_m),
                center.second + (offset_m.second / cell_size_m)
        };
        // Reserve (optional) to avoid inter-loop collisions
        reserve(cell, pos);

        return { pos, cell };
    }

private:
    FireModelParameters& parameters_;
    std::shared_ptr<GridMap> grid_map_;

    // Reservations per integer cell (to avoid overlapping across loops)
    std::unordered_map<long long, std::vector<std::pair<double,double>>> reserved_;

    static long long key(const std::pair<int,int>& c) {
        return (static_cast<long long>(c.first) << 32) ^ (c.second & 0xffffffff);
    }

    std::pair<int,int> choose_cell(const SpawnConfig& sc) {
        switch (sc.mode) {
            case StartMode::GroundStationOnly:
                return grid_map_->GetGroundstation()->GetGridPosition();
            case StartMode::RandomAnywhere:
                return grid_map_->GetRandomPointInGrid();
            case StartMode::SameCellWithOffsets: {
                // Fix a cell per group_key for this episode
                auto it = fixed_cell_.find(sc.group_key);
                if (it != fixed_cell_.end()) return it->second;
                auto c = grid_map_->GetGroundstation()->GetGridPosition(); // pick policy per agent
                fixed_cell_[sc.group_key] = c;
                return c;
            }
            case StartMode::MixedByPct:
            default: {
                std::uniform_real_distribution<> dist(0.0, 1.0);
                auto r = dist(parameters_.gen_);
                if (r <= sc.groundstation_start_percentage_) {
                    return grid_map_->GetGroundstation()->GetGridPosition();
                } else {
                    return grid_map_->GetRandomPointInGrid();
                }
            }
        }
    }

    static std::pair<double,double> cell_center_grid(const std::pair<int,int>& cell) {
        // center in grid coords (x,y) = (i+0.5, j+0.5)
        return { cell.first + 0.5, cell.second + 0.5 };
    }

    std::pair<double,double> compute_offset(const SpawnConfig& sc, int idx_in_group) {
        if (sc.group_size <= 1)
            return {0.0, 0.0};

        double cell_size_m = parameters_.GetCellSize();
        double r = sc.drone_radius_m;
        double margin = std::max(0.05 * cell_size_m, 0.1); // small edge margin

        // ring radius: keep >= 2r spacing and inside the cell
        double R = std::min(cell_size_m * 0.5 - margin, std::max(2.0 * r, cell_size_m * 0.35));

        // Evenly distribute around the ring
        double theta = (2.0 * M_PI) * (static_cast<double>(idx_in_group) / static_cast<double>(sc.group_size));
        double dx = R * std::cos(theta);
        double dy = R * std::sin(theta);

        return {dx, dy};
    }

    void reserve(const std::pair<int,int>& cell, const std::pair<double,double>& pos_grid) {
        reserved_[key(cell)].push_back(pos_grid);
    }

    std::unordered_map<std::string, std::pair<int,int>> fixed_cell_;
};

class GoalSelector {
public:
    explicit GoalSelector(std::shared_ptr<GridMap> gm) : grid_map_(std::move(gm)) {}
    std::pair<double,double> pick_goal(const GoalConfig& gc,
                                       const std::string& rl_mode,
                                       const bool is_explorer,
                                       const std::pair<double,double>& start_grid_double,
                                       std::mt19937& gen)
    {
        if (!is_explorer){
            if (rl_mode == "eval") {
                auto p = grid_map_->GetNextFire(start_grid_double);
                if (p != std::pair<double,double>{-1,-1}) return p;
                return grid_map_->GetGroundstation()->GetGridPositionDouble();
            }

            std::uniform_real_distribution<double> dist(0.0, 1.0);
            if (dist(gen) < gc.fire_goal_pct) {
                auto p = grid_map_->GetNextFire(start_grid_double);
                if (p != std::pair<double,double>{-1,-1}) return p;
            }
        }
        return grid_map_->GetNonGroundStationCorner();
    }

private:
    std::shared_ptr<GridMap> grid_map_;
};


#endif //ROSHAN_START_GOAL_STRATEGY_H
