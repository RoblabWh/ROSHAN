//
// Created by nex on 12.06.23.
//

#ifndef ROSHAN_UTILS_H
#define ROSHAN_UTILS_H

#include <string>
#include <filesystem>
#include <optional>

std::string formatTime(int seconds);

enum CellState { GENERIC_UNBURNED = 0,
                 SEALED = 1,
                 WOODY_NEEDLE_LEAVED_TREES = 2,
                 WOODY_BROADLEAVED_DECIDUOUS_TREES = 3,
                 WOODY_BROADLEAVED_EVERGREEN_TREES = 4,
                 LOW_GROWING_WOODY_PLANTS = 5,
                 PERMANENT_HERBACEOUS = 6,
                 PERIODICALLY_HERBACEOUS = 7,
                 LICHENS_AND_MOSSES = 8,
                 NON_AND_SPARSLEY_VEGETATED = 9,
                 WATER = 10,
                 SNOW_AND_ICE = 11,
                 GENERIC_BURNING = 12,
                 GENERIC_BURNED = 13,
                 GENERIC_FLOODED = 14,
                 OUTSIDE_AREA = 15,
                 OUTSIDE_GRID = 16,
                 CELL_STATE_COUNT};

std::string CellStateToString(CellState cell_state);
std::optional<std::filesystem::path> find_project_root(const std::filesystem::path& start);

#endif //ROSHAN_UTILS_H
