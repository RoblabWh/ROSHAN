//
// UICallbacks.h - Grouped callback function types for ImGui UI
//

#ifndef ROSHAN_UICALLBACKS_H
#define ROSHAN_UICALLBACKS_H

#include <functional>
#include <vector>
#include <memory>
#include "externals/pybind11/include/pybind11/pybind11.h"
#include "firespin/firemodel_firecell.h"

namespace py = pybind11;

namespace ui {

// Forward declarations to avoid circular dependencies
// These will be replaced with actual includes in implementation files

// Callback group for RL-related operations
struct __attribute__((visibility("hidden"))) RLCallbacks {
    std::function<py::dict()> getRLStatus;
    std::function<void(py::dict)> setRLStatus;
    std::function<void()> resetDrones;
};

// Callback group for GridMap operations
struct GridMapCallbacks {
    std::function<void(std::vector<std::vector<int>>*, bool)> resetGridMap;
    std::function<void()> setUniformRasterData;
    std::function<void()> fillRasterWithEnum;
    std::function<void(CellState, int, int)> setNoise;
};

// Callback group for drone control
struct DroneCallbacks {
    std::function<void(int, double, double, int)> moveDrone;
};

// Callback group for fire operations
struct FireCallbacks {
    std::function<void()> startFires;
};

// Central callbacks container that aggregates all callback groups
struct __attribute__((visibility("hidden"))) UICallbacks {
    RLCallbacks rl;
    GridMapCallbacks gridMap;
    DroneCallbacks drone;
    FireCallbacks fire;

    // Convenience method to check if all required callbacks are set
    bool IsValid() const {
        return rl.getRLStatus && rl.setRLStatus && rl.resetDrones &&
               gridMap.resetGridMap && gridMap.setUniformRasterData &&
               gridMap.fillRasterWithEnum && gridMap.setNoise &&
               drone.moveDrone && fire.startFires;
    }
};

} // namespace ui

#endif // ROSHAN_UICALLBACKS_H
