//
// Created by nex on 28.11.24.
//

#include "groundstation.h"

Groundstation::Groundstation(std::pair<int, int> point, FireModelParameters &parameters) : parameters_(parameters) {
    double x = (point.first + 0.5); // position in grid + offset to center
    double y = (point.second + 0.5); // position in grid + offset to center
    position_ = std::make_pair(x * parameters_.GetCellSize(), y * parameters_.GetCellSize());
}

void Groundstation::Render(std::pair<int, int> position, int size) {
    renderer_.RenderGroundStation(position, size);
}

std::pair<double, double> Groundstation::GetGridPositionDouble() {
    double x, y;
    x = position_.first / parameters_.GetCellSize();
    y = position_.second / parameters_.GetCellSize();
    return std::make_pair(x, y);
}

std::pair<int, int> Groundstation::GetGridPosition() {
    int x, y;
    parameters_.ConvertRealToGridCoordinates(position_.first, position_.second, x, y);
    return std::make_pair(x, y);
}
