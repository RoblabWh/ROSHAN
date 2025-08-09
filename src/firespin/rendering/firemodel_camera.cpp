//
// Created by nex on 27.06.23.
//

#include "firemodel_camera.h"

void FireModelCamera::Move(double dx, double dy) {
    x_ -= dy * camera_speed_;
    y_ -= dx * camera_speed_;
}

void FireModelCamera::Zoom(double factor) {
    if (factor <= 0.0) {
        return;
    }
    zoom_ = std::clamp(zoom_ * factor, kMinZoom, kMaxZoom);
}

std::pair<int, int> FireModelCamera::ScreenToGridPosition(int screenX, int screenY) const {

    // Convert screen coordinates to grid coordinates
    int gridX = static_cast<int>(((screenY - offset_x_) / static_cast<int>(cell_size_)) - x_);
    int gridY = static_cast<int>(((screenX - offset_y_) / static_cast<int>(cell_size_)) - y_);

    return std::make_pair(gridX, gridY);
}

std::pair<int, int> FireModelCamera::GridToScreenPosition(double worldX, double worldY) const {

    int screenY = static_cast<int>(((worldX + x_) * static_cast<int>(cell_size_)) + offset_x_);
    int screenX = static_cast<int>(((worldY + y_) * static_cast<int>(cell_size_)) + offset_y_);

    return std::make_pair(screenX, screenY);
}

void FireModelCamera::Update(int width, int height, int rows, int cols) {
    SetViewport(width, height);
    SetCellSize(rows, cols);
    SetOffset(rows, cols);
}

void FireModelCamera::SetCellSize(int rows, int cols) {
    double cell_size = double(std::min(GetViewportWidth(), GetViewportHeight())) / (double)std::max(rows, cols) * zoom_;
    cell_size_ = cell_size <= 1.0 ? 1.0 : cell_size >= 200.0 ? 200.0 : cell_size;
}

void FireModelCamera::SetOffset(int rows, int cols) {
    offset_x_ = (GetViewportHeight() - rows * static_cast<int>(cell_size_)) / 2;
    offset_y_ = (GetViewportWidth() - cols * static_cast<int>(cell_size_)) / 2;
}

void FireModelCamera::SetViewport(int screen_width, int screen_height) {
    viewport_width_ = screen_width;
    viewport_height_ = screen_height;
}