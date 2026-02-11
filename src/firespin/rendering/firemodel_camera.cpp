//
// Created by nex on 27.06.23.
//

#include "firemodel_camera.h"

void FireModelCamera::Move(double dx, double dy) {
    double adjusted_speed = camera_speed_ / target_zoom_;
    target_x_ -= dy * adjusted_speed;
    target_y_ -= dx * adjusted_speed;
}

void FireModelCamera::Zoom(double factor) {
    if (factor <= 0.0) {
        return;
    }
    target_zoom_ = std::clamp(target_zoom_ * factor, kMinZoom, kMaxZoom);
}

void FireModelCamera::ZoomToPoint(double factor, int mouseX, int mouseY, int rows, int cols) {
    if (factor <= 0.0) return;

    double new_zoom = std::clamp(target_zoom_ * factor, kMinZoom, kMaxZoom);
    if (new_zoom == target_zoom_) return;

    // Compute grid position under cursor at current target state
    // We need to simulate what cell_size and offset would be at current target zoom
    double base_cell = double(std::min(viewport_width_, viewport_height_)) / (double)std::max(rows, cols);
    double old_cell = std::clamp(base_cell * target_zoom_, 1.0, 200.0);
    double old_off_x = (viewport_height_ - rows * old_cell) / 2.0;
    double old_off_y = (viewport_width_ - cols * old_cell) / 2.0;

    // Grid position under mouse (float precision)
    double gridX = (mouseY - old_off_x) / old_cell - target_x_;
    double gridY = (mouseX - old_off_y) / old_cell - target_y_;

    // Compute cell_size and offset at new zoom
    double new_cell = std::clamp(base_cell * new_zoom, 1.0, 200.0);
    double new_off_x = (viewport_height_ - rows * new_cell) / 2.0;
    double new_off_y = (viewport_width_ - cols * new_cell) / 2.0;

    // Adjust target position so grid point stays under cursor
    target_x_ = (mouseY - new_off_x) / new_cell - gridX;
    target_y_ = (mouseX - new_off_y) / new_cell - gridY;
    target_zoom_ = new_zoom;
}

void FireModelCamera::ResetView() {
    target_x_ = 0;
    target_y_ = 0;
    target_zoom_ = 1.0;
}

bool FireModelCamera::IsAnimating() const {
    return std::abs(target_x_ - x_) > kSnapThreshold ||
           std::abs(target_y_ - y_) > kSnapThreshold ||
           std::abs(target_zoom_ - zoom_) > kSnapThreshold;
}

std::pair<int, int> FireModelCamera::ScreenToGridPosition(int screenX, int screenY) const {
    // Use double precision throughout, floor only on final result
    double gridX = std::floor((screenY - offset_x_) / cell_size_ - x_);
    double gridY = std::floor((screenX - offset_y_) / cell_size_ - y_);

    return std::make_pair(static_cast<int>(gridX), static_cast<int>(gridY));
}

std::pair<int, int> FireModelCamera::GridToScreenPosition(double worldX, double worldY) const {
    // Use double precision throughout, round only on final result
    int screenY = static_cast<int>(std::round((worldX + x_) * cell_size_ + offset_x_));
    int screenX = static_cast<int>(std::round((worldY + y_) * cell_size_ + offset_y_));

    return std::make_pair(screenX, screenY);
}

void FireModelCamera::Update(int width, int height, int rows, int cols) {
    // Lerp current state toward target
    if (IsAnimating()) {
        x_ += (target_x_ - x_) * kSmoothFactor;
        y_ += (target_y_ - y_) * kSmoothFactor;
        zoom_ += (target_zoom_ - zoom_) * kSmoothFactor;

        // Snap to target when close enough
        if (std::abs(target_x_ - x_) <= kSnapThreshold) x_ = target_x_;
        if (std::abs(target_y_ - y_) <= kSnapThreshold) y_ = target_y_;
        if (std::abs(target_zoom_ - zoom_) <= kSnapThreshold) zoom_ = target_zoom_;
    }

    SetViewport(width, height);
    SetCellSize(rows, cols);
    SetOffset(rows, cols);
}

void FireModelCamera::SetCellSize(int rows, int cols) {
    double cell_size = double(std::min(GetViewportWidth(), GetViewportHeight())) / (double)std::max(rows, cols) * zoom_;
    cell_size_ = cell_size <= 1.0 ? 1.0 : cell_size >= 200.0 ? 200.0 : cell_size;
}

void FireModelCamera::SetOffset(int rows, int cols) {
    offset_x_ = (GetViewportHeight() - rows * cell_size_) / 2.0;
    offset_y_ = (GetViewportWidth() - cols * cell_size_) / 2.0;
}

void FireModelCamera::SetViewport(int screen_width, int screen_height) {
    viewport_width_ = screen_width;
    viewport_height_ = screen_height;
}
