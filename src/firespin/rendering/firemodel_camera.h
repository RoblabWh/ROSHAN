//
// Created by nex on 27.06.23.
//

#ifndef ROSHAN_FIREMODEL_CAMERA_H
#define ROSHAN_FIREMODEL_CAMERA_H

#include <utility>
#include <algorithm>
#include <cmath>

namespace {
    constexpr double kMinZoom = 0.1;
    constexpr double kMaxZoom = 10.0;
    constexpr double kSmoothFactor = 0.15;
    constexpr double kSnapThreshold = 0.001;
}

class FireModelCamera {
public:
    FireModelCamera()
        : x_(0), y_(0), zoom_(1.0), camera_speed_(0.3),
          target_x_(0), target_y_(0), target_zoom_(1.0) {};

    void Move(double dx, double dy);
    void Zoom(double factor);
    void ZoomToPoint(double factor, int mouseX, int mouseY, int rows, int cols);
    void ResetView();

    [[nodiscard]] double GetX() const { return x_; }
    [[nodiscard]] double GetY() const { return y_; }
    [[nodiscard]] double GetZoom() const { return zoom_; }
    [[nodiscard]] double GetTargetZoom() const { return target_zoom_; }
    [[nodiscard]] bool IsAnimating() const;

    void SetCellSize(int rows, int cols);
    [[nodiscard]] double GetCellSize() const { return cell_size_; }
    void SetOffset(int rows, int cols);
    void SetViewport(int screen_width, int screen_height);
    [[nodiscard]] double GetViewportWidth() const { return viewport_width_; }
    [[nodiscard]] double GetViewportHeight() const { return viewport_height_; }
    [[nodiscard]] double GetOffsetX() const { return offset_x_; }
    [[nodiscard]] double GetOffsetY() const { return offset_y_; }
    void Update(int width, int height, int rows, int cols);

    [[nodiscard]] std::pair<int, int> ScreenToGridPosition(int screenX, int screenY) const;
    [[nodiscard]] std::pair<int, int> GridToScreenPosition(double worldX, double worldY) const;

    void SetTarget(double x, double y) { target_x_ = x; target_y_ = y; }

private:
    // Current state (used for rendering, interpolated each frame)
    double x_;
    double y_;
    double zoom_;
    double camera_speed_;
    double cell_size_{};
    double offset_x_{};
    double offset_y_{};
    double viewport_width_{};
    double viewport_height_{};

    // Target state (set by input, camera lerps toward this)
    double target_x_;
    double target_y_;
    double target_zoom_;
};


#endif //ROSHAN_FIREMODEL_CAMERA_H
