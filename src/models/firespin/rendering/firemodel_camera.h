//
// Created by nex on 27.06.23.
//

#ifndef ROSHAN_FIREMODEL_CAMERA_H
#define ROSHAN_FIREMODEL_CAMERA_H

#include <utility>
#include <algorithm>
#include <math.h>

class FireModelCamera {
public:
    FireModelCamera() : x_(0), y_(0), zoom_(1.0), camera_speed_(0.3) {};
    void Move(double dx, double dy);
    void Zoom(double factor);
    double GetX() const { return x_; }
    double GetY() const { return y_; }
    double GetZoom() const { return zoom_; }
    void SetCellSize(int rows, int cols);
    double GetCellSize() const { return cell_size_; }
    void SetOffset(int rows, int cols);
    void SetViewport(int screen_width, int screen_height);
    double GetViewportWidth() const { return viewport_width_; }
    double GetViewportHeight() const { return viewport_height_; }
    double GetOffsetX() const { return offset_x_; }
    double GetOffsetY() const { return offset_y_; }
    void Update(int width, int height, int rows, int cols);

    std::pair<int, int> ScreenToGridPosition(int screenX, int screenY) const;
    std::pair<int, int> GridToScreenPosition(double worldX, double worldY) const;
private:
    double x_;
    double y_;
    double zoom_;
    double camera_speed_;
    double cell_size_;
    double offset_x_;
    double offset_y_;
    double viewport_width_;
    double viewport_height_;
};


#endif //ROSHAN_FIREMODEL_CAMERA_H
