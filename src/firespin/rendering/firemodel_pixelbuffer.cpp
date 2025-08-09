//
// Created by nex on 02.07.23.
//

#include "firemodel_pixelbuffer.h"


void PixelBuffer::Draw(const SDL_Rect rect, Uint32 base_color, const std::vector<std::vector<int>>& noise_map, int grid_offset) {
    int x_start = rect.x < 0 ? 0 : rect.x;
    int y_start = rect.y < 0 ? 0 : rect.y;
    int x_end = rect.x + rect.w > width_ ? width_ : rect.x + rect.w;
    int y_end = rect.y + rect.h > height_ ? height_ : rect.y + rect.h;

    if ((x_end - x_start) == 0 || (y_end - y_start) == 0) {
        return;
    }

    Uint8 r, g, b, a;
    SDL_GetRGBA(base_color, format_, &r, &g, &b, &a);

    int noise_size = static_cast<int>(noise_map.size());

    for (int y = y_start; y < y_end + grid_offset; ++y) {
        for (int x = x_start; x < x_end + grid_offset; ++x) {
            int noise_x = (x - x_start) * noise_size / (x_end - x_start);
            int noise_y = (y - y_start) * noise_size / (y_end - y_start);
            int noise = noise_map[noise_y][noise_x];
            Uint8 new_r = std::clamp(static_cast<int>(r) + noise, 0, 255);
            Uint8 new_g = std::clamp(static_cast<int>(g) + noise, 0, 255);
            Uint8 new_b = std::clamp(static_cast<int>(b) + noise, 0, 255);
            Uint32 noise_color = SDL_MapRGBA(format_, new_r, new_g, new_b, a);
            pixels_[y * width_ + x] = noise_color;
        }
    }
}

void PixelBuffer::Draw(const SDL_Rect rect, Uint32 base_color, int grid_offset) {
    int x_start = rect.x < 0 ? 0 : rect.x;
    int y_start = rect.y < 0 ? 0 : rect.y;
    int x_end = rect.x + rect.w > width_ ? width_ : rect.x + rect.w;
    int y_end = rect.y + rect.h > height_ ? height_ : rect.y + rect.h;

    for (int y = y_start; y < y_end + grid_offset; ++y) {
        for (int x = x_start; x < x_end + grid_offset; ++x) {
            pixels_[y * width_ + x] = base_color;
        }
    }
}

PixelBuffer::PixelBuffer(int width, int height, SDL_Color background_color, SDL_PixelFormat* format) : width_(width), height_(height),
                                                  pixels_(width_ * height_), format_(format) {
    background_color_ = SDLColorToUint32(background_color);
    Reset();
}

Uint32 PixelBuffer::SDLColorToUint32(const SDL_Color color) {
    return SDL_MapRGBA(format_, color.r, color.g, color.b, color.a);
}

void PixelBuffer::Reset() {
    std::fill(pixels_.begin(), pixels_.end(), background_color_);
}

void PixelBuffer::Resize(int width, int height) {
    width_ = width;
    height_ = height;
    pixels_.resize(width_ * height_);
    Reset();
}
