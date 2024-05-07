//
// Created by nex on 02.07.23.
//

#include "firemodel_pixelbuffer.h"

void PixelBuffer::Draw(const SDL_Rect rect, Uint32 color) {
    int x_start = rect.x < 0 ? 0 : rect.x;
    int y_start = rect.y < 0 ? 0 : rect.y;
    int x_end = rect.x + rect.w > width_ ? width_ : rect.x + rect.w;
    int y_end = rect.y + rect.h > height_ ? height_ : rect.y + rect.h;

    for (int y = y_start; y < y_end; ++y) {
        for (int x = x_start; x < x_end; ++x) {
            pixels_[y * width_ + x] = color;
        }
    }
}

void PixelBuffer::DrawGrid(const SDL_Rect rect, Uint32 color) {
    int x_start = rect.x < 0 ? 0 : rect.x;
    int y_start = rect.y < 0 ? 0 : rect.y;
    int x_end = rect.x + rect.w > width_ ? width_ : rect.x + rect.w;
    int y_end = rect.y + rect.h > height_ ? height_ : rect.y + rect.h;

    for (int y = y_start; y < y_end - 1; ++y) {
        for (int x = x_start; x < x_end - 1; ++x) {
            pixels_[y * width_ + x] = color;
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
