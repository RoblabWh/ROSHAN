//
// Created by nex on 02.07.23.
//

#include "firemodel_pixelbuffer.h"


void PixelBuffer::Draw(const SDL_Rect rect, Uint32 base_color, const std::vector<std::vector<int>>& noise_map, int grid_offset, int phase_offset) {
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
            int noise_x = ((x - x_start) * noise_size / (x_end - x_start) + phase_offset) % noise_size;
            int noise_y = ((y - y_start) * noise_size / (y_end - y_start) + phase_offset) % noise_size;
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

static Uint32 LerpColor(Uint32 a, Uint32 b, double t, SDL_PixelFormat* format) {
    Uint8 ar, ag, ab, aa;
    Uint8 br, bg, bb, ba;
    SDL_GetRGBA(a, format, &ar, &ag, &ab, &aa);
    SDL_GetRGBA(b, format, &br, &bg, &bb, &ba);

    auto r = static_cast<Uint8>(ar + (br - ar) * t);
    auto g = static_cast<Uint8>(ag + (bg - ag) * t);
    auto b_ = static_cast<Uint8>(ab + (bb - ab) * t);
    auto a_ = static_cast<Uint8>(aa + (ba - aa) * t);
    return SDL_MapRGBA(format, r, g, b_, a_);
}

void PixelBuffer::DrawBlendedEdge(const SDL_Rect& rect, Uint32 base_color, Uint32 neighbor_color, Edge edge) {
    const int strip_width = 4; // Width of the strip to blend

    int x_start = std::max(rect.x, 0);
    int y_start = std::max(rect.y, 0);
    int x_end = std::min(rect.x + rect.w, width_);
    int y_end = std::min(rect.y + rect.h, height_);

    if (edge == Edge::Left) {
        int x_max = std::min(x_start + strip_width, x_end);
        for (int x = x_start; x < x_max; ++x) {
            double t = static_cast<double>(x - x_start) / std::max(1, strip_width);
            Uint32 color = LerpColor(neighbor_color, base_color, t, format_);
            for (int y = y_start; y < y_end; ++y) {
                pixels_[y * width_ + x] = color;
            }
        }
    } else if (edge == Edge::Right) {
        int x_min = std::max(x_end - strip_width, x_start);
        for (int x = x_min; x < x_end; ++x) {
            double t = static_cast<double>(x_end - 1 - x) / std::max(1, strip_width);
            Uint32 color = LerpColor(neighbor_color, base_color, t, format_);
            for (int y = y_start; y < y_end; ++y) {
                pixels_[y * width_ + x] = color;
            }
        }
    } else if (edge == Edge::Top) {
        int y_max = std::min(y_start + strip_width, y_end);
        for (int y = y_start; y < y_max; ++y) {
            double t = static_cast<double>(y - y_start) / std::max(1, strip_width);
            Uint32 color = LerpColor(neighbor_color, base_color, t, format_);
            for (int x = x_start; x < x_end; ++x) {
                pixels_[y * width_ + x] = color;
            }
        }
    } else if (edge == Edge::Bottom) {
        int y_min = std::max(y_end - strip_width, y_start);
        for (int y = y_min; y < y_end; ++y) {
            double t = static_cast<double>(y_end - 1 - y) / std::max(1, strip_width);
            Uint32 color = LerpColor(neighbor_color, base_color, t, format_);
            for (int x = x_start; x < x_end; ++x) {
                pixels_[y * width_ + x] = color;
            }
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
