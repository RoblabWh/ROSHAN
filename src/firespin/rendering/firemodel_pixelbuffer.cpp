//
// Created by nex on 02.07.23.
//

#include "firemodel_pixelbuffer.h"


void PixelBuffer::Draw(const SDL_Rect rect, Uint32 base_color, const std::vector<std::vector<int>>& noise_map, int grid_offset, int phase_offset) {
    int x_start = rect.x < 0 ? 0 : rect.x;
    int y_start = rect.y < 0 ? 0 : rect.y;
    int x_end = rect.x + rect.w > width_ ? width_ : rect.x + rect.w;
    int y_end = rect.y + rect.h > height_ ? height_ : rect.y + rect.h;

    int cell_w = x_end - x_start;
    int cell_h = y_end - y_start;
    if (cell_w == 0 || cell_h == 0) {
        return;
    }

    // Decompose ARGB8888 with bit ops instead of SDL_GetRGBA
    Uint8 a = (base_color >> 24) & 0xFF;
    Uint8 r = (base_color >> 16) & 0xFF;
    Uint8 g = (base_color >> 8) & 0xFF;
    Uint8 b = base_color & 0xFF;

    int noise_size = static_cast<int>(noise_map.size());

    for (int y = y_start; y < y_end + grid_offset; ++y) {
        int noise_y = ((y - y_start) * noise_size / cell_h + phase_offset) % noise_size;
        Uint32* row = &pixels_[y * width_];
        for (int x = x_start; x < x_end + grid_offset; ++x) {
            int noise_x = ((x - x_start) * noise_size / cell_w + phase_offset) % noise_size;
            int noise = noise_map[noise_y][noise_x];
            Uint8 new_r = std::clamp(static_cast<int>(r) + noise, 0, 255);
            Uint8 new_g = std::clamp(static_cast<int>(g) + noise, 0, 255);
            Uint8 new_b = std::clamp(static_cast<int>(b) + noise, 0, 255);
            // Pack ARGB8888 directly instead of SDL_MapRGBA
            row[x] = (static_cast<Uint32>(a) << 24) | (static_cast<Uint32>(new_r) << 16) |
                     (static_cast<Uint32>(new_g) << 8) | static_cast<Uint32>(new_b);
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

static Uint32 LerpColor(Uint32 a, Uint32 b, double t) {
    // Decompose ARGB8888 with bit ops
    Uint8 ar = (a >> 16) & 0xFF, ag = (a >> 8) & 0xFF, ab = a & 0xFF, aa = (a >> 24) & 0xFF;
    Uint8 br = (b >> 16) & 0xFF, bg = (b >> 8) & 0xFF, bb = b & 0xFF, ba = (b >> 24) & 0xFF;

    auto r = static_cast<Uint8>(ar + (br - ar) * t);
    auto g = static_cast<Uint8>(ag + (bg - ag) * t);
    auto b_ = static_cast<Uint8>(ab + (bb - ab) * t);
    auto a_ = static_cast<Uint8>(aa + (ba - aa) * t);
    return (static_cast<Uint32>(a_) << 24) | (static_cast<Uint32>(r) << 16) |
           (static_cast<Uint32>(g) << 8) | static_cast<Uint32>(b_);
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
            Uint32 color = LerpColor(neighbor_color, base_color, t);
            for (int y = y_start; y < y_end; ++y) {
                pixels_[y * width_ + x] = color;
            }
        }
    } else if (edge == Edge::Right) {
        int x_min = std::max(x_end - strip_width, x_start);
        for (int x = x_min; x < x_end; ++x) {
            double t = static_cast<double>(x_end - 1 - x) / std::max(1, strip_width);
            Uint32 color = LerpColor(neighbor_color, base_color, t);
            for (int y = y_start; y < y_end; ++y) {
                pixels_[y * width_ + x] = color;
            }
        }
    } else if (edge == Edge::Top) {
        int y_max = std::min(y_start + strip_width, y_end);
        for (int y = y_start; y < y_max; ++y) {
            double t = static_cast<double>(y - y_start) / std::max(1, strip_width);
            Uint32 color = LerpColor(neighbor_color, base_color, t);
            for (int x = x_start; x < x_end; ++x) {
                pixels_[y * width_ + x] = color;
            }
        }
    } else if (edge == Edge::Bottom) {
        int y_min = std::max(y_end - strip_width, y_start);
        for (int y = y_min; y < y_end; ++y) {
            double t = static_cast<double>(y_end - 1 - y) / std::max(1, strip_width);
            Uint32 color = LerpColor(neighbor_color, base_color, t);
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
