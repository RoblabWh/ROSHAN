//
// Created by nex on 02.07.23.
//

#ifndef ROSHAN_FIREMODEL_PIXELBUFFER_H
#define ROSHAN_FIREMODEL_PIXELBUFFER_H

#include <SDL.h>
#include <vector>
#include <algorithm>
#include "imgui.h"

enum class Edge { Left, Right, Top, Bottom };

class PixelBuffer {
public:
    PixelBuffer(int width, int height, SDL_Color background_color, SDL_PixelFormat* format);
    ~PixelBuffer() = default;

    void Draw(SDL_Rect rect, Uint32 base_color, int grid_offset = 0);
    void Draw(SDL_Rect rect, Uint32 base_color, const std::vector<std::vector<int>>& noise_map, int grid_offset = 0, int phase_offset = 0);
    void DrawBlendedEdge(const SDL_Rect& rect, Uint32 base_color, Uint32 neighbor_color, Edge edge);
    void Reset();
    void Resize(int width, int height);
    [[nodiscard]] int GetPitch() const { return static_cast<int>(width_ * sizeof(Uint32)); }
    [[nodiscard]] int GetWidth() const { return width_; }
    [[nodiscard]] int GetHeight() const { return height_; }
    Uint32* GetData() { return pixels_.data(); }
private:
    int width_;
    int height_;
    std::vector<Uint32> pixels_;
    SDL_PixelFormat* format_;
    Uint32 background_color_;

    Uint32 SDLColorToUint32(SDL_Color color);
};


#endif //ROSHAN_FIREMODEL_PIXELBUFFER_H
