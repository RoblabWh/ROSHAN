//
// Created by nex on 02.07.23.
//

#ifndef ROSHAN_FIREMODEL_PIXELBUFFER_H
#define ROSHAN_FIREMODEL_PIXELBUFFER_H

#include <SDL.h>
#include <vector>
#include <algorithm>
#include "imgui.h"

class PixelBuffer {
public:
    PixelBuffer(int width, int height, SDL_Color background_color, SDL_PixelFormat* format);
    ~PixelBuffer() {}

    void Draw(const SDL_Rect rect, Uint32 base_color, int grid_offset = 0);
    void Draw(const SDL_Rect rect, Uint32 base_color, const std::vector<std::vector<int>>& noise_map, int grid_offset = 0);
    void DrawGrid(const SDL_Rect rect, Uint32 color);
    void Reset();
    void Resize(int width, int height);
    int GetPitch() const { return width_ * sizeof(Uint32); }
    int GetWidth() const { return width_; }
    int GetHeight() const { return height_; }
    Uint32* GetData() { return pixels_.data(); }
private:
    int width_;
    int height_;
    std::vector<Uint32> pixels_;
    SDL_PixelFormat* format_;
    Uint32 background_color_;

    Uint32 SDLColorToUint32(const SDL_Color color);
};


#endif //ROSHAN_FIREMODEL_PIXELBUFFER_H
