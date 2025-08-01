//
// Created by nex on 28.11.24.
//

#ifndef ROSHAN_TEXTURERENDERER_H
#define ROSHAN_TEXTURERENDERER_H

#include <SDL.h>
#include <memory>
#include <stdexcept>

class TextureRenderer {
public:
    TextureRenderer() = default;
    TextureRenderer(SDL_Renderer* renderer, const char *texture_path);
    ~TextureRenderer() = default;
    void Render(std::pair<int, int> position, int size, int view_range, double angle, bool active=false, bool fast_drone=false);
    void RenderGoal(std::pair<double, double> position, int size);
private:
    std::shared_ptr<SDL_Texture> texture_;
    SDL_Renderer* renderer_;
};


#endif //ROSHAN_TEXTURERENDERER_H
