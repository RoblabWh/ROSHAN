//
// Created by nex on 28.11.24.
//

#include "texturerenderer.h"

#include <SDL_image.h>

TextureRenderer::TextureRenderer(std::shared_ptr<SDL_Renderer> renderer, const char *texture_path)
        : renderer_(renderer)
{
    // Load the arrow texture
    SDL_Surface* drone_surface = IMG_Load(texture_path);
    if (drone_surface == NULL) {
        SDL_Log("Unable to load image: %s", SDL_GetError());
        throw std::runtime_error("Unable to load image");
    }

    SDL_Texture* texture = SDL_CreateTextureFromSurface(renderer_.get(), drone_surface);
    if (!texture) {
        SDL_Log("Unable to create texture: %s", SDL_GetError());
        SDL_FreeSurface(drone_surface);
        throw std::runtime_error("Unable to create texture");
    }

    auto textureDeleter = [](SDL_Texture* t) { SDL_DestroyTexture(t); };
    texture_ = std::shared_ptr<SDL_Texture>(texture, textureDeleter);

    SDL_FreeSurface(drone_surface);
}

void TextureRenderer::Render(std::pair<int, int> position, int size, int view_range, double angle, bool active) {
    // Render the Drone
    SDL_Rect destRect = {position.first, position.second, size, size}; // x, y, width and height of the arrow
    SDL_RenderCopyEx(renderer_.get(), texture_.get(), NULL, &destRect, angle * 180 / M_PI, NULL, SDL_FLIP_NONE);
    SDL_Rect view_range_rect = {position.first - (view_range * size) / 2, position.second - (view_range * size) / 2, (view_range + 1) * size, (view_range + 1) * size};
    if (active) {
        SDL_SetRenderDrawColor(renderer_.get(), 255, 0, 0, 255);
    } else {
        SDL_SetRenderDrawColor(renderer_.get(), 20, 20, 20, 255);
    }
    SDL_RenderDrawRect(renderer_.get(), &view_range_rect);
}

void TextureRenderer::RenderGoal(std::pair<double, double> position, int size) {
    // Render the Goal
    SDL_Rect destRect = {static_cast<int>(position.first), static_cast<int>(position.second), size, size}; // x, y, width and height of the arrow
    SDL_RenderCopyEx(renderer_.get(), texture_.get(), NULL, &destRect, 0, NULL, SDL_FLIP_NONE);
}
