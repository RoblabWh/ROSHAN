//
// Created by nex on 28.11.24.
//

#include "texturerenderer.h"

#include <SDL_image.h>

#include <utility>

TextureRenderer::TextureRenderer(SDL_Renderer* renderer, const char *texture_path)
        : renderer_(renderer)
{
    // Load the arrow texture
    SDL_Surface* drone_surface = IMG_Load(texture_path);
    if (drone_surface == nullptr) {
        SDL_Log("Unable to load image: %s", SDL_GetError());
        throw std::runtime_error("Unable to load image");
    }

    SDL_Texture* texture = SDL_CreateTextureFromSurface(renderer_, drone_surface);
    if (!texture) {
        SDL_Log("Unable to create texture: %s", SDL_GetError());
        SDL_FreeSurface(drone_surface);
        throw std::runtime_error("Unable to create texture");
    }

    auto textureDeleter = [](SDL_Texture* t) { SDL_DestroyTexture(t); };
    texture_ = std::shared_ptr<SDL_Texture>(texture, textureDeleter);

    SDL_FreeSurface(drone_surface);
}

void TextureRenderer::Render(std::pair<int, int> position, int size, int view_range, double angle, bool active, bool fast_drone) {
    // Render the Drone
    SDL_Rect destRect = {position.first, position.second, size, size}; // x, y, width and height of the arrow

    auto alpha = 255; // Default alpha value
    if (fast_drone) alpha = 190;
    SDL_SetTextureAlphaMod(texture_.get(), alpha);
    SDL_RenderCopyEx(renderer_, texture_.get(), nullptr, &destRect, angle * 180 / M_PI, nullptr, SDL_FLIP_NONE);
    SDL_Rect view_range_rect = {position.first - (view_range * size) / 2, position.second - (view_range * size) / 2, (view_range + 1) * size, (view_range + 1) * size};
    if (active) {
        SDL_SetRenderDrawColor(renderer_, 255, 0, 0, alpha);
    } else {
        SDL_SetRenderDrawColor(renderer_, 20, 20, 20, alpha);
    }
    SDL_RenderDrawRect(renderer_, &view_range_rect);
}

void TextureRenderer::RenderGoal(std::pair<double, double> position, int size) {
    // Render the Goal
    SDL_Rect destRect = {static_cast<int>(position.first), static_cast<int>(position.second), size, size}; // x, y, width and height of the arrow
    SDL_RenderCopyEx(renderer_, texture_.get(), nullptr, &destRect, 0, nullptr, SDL_FLIP_NONE);
}
