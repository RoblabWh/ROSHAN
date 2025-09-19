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

void TextureRenderer::RenderDrone(std::pair<int, int> position, int drone_size, int alpha) {
    // Render the drone texture at the actual position
    SDL_Rect destRect = {position.first, position.second, drone_size, drone_size}; // x, y, width and height of the arrow
    SDL_SetTextureColorMod(texture_.get(), 255, 255, 255);

    SDL_SetTextureAlphaMod(texture_.get(), alpha);
    SDL_RenderCopyEx(renderer_, texture_.get(), nullptr, &destRect, 0, nullptr, SDL_FLIP_NONE);
}

void TextureRenderer::RenderViewRange(std::pair<int, int> position, int size, int view_range, int alpha) {
    SDL_SetRenderDrawColor(renderer_, 20, 20, 20, alpha);
    SDL_Rect view_range_rect = {position.first - (view_range * size) / 2, position.second - (view_range * size) / 2, (view_range + 1) * size, (view_range + 1) * size};
    SDL_RenderDrawRect(renderer_, &view_range_rect);
}

void TextureRenderer::RenderGroundStation(std::pair<int, int> position, int size) {
    // Render the Goal
    SDL_Rect destRect = {position.first, position.second, size, size}; // x, y, width and height of the arrow
    SDL_RenderCopyEx(renderer_, texture_.get(), nullptr, &destRect, 0, nullptr, SDL_FLIP_NONE);
}

void TextureRenderer::Render(std::pair<int, int> position, int size, int view_range, double angle, bool active, bool fast_drone) {
    auto alpha = 255; // Default alpha value for the drone
    if (fast_drone) alpha = 190; // Reduce alpha for fast drones to reduce clutter

    // Shadow experiment (looks shit)
//    if(view_range > 0) {
//        // Render shadow with slight offset
//        SDL_Rect shadowRect = {position.first - 2, position.second - 2, static_cast<int>(size * 0.8), static_cast<int>(size * 0.8)};
//        // Scale shadow alpha relative to the drone alpha to respect fast_drone settings
//        auto shadow_alpha = alpha * 80 / 255;
//        SDL_SetTextureColorMod(texture_.get(), 20, 20, 20);
//        SDL_SetTextureAlphaMod(texture_.get(), shadow_alpha);
//        SDL_RenderCopyEx(renderer_, texture_.get(), nullptr, &shadowRect, angle * 180 / M_PI, nullptr, SDL_FLIP_NONE);
//    }
    auto drone_size = size / 5;
    auto drone_offset = drone_size / 2;
    // Render the drone texture at the actual position
    SDL_Rect destRect = {position.first - drone_offset, position.second - drone_offset, drone_size, drone_size}; // x, y, width and height of the arrow
    SDL_SetTextureColorMod(texture_.get(), 255, 255, 255);

    SDL_SetTextureAlphaMod(texture_.get(), alpha);
    SDL_RenderCopyEx(renderer_, texture_.get(), nullptr, &destRect, angle * 180 / M_PI, nullptr, SDL_FLIP_NONE);

    if (view_range > 0) {
        if (active) {
            SDL_SetRenderDrawColor(renderer_, 255, 0, 0, alpha);
        } else {
            SDL_SetRenderDrawColor(renderer_, 20, 20, 20, alpha);
        }
        auto view_offset = size - drone_offset;
        SDL_Rect view_range_rect = {position.first - (view_range * view_offset) / 2, position.second - (view_range * view_offset) / 2, (view_range + 1) * view_offset, (view_range + 1) * view_offset};
        SDL_RenderDrawRect(renderer_, &view_range_rect);
    }
}

void TextureRenderer::RenderGoal(std::pair<double, double> position, int size) {
    // Render the Goal
    SDL_Rect destRect = {static_cast<int>(position.first), static_cast<int>(position.second), size, size}; // x, y, width and height of the arrow
    SDL_RenderCopyEx(renderer_, texture_.get(), nullptr, &destRect, 0, nullptr, SDL_FLIP_NONE);
}
