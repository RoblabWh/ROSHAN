//
// Created by nex on 13.07.23.
//

#include "DroneRenderer.h"
#include <SDL_image.h>

DroneRenderer::DroneRenderer(std::shared_ptr<SDL_Renderer> renderer)
        : renderer_(renderer)
{
    // Load the arrow texture
    SDL_Surface* drone_surface = IMG_Load("../assets/drone.png");
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
    drone_texture_ = std::shared_ptr<SDL_Texture>(texture, textureDeleter);

    SDL_FreeSurface(drone_surface);
}

void DroneRenderer::Render(std::pair<int, int> position, int size, int view_range, double angle) {
    // Render the arrow
    SDL_Rect destRect = {position.first, position.second, size, size}; // x, y, width and height of the arrow
    SDL_RenderCopyEx(renderer_.get(), drone_texture_.get(), NULL, &destRect, angle * 180 / M_PI, NULL, SDL_FLIP_NONE);
    SDL_Rect view_range_rect = {position.first - (view_range * size) / 2, position.second - (view_range * size) / 2, (view_range + 1) * size, (view_range + 1) * size};
    SDL_SetRenderDrawColor(renderer_.get(), 0, 0, 0, 255);
    SDL_RenderDrawRect(renderer_.get(), &view_range_rect);
}
