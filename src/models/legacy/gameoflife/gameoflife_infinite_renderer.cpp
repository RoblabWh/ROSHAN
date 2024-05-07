//
// Created by nex on 08.06.23.
//

#include "gameoflife_infinite_renderer.h"

std::shared_ptr<GameOfLifeInfiniteRenderer> GameOfLifeInfiniteRenderer::instance_ = nullptr;


void GameOfLifeInfiniteRenderer::Render(CellStateGOF state, int cell_size) {
    SDL_SetRenderDrawColor(renderer_.get(), (Uint8)(background_color_.x * 255), (Uint8)(background_color_.y * 255), (Uint8)(background_color_.z * 255), (Uint8)(background_color_.w * 255));
    SDL_RenderClear(renderer_.get());
    SDL_GetRendererOutputSize(renderer_.get(), &width_, &height_);
    DrawCells(state, cell_size);
    DrawGrid(cell_size);
}

GameOfLifeInfiniteRenderer::GameOfLifeInfiniteRenderer(std::shared_ptr<SDL_Renderer> renderer)
    : renderer_(renderer) {
    SDL_GetRendererOutputSize(renderer_.get(), &width_, &height_);
}

void GameOfLifeInfiniteRenderer::DrawCells(CellStateGOF state, int cell_size) {
    SDL_SetRenderDrawColor(renderer_.get(), 255, 255, 255, 255);

    for (const auto& [cell, is_alive] : state) {
        if (is_alive) {
            SDL_Rect cellRect = {cell.first * cell_size, cell.second * cell_size, cell_size, cell_size};
            SDL_RenderFillRect(renderer_.get(), &cellRect);
        }
    }
}

void GameOfLifeInfiniteRenderer::DrawGrid(int cell_size) {
    SDL_SetRenderDrawColor(renderer_.get(), 53, 53, 53, 255);  // color for the grid

    // Draw vertical grid lines
    for (int i = 0; i <= width_; i += cell_size) {
        SDL_RenderDrawLine(renderer_.get(), i, 0, i, height_);
    }

    // Draw horizontal grid lines
    for (int i = 0; i <= height_; i += cell_size) {
        SDL_RenderDrawLine(renderer_.get(), 0, i, width_, i);
    }
}
