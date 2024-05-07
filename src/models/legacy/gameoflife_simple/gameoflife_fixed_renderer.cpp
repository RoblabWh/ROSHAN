//
// Created by nex on 08.06.23.
//

#include "gameoflife_fixed_renderer.h"

std::shared_ptr<GameOfLifeFixedRenderer> GameOfLifeFixedRenderer::instance_ = nullptr;

void GameOfLifeFixedRenderer::Render(std::vector<std::vector<bool>> state, int cell_size, int rows, int cols) {
    SDL_SetRenderDrawColor(renderer_, (Uint8)(background_color_.x * 255), (Uint8)(background_color_.y * 255), (Uint8)(background_color_.z * 255), (Uint8)(background_color_.w * 255));
    SDL_RenderClear(renderer_);
    SDL_GetRendererOutputSize(renderer_, &width_, &height_);
    DrawCells(state, cell_size, rows, cols);
    DrawGrid(cell_size, rows, cols);
}

GameOfLifeFixedRenderer::GameOfLifeFixedRenderer(SDL_Renderer *renderer) {
    renderer_ = renderer;
}

void GameOfLifeFixedRenderer::DrawCells(std::vector<std::vector<bool>> state, int cell_size, int rows, int cols) {
    SDL_SetRenderDrawColor(renderer_, 255, 255, 255, 255);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (state[i][j]) {
                SDL_Rect cellRect = {i * cell_size, j * cell_size, cell_size, cell_size};
                SDL_RenderFillRect(renderer_, &cellRect);
            }
        }
    }
}

void GameOfLifeFixedRenderer::DrawGrid(int cell_size, int rows, int cols) {
    SDL_SetRenderDrawColor(renderer_, 53, 53, 53, 255);  // color for the grid

    // Draw vertical grid lines
    for (int i = 0; i <= rows * cell_size; i += cell_size) {
        SDL_RenderDrawLine(renderer_, i, 0, i, cols * cell_size);
    }

    // Draw horizontal grid lines
    for (int i = 0; i <= cols * cell_size; i += cell_size) {
        SDL_RenderDrawLine(renderer_, 0, i, rows * cell_size, i);
    }
}

//void GameOfLifeInfiniteRenderer::DrawCells(int width, int height, std::vector<std::vector<bool>> state, int cell_size) {
//    SDL_SetRenderDrawColor(renderer_, 255, 255, 255, 255);
//
//    int screenWidth, screenHeight;
//    SDL_GetRendererOutputSize(renderer_, &screenWidth, &screenHeight);
//
//    int rows = std::ceil(static_cast<float>(screenWidth) / cell_size);
//    int cols = std::ceil(static_cast<float>(screenHeight) / cell_size);
//
//    for (int i = 0; i < rows; ++i) {
//        for (int j = 0; j < cols; ++j) {
//            if (state[i][j]) {
//                SDL_Rect cellRect = {i * cell_size, j * cell_size, cell_size, cell_size};
//                SDL_RenderFillRect(renderer_, &cellRect);
//            }
//        }
//    }
//}
//
//void GameOfLifeInfiniteRenderer::DrawGrid(int width, int height, int cell_size) {
//    SDL_SetRenderDrawColor(renderer_, 53, 53, 53, 255);  // color for the grid
//
//    // Draw vertical grid lines
//    for (int i = 0; i <= width; i += cell_size) {
//        SDL_RenderDrawLine(renderer_, i, 0, i, height);
//    }
//
//    // Draw horizontal grid lines
//    for (int i = 0; i <= height; i += cell_size) {
//        SDL_RenderDrawLine(renderer_, 0, i, width, i);
//    }
//}
