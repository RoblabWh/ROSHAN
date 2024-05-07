//
// Created by nex on 08.06.23.
//

#ifndef ROSHAN_GAMEOFLIFESIMPLE_RENDERER_H
#define ROSHAN_GAMEOFLIFESIMPLE_RENDERER_H

#include "imgui.h"
#include <vector>
#include <SDL.h>
#include <unordered_map>
#include <memory>

class GameOfLifeFixedRenderer{

public:
    //only one instance of this class can be created
    static std::shared_ptr<GameOfLifeFixedRenderer> GetInstance(std::shared_ptr<SDL_Renderer> renderer) {
        if (instance_ == nullptr) {
            instance_ = std::shared_ptr<GameOfLifeFixedRenderer>(new GameOfLifeFixedRenderer(renderer.get()));
        }
        return instance_;
    }
    void Render(std::vector<std::vector<bool>> state, int cell_size, int rows, int cols);
    ImVec4 background_color_ = ImVec4(0.0f, 0.0f, 0.0f, 1.00f);

private:
    explicit GameOfLifeFixedRenderer(SDL_Renderer* renderer);
    //~GameOfLifeInfiniteRenderer(){}
    static std::shared_ptr<GameOfLifeFixedRenderer> instance_;
    void DrawCells(std::vector<std::vector<bool>> state, int cell_size, int rows, int cols);
    void DrawGrid(int cell_size, int rows, int cols);

    int width_;
    int height_;

    SDL_Renderer* renderer_;

};


#endif //ROSHAN_GAMEOFLIFE_INFINITE_RENDERER_H
