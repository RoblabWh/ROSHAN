//
// Created by nex on 08.06.23.
//

#ifndef ROSHAN_GAMEOFLIFE_INFINITE_RENDERER_H
#define ROSHAN_GAMEOFLIFE_INFINITE_RENDERER_H

#include "imgui.h"
#include "gameoflife_types.h"
#include <vector>
#include <SDL.h>
#include <unordered_map>
#include <memory>

class GameOfLifeInfiniteRenderer{

public:
    //only one instance of this class can be created
    static std::shared_ptr<GameOfLifeInfiniteRenderer> GetInstance(std::shared_ptr<SDL_Renderer> renderer) {
        if (instance_ == nullptr) {
            instance_ = std::shared_ptr<GameOfLifeInfiniteRenderer>(new GameOfLifeInfiniteRenderer(renderer));
        }
        return instance_;
    }
    ~GameOfLifeInfiniteRenderer(){}

    void Render(CellStateGOF state, int cell_size);
    ImVec4 background_color_ = ImVec4(0.0f, 0.0f, 0.0f, 1.00f);

private:
    explicit GameOfLifeInfiniteRenderer(std::shared_ptr<SDL_Renderer> renderer);
    static std::shared_ptr<GameOfLifeInfiniteRenderer> instance_;
    void DrawCells(CellStateGOF state, int cell_size);
    void DrawGrid(int cell_size);
    int width_;
    int height_;

    std::shared_ptr<SDL_Renderer> renderer_;

};


#endif //ROSHAN_GAMEOFLIFE_INFINITE_RENDERER_H
