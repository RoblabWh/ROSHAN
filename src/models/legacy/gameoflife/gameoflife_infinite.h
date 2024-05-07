//
// Created by nex on 07.06.23.
//

#ifndef ROSHAN_GAMEOFLIFE_INFINITE_H
#define ROSHAN_GAMEOFLIFE_INFINITE_H

#include "model_interface.h"
#include "gameoflife_types.h"
#include "gameoflife_infinite_renderer.h"
#include "imgui.h"
#include <SDL.h>
#include <random>
#include <chrono>


class GameOfLifeInfinite : public IModel{

public:
    //only one instance of this class can be created
    static std::shared_ptr<GameOfLifeInfinite> GetInstance(std::shared_ptr<SDL_Renderer> renderer) {
        if (instance_ == nullptr) {
            instance_ = std::shared_ptr<GameOfLifeInfinite>(new GameOfLifeInfinite(renderer));
        }
        return instance_;
    }

    ~GameOfLifeInfinite(){}

    void Initialize() override;
    void Update() override;
    std::tuple<std::vector<std::deque<std::shared_ptr<State>>>, std::vector<double>, std::vector<bool>, std::pair<bool, bool>> Step(std::vector<std::shared_ptr<Action>> actions) override;
    std::vector<std::deque<std::shared_ptr<State>>> GetObservations() override;
    void Reset() override;
    void Config() override;
    void Render() override;
    bool AgentIsRunning() override {return false;}
    void SetWidthHeight(int width, int height) override;
    void HandleEvents(SDL_Event event, ImGuiIO* io) override;
    void ShowPopups() override;
    void ImGuiSimulationSpeed() override;
    void ImGuiModelMenu() override;
    void ShowControls(std::function<void(bool&, bool&, int&)> controls, bool &update_simulation, bool &render_simulation, int &delay) override;

private:
    GameOfLifeInfinite(std::shared_ptr<SDL_Renderer> renderer);

    void RandomizeCells();

    CellStateGOF state_;
    Neighbors GetNeighbors(const Cell& cell) const;
    int CountLiveNeighbors(const Cell& cell) const;

    std::shared_ptr<GameOfLifeInfiniteRenderer> model_renderer_;
    int width_ = 0;
    int height_ = 0;
    int cell_size_ = 10;


    //std::vector<std::vector<bool>> cellState;
    static std::shared_ptr<GameOfLifeInfinite> instance_;

};


#endif //ROSHAN_GAMEOFLIFE_INFINITE_H
