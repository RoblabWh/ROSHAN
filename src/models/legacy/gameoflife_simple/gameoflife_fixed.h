//
// Created by nex on 07.06.23.
//

#ifndef ROSHAN_GAMEOFLIFESIMPLE_H
#define ROSHAN_GAMEOFLIFESIMPLE_H

#include "model_interface.h"
#include "gameoflife_fixed_renderer.h"
#include "imgui.h"
#include <SDL.h>
#include <random>
#include <chrono>


class GameOfLifeFixed : public IModel{

public:
    //only one instance of this class can be created
    static std::shared_ptr<GameOfLifeFixed> GetInstance(std::shared_ptr<SDL_Renderer> renderer) {
        if (instance_ == nullptr) {
            instance_ = std::shared_ptr<GameOfLifeFixed>(new GameOfLifeFixed(renderer));
        }
        return instance_;
    }

    ~GameOfLifeFixed(){}

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
    GameOfLifeFixed(std::shared_ptr<SDL_Renderer> renderer);

    void RandomizeCells();

    std::shared_ptr<GameOfLifeFixedRenderer> model_renderer_;
    int width_ = 0;
    int height_ = 0;
    int cell_size_ = 10;

    int rows_;
    int cols_;
    int current_state_;
    std::vector<std::vector<bool>> state_[2];


    //std::vector<std::vector<bool>> cellState;
    static std::shared_ptr<GameOfLifeFixed> instance_;

};


#endif //ROSHAN_GAMEOFLIFE_INFINITE_H
