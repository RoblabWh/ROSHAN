//
// Created by nex on 06.06.23.
//

#ifndef ROSHAN_ENGINE_CORE_H
#define ROSHAN_ENGINE_CORE_H

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <chrono>
#include <unistd.h>
#include <limits.h>
#include <filesystem>
#include "imgui.h"
#include "imgui_impl_sdl2.h"
#include "imgui_impl_sdlrenderer2.h"
#include <stdio.h>
#include <SDL.h>
#include <iostream>
#include "model_interface.h"
#include "models/firespin/firemodel.h"
#include "corine/dataset_handler.h"
#include "agent.h"
#include "action.h"

#if !SDL_VERSION_ATLEAST(2,0,17)
#error This backend requires SDL 2.0.17+ because of SDL_RenderGeometry() function
#endif

class EngineCore {

public:
    static std::shared_ptr<EngineCore> GetInstance() {
        if (instance_ == nullptr) {
            instance_ = std::shared_ptr<EngineCore>(new EngineCore());
        }
        return instance_;
    }
    EngineCore(){}
    ~EngineCore(){}

    bool Init(int mode);
    void Clean();

    void Update();
    void Render();
    void HandleEvents();
    void SendDataToModel(std::string data);

    // RL-Related
    // Observe the current state of the environment
    bool AgentIsRunning();
    std::string GetUserInput();
    std::vector<std::deque<std::shared_ptr<State>>> GetObservations();
    std::tuple<std::vector<std::deque<std::shared_ptr<State>>>, std::vector<double>, std::vector<bool>, std::pair<bool, bool>>
    Step(std::vector<std::shared_ptr<Action>> actions);

    inline bool IsRunning() { return is_running_; }
    void ImGuiSimulationControls(bool &update_simulation, bool &render_simulation, int &delay);

private:

    bool is_running_;

    std::shared_ptr<SDL_Window> window_;
    std::shared_ptr<SDL_Renderer> renderer_;
    SDL_WindowFlags window_flags_;

    std::shared_ptr<ImGuiIO> io_;

    // Our model
    std::shared_ptr<IModel> model_;

    // Our Simulation State
    bool update_simulation_ = false;
    bool render_simulation_ = true;
    int delay_ = 0;

    // For Init of the Window
    int width_ = 1600;
    int height_ = 900;

    // For the Node.js server
    void StartServer();
    void StopServer();

    static std::shared_ptr<EngineCore> instance_;

    // ImGui Stuff
    bool ImGuiModelSelection();

    // AI Stuff
    int mode_;
    std::shared_ptr<Agent> agent_;

    void StyleColorsEnemyMouse(ImGuiStyle *dst);
};


#endif //ROSHAN_ENGINE_CORE_H
