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
#include <climits>
#include <filesystem>
#include "imgui.h"
#include "imgui_impl_sdl2.h"
#include "imgui_impl_sdlrenderer2.h"
#include <cstdio>
#include <memory>
#include <SDL.h>
#include <iostream>
#include <thread>
#include "model_interface.h"
#include "firespin/firemodel.h"
#include "corine/dataset_handler.h"
#include "reinforcementlearning/agents/agent.h"
#include "reinforcementlearning/actions/action.h"
#include "src/utils.h"
#include "externals/pybind11/include/pybind11/pybind11.h"

#if !SDL_VERSION_ATLEAST(2,0,17)
#error This backend requires SDL 2.0.17+ because of SDL_RenderGeometry() function
#endif

namespace py = pybind11;

class EngineCore {

public:
    EngineCore()= default;
    ~EngineCore();

    bool Init(int mode, const std::string& map_path = "");
    void Clean();

    void Update();
    void Render();
    void HandleEvents();
    void SendDataToModel(std::string data);
    void UpdateReward();
    void SendRLStatusToModel(pybind11::dict status);
    pybind11::dict GetRLStatusFromModel();

    // RL-Related
    // Observe the current state of the environment
    bool AgentIsRunning();
    std::string GetUserInput();
    std::unordered_map<std::string, std::vector<std::deque<std::shared_ptr<State>>>> GetObservations();
    std::tuple<std::unordered_map<std::string, std::vector<std::deque<std::shared_ptr<State>>>>,
            std::vector<double>,
            std::vector<bool>,
            std::unordered_map<std::string, bool>,
            double>
    Step(const std::string& agent_type, std::vector<std::shared_ptr<Action>> actions);

    [[nodiscard]] inline bool IsRunning() const { return is_running_; }
    bool InitialModeSelectionDone();
    int GetViewRange(const std::string& agent_type);
    int GetTimeSteps();
    int GetMapSize();

private:

    bool is_running_{};

    SDL_Window* window_;
    SDL_Renderer* renderer_;
    SDL_WindowFlags window_flags_;

    ImGuiIO* io_;

    // Model
    std::shared_ptr<IModel> model_ = nullptr;

    // Simulation State
    bool update_simulation_ = false;
    bool render_simulation_ = true;
    int delay_ = 0;

    // For Init of the Window
    int width_ = 1600;
    int height_ = 900;

    // For the Node.js server
    static void StartServer();
    static void StopServer();

    // GUI Stuff
    bool SDLInit();
    bool ImGuiInit();

    // Flags
    Mode mode_;

    static void StyleColorsEnemyMouse(ImGuiStyle *dst);
};


#endif //ROSHAN_ENGINE_CORE_H
