//
// Created by nex on 08.06.23.
//

#ifndef ROSHAN_FIREMODEL_H
#define ROSHAN_FIREMODEL_H

#define NOSPEEDTEST

#include <cstdlib>
#include <iostream>
#include <chrono>
#include <thread>
#include <random>
#include "utils.h"
#include "model_interface.h"
#include "firemodel_gridmap.h"
#include "firespin/rendering/firemodel_renderer.h"
#include "firespin/rendering/imgui/UIManager.h"
#include "model_parameters.h"
#include "wind.h"
#include "corine/dataset_handler.h"
#include "reinforcementlearning/actions/fly_action.h"
#include "reinforcementlearning/rl_handler.h"
#include "src/utils.h"
#include "fire_generator.h"

namespace py = pybind11;

class __attribute__((visibility("hidden"))) FireModel : public IModel{
public:
    explicit FireModel(Mode mode, const std::string& config_path);

    static std::shared_ptr<FireModel> Create(Mode mode, const std::string& config_path) {
        return std::make_shared<FireModel>(mode, config_path);
    }

    ~FireModel() override;

    void Update() override;
//    std::tuple<std::unordered_map<std::string,std::vector<std::deque<std::shared_ptr<State>>>>,
//    std::vector<double>,
//    std::vector<bool>,
//    std::unordered_map<std::string, bool>,
//    double>
    StepResult Step(const std::string& agent_type, std::vector<std::shared_ptr<Action>> actions) override;
    std::unordered_map<std::string, std::vector<std::deque<std::shared_ptr<State>>>> GetObservations() override;
    void Render() override;
    void SetRenderer(SDL_Renderer* renderer) override;
    bool GetEarlyClosing() override;
    void HandleEvents(SDL_Event event, ImGuiIO* io) override;
    void ImGuiRendering(bool &update_simulation, bool &render_simulation, int &delay, float framerate) override;
    std::string GetUserInput() override;
    void GetData(std::string data) override;
    bool AgentIsRunning() override;
    void SetRLStatus(pybind11::dict status) override;
    void UpdateReward() override;
    pybind11::dict GetRLStatus() override;
    void InitializeMap() override;
    void LoadMap(const std::string& path);
    bool InitialModeSelectionDone() override;
    void CheckReset() override;
private:

    // GridMap for the FireModel
    std::shared_ptr<GridMap> gridmap_;
    // Datahandler for the CORINE Dataset
    std::shared_ptr<DatasetHandler> dataset_handler_;
    // Renderer for the FireModel
    std::shared_ptr<FireModelRenderer> model_renderer_;
    std::shared_ptr<Wind> wind_;
    FireModelParameters parameters_;
    static std::shared_ptr<FireModel> instance_;
    double running_time_;

    void ResetGridMap(std::vector<std::vector<int>>* rasterData = nullptr, bool full_reset = false);

    //Current RasterData for the GridMap
    std::vector<std::vector<int>> current_raster_data_;
    // Agent Stuff
    std::shared_ptr<ReinforcementLearningHandler> rl_handler_;
    std::shared_ptr<FireGenerator> fire_generator_;

    // Flags
    Mode mode_;
    bool reset_next_step_ = false;
    std::string last_rl_mode_{};

    // Dirty Variables
    std::string user_input_;
    std::string model_output_;

    //Measurement
    Timer timer_;

    //ImGui Stuff
    std::unique_ptr<ui::UIManager> ui_manager_;
    void setupImGui();

    void SetUniformRasterData();
    void FillRasterWithEnum();
    void TestBurndownHeadless();

    void setupRLHandler();
};


#endif //ROSHAN_FIREMODEL_H
