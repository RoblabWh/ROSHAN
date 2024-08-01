//
// Created by nex on 08.06.23.
//

#ifndef ROSHAN_FIREMODEL_H
#define ROSHAN_FIREMODEL_H

#define NOSPEEDTEST

#include <cstdlib>
#include <iostream>
#include <set>
#include <map>
#include <chrono>
#include <thread>
#include "src/models/firespin/utils.h"
#include "model_interface.h"
#include "firemodel_gridmap.h"
#include "src/models/firespin/rendering/firemodel_renderer.h"
#include "models/firespin/rendering/firemodel_imgui.h"
#include "model_parameters.h"
#include "wind.h"
#include "corine/dataset_handler.h"
#include "reinforcementlearning/drone_agent/drone.h"
#include "reinforcementlearning/drone_agent/drone_action.h"
#include "reinforcementlearning/reinforcementlearning_handler.h"
#include "src/utils.h"

namespace py = pybind11;

class FireModel : public IModel{
public:
    //only one instance of this class can be created
    static std::shared_ptr<FireModel> GetInstance(Mode mode, const std::string& map_path="") {
        if (instance_ == nullptr) {
            instance_ = std::shared_ptr<FireModel>(new FireModel(mode, map_path));
        }

        return instance_;
    }
    ~FireModel() override;

    void Update() override;
    std::tuple<std::vector<std::deque<std::shared_ptr<State>>>, std::vector<double>, std::vector<bool>, std::pair<bool, bool>, double> Step(std::vector<std::shared_ptr<Action>> actions) override;
    std::vector<std::deque<std::shared_ptr<State>>> GetObservations() override;
    void Render() override;
    void SetRenderer(std::shared_ptr<SDL_Renderer> renderer) override;
    bool AgentIsRunning() override;
    void HandleEvents(SDL_Event event, ImGuiIO* io) override;
    void ImGuiRendering(bool &update_simulation, bool &render_simulation, int &delay, float framerate) override;
    std::string GetUserInput() override;
    void GetData(std::string data) override;
    void GetRLStatus(pybind11::dict status) override;
    int GetViewRange() override;
    int GetTimeSteps() override;
    void LoadMap(std::string path);

private:
    explicit FireModel(Mode mode, const std::string& map_path);

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

    void ResetGridMap(std::vector<std::vector<int>>* rasterData = nullptr);

    //Current RasterData for the GridMap
    std::vector<std::vector<int>> current_raster_data_;
    // Agent Stuff
    std::shared_ptr<ReinforcementLearningHandler> rl_handler_;

    // Flags
    Mode mode_;
    bool agent_is_running_;

    // Dirty Variables
    std::string user_input_;
    std::string model_output_;

    //Measurement
    Timer timer_;

    //ImGui Stuff
    std::shared_ptr<ImguiHandler> imgui_handler_;
    void setupImGui();

    void SetUniformRasterData();
    void FillRasterWithEnum();
    void TestBurndownHeadless();
    void StartFires(int percentage);

    void setupRLHandler();
};


#endif //ROSHAN_FIREMODEL_H
