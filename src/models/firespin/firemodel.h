//
// Created by nex on 08.06.23.
//

#ifndef ROSHAN_FIREMODEL_H
#define ROSHAN_FIREMODEL_H

#include <cstdlib>
#include <iostream>
#include <set>
#include <map>
#include <chrono>
#include <thread>
#include "utils.h"
#include "model_interface.h"
#include "firemodel_gridmap.h"
#include "src/models/firespin/rendering/firemodel_renderer.h"
#include "models/firespin/rendering/firemodel_imgui.h"
#include "model_parameters.h"
#include "wind.h"
#include "corine/dataset_handler.h"
#include "agents/drone_agent/drone.h"
#include "agents/drone_agent/drone_action.h"

class FireModel : public IModel{
public:
    //only one instance of this class can be created
    static std::shared_ptr<FireModel> GetInstance(std::shared_ptr<SDL_Renderer> renderer, int mode) {
        if (instance_ == nullptr) {
            instance_ = std::shared_ptr<FireModel>(new FireModel(renderer, mode));
        }

        return instance_;
    }
    ~FireModel() override;

    void Initialize() override;
    void Update() override;
    std::tuple<std::vector<std::deque<std::shared_ptr<State>>>, std::vector<double>, std::vector<bool>, std::pair<bool, bool>> Step(std::vector<std::shared_ptr<Action>> actions) override;
    std::vector<std::deque<std::shared_ptr<State>>> GetObservations() override;
    void Reset() override;
    void Render() override;
    bool AgentIsRunning() override;
    void SetWidthHeight(int width, int height) override;
    void HandleEvents(SDL_Event event, ImGuiIO* io) override;
    void ImGuiSimulationSpeed() override;
    void ImGuiRendering(std::function<void(bool&, bool&, int&)> controls, bool &update_simulation, bool &render_simulation, int &delay) override;
    std::string GetUserInput() override;
    void GetData(std::string data) override;

private:
    explicit FireModel(std::shared_ptr<SDL_Renderer> renderer, int mode);

    std::shared_ptr<GridMap> gridmap_;
    std::shared_ptr<FireModelRenderer> model_renderer_;
    std::shared_ptr<Wind> wind_;
    FireModelParameters parameters_;
    static std::shared_ptr<FireModel> instance_;
    double running_time_;

    std::shared_ptr<DatasetHandler> dataset_handler_;
    void ResetGridMap(std::vector<std::vector<int>>* rasterData = nullptr);

    //current data
    std::vector<std::vector<int>> current_raster_data_;
    // Agent Stuff
    bool MoveDrone(int drone_idx, double speed_x, double speed_y, int water_dispense);
    bool MoveDroneByAngle(int drone_idx, double netout_speed, double netout_angle, int water_dispense);
    double CalculateReward(bool drone_in_grid, bool fire_extinguished, bool drone_terminal, int water_dispensed, int near_fires, double max_distance, double distance_to_fire);
    void ResetDrones();
    std::shared_ptr<std::vector<std::shared_ptr<DroneAgent>>> drones_;
    std::vector<std::deque<DroneState>> observations_;
    CircularBuffer<float> rewards_;
    std::vector<float> all_rewards_;

    // RL Flags
    bool python_code_ = true;
    bool agent_is_running_ = false;

    // Dirty Variables
    double last_distance_to_fire_;
    int last_near_fires_;
    std::string user_input_;
    std::string model_output_;

    //ImGui Stuff
    std::shared_ptr<ImguiHandler> imgui_handler_;
    void setupImGui();

    void RandomizeCells();
    void SetUniformRasterData();
    void FillRasterWithEnum();
};


#endif //ROSHAN_FIREMODEL_H
