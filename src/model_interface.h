//
// Created by nex on 07.06.23.
//

#ifndef ROSHAN_MODEL_INTERFACE_H
#define ROSHAN_MODEL_INTERFACE_H

#include <vector>
#include "imgui.h"
#include <SDL.h>
#include <functional>
#include <deque>
#include "state.h"
#include "reinforcementlearning/actions/action.h"
#include <memory>
#include "externals/pybind11/include/pybind11/pybind11.h"
#include "utils.h"

class IModel {
public:
    virtual ~IModel() = default;

    virtual void Update() = 0;
//    virtual std::tuple<std::unordered_map<std::string,std::vector<std::deque<std::shared_ptr<State>>>>,
//    std::vector<double>,
//    std::vector<bool>,
//    std::unordered_map<std::string, bool>,
//    double>
    virtual StepResult Step(const std::string& agent_type, std::vector<std::shared_ptr<Action>> actions) = 0;
    virtual std::unordered_map<std::string, std::vector<std::deque<std::shared_ptr<State>>>> GetObservations() = 0;
    virtual void Render() = 0;
    virtual bool GetEarlyClosing() = 0;
    virtual void HandleEvents(SDL_Event event, ImGuiIO* io) = 0;
    virtual void GetData(std::string data) = 0;
    virtual void SetRLStatus(pybind11::dict status) = 0;
    virtual bool AgentIsRunning() = 0;
    virtual void UpdateReward() = 0;
    virtual pybind11::dict GetRLStatus() = 0;
    virtual void SetRenderer(SDL_Renderer* renderer) = 0;
    virtual void ImGuiRendering(bool &update_simulation, bool &render_simulation, int &delay, float framerate) = 0;
    virtual std::string GetUserInput() = 0;
    virtual void InitializeMap() = 0;
    virtual bool InitialModeSelectionDone() = 0;
    virtual void CheckReset() = 0;
};



#endif //ROSHAN_MODEL_INTERFACE_H
