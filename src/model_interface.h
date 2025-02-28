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
#include "action.h"
#include <memory>
#include "externals/pybind11/include/pybind11/pybind11.h"

class IModel {
public:
    virtual ~IModel() = default;

    virtual void Update() = 0;
    virtual std::tuple<std::vector<std::deque<std::shared_ptr<State>>>, std::vector<double>, std::vector<bool>, std::pair<bool, bool>, double> Step(std::vector<std::shared_ptr<Action>> actions) = 0;
    virtual std::vector<std::deque<std::shared_ptr<State>>> GetObservations() = 0;
    virtual void Render() = 0;
    virtual bool AgentIsRunning() = 0;
    virtual void HandleEvents(SDL_Event event, ImGuiIO* io) = 0;
    virtual void GetData(std::string data) = 0;
    virtual void SetRLStatus(pybind11::dict status) = 0;
    virtual void UpdateReward() = 0;
    virtual pybind11::dict GetRLStatus() = 0;
    virtual void SetRenderer(std::shared_ptr<SDL_Renderer> renderer) = 0;
    virtual void ImGuiRendering(bool &update_simulation, bool &render_simulation, int &delay, float framerate) = 0;
    virtual std::string GetUserInput() = 0;
    virtual int GetViewRange() = 0;
    virtual int GetTimeSteps() = 0;
    virtual int GetMapSize() = 0;
    virtual bool InitialModeSelectionDone() = 0;
};



#endif //ROSHAN_MODEL_INTERFACE_H
