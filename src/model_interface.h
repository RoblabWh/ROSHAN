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

class IModel {
public:
    virtual ~IModel() = default;

    virtual void Initialize() = 0;
    virtual void Update() = 0;
    virtual std::tuple<std::vector<std::deque<std::shared_ptr<State>>>, std::vector<double>, std::vector<bool>, std::pair<bool, bool>> Step(std::vector<std::shared_ptr<Action>> actions) = 0;
    virtual std::vector<std::deque<std::shared_ptr<State>>> GetObservations() = 0;
    virtual void Reset() = 0;
    virtual void Render() = 0;
    virtual bool AgentIsRunning() = 0;
    virtual void SetWidthHeight(int width, int height) = 0;
    virtual void HandleEvents(SDL_Event event, ImGuiIO* io) = 0;
    virtual void ImGuiSimulationSpeed() = 0;
    virtual void GetData(std::string data) = 0;
    virtual void ImGuiRendering(std::function<void(bool&, bool&, int&)> controls, bool &update_simulation, bool &render_simulation, int &delay) = 0;
    virtual std::string GetUserInput() = 0;
};



#endif //ROSHAN_MODEL_INTERFACE_H
