//
// Created by nex on 04.06.24.
//

#ifndef ROSHAN_REINFORCEMENTLEARNING_HANDLER_H
#define ROSHAN_REINFORCEMENTLEARNING_HANDLER_H

#include <shared_mutex>
#include <deque>
#include <utility>
#include <vector>
#include "externals/pybind11/include/pybind11/embed.h"
#include "externals/pybind11/include/pybind11/stl.h"
#include "state.h"
#include "firespin/rendering/firemodel_renderer.h"
#include "agents/agent_factory.h"
#include "src/reinforcementlearning/agents/agent_state.h"
#include "reinforcementlearning/actions/fly_action.h"
#include "src/utils.h"

namespace py = pybind11;

class __attribute__((visibility("default"))) ReinforcementLearningHandler {

public:
    explicit ReinforcementLearningHandler(FireModelParameters &parameters);
    //only one instance of this class can be created

    static std::shared_ptr<ReinforcementLearningHandler> Create(FireModelParameters &parameters) {
        return std::make_shared<ReinforcementLearningHandler>(parameters);
    }

    ~ReinforcementLearningHandler(){
        if (rl_status_) {
            rl_status_.attr("clear")();
        }
    }

    std::unordered_map<std::string, std::vector<std::deque<std::shared_ptr<State>>>> GetObservations();
    void StepDroneManual(int drone_idx, double speed_x, double speed_y, int water_dispense);
    void ResetEnvironment(Mode mode);
    void InitFires() const;
    std::tuple<std::unordered_map<std::string, std::vector<std::deque<std::shared_ptr<State>>>>,
            std::vector<double>,
            std::vector<bool>,
            std::unordered_map<std::string, bool>,
            double> Step(const std::string& agent_type, std::vector<std::shared_ptr<Action>> actions);
    void SetModelRenderer(std::shared_ptr<FireModelRenderer> model_renderer) { model_renderer_ = std::move(model_renderer); }
    void SetGridMap(std::shared_ptr<GridMap> gridmap) { gridmap_ = std::move(gridmap); }
    void SetRLStatus(py::dict status);
    void UpdateReward();
    py::dict GetRLStatus() { return rl_status_; }
    std::shared_ptr<std::vector<std::shared_ptr<FlyAgent>>> GetDrones() {
        if (agents_by_type_.find("FlyAgent") == agents_by_type_.end()) {
            return std::make_shared<std::vector<std::shared_ptr<FlyAgent>>>();
        }
        auto& agents = agents_by_type_["FlyAgent"];
        auto fly_agents = std::make_shared<std::vector<std::shared_ptr<FlyAgent>>>();

        for(const auto& agent : agents) {
            auto fly_agent = std::dynamic_pointer_cast<FlyAgent>(agent);
            if (fly_agent){
                fly_agents->push_back(std::shared_ptr<FlyAgent>(fly_agent));
            } else {
                std::cerr << "Non-FlyAgent is not a FlyAgent!\n";
            }
        }
        return fly_agents;
    }
    std::function<void(float)> startFires;
private:
    std::shared_ptr<GridMap> gridmap_;
    std::shared_ptr<FireModelRenderer> model_renderer_;
    FireModelParameters& parameters_;
    std::unordered_map<std::string, std::vector<std::shared_ptr<Agent>>> agents_by_type_;

    //Flags
    bool agent_is_running_;
    bool eval_mode_ = false;

    // Rewards Collection for Debugging!
    int total_env_steps_;

    pybind11::dict rl_status_; // Status of the current episode
};


#endif //ROSHAN_REINFORCEMENTLEARNING_HANDLER_H
