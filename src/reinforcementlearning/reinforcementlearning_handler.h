//
// Created by nex on 04.06.24.
//

#ifndef ROSHAN_REINFORCEMENTLEARNING_HANDLER_H
#define ROSHAN_REINFORCEMENTLEARNING_HANDLER_H

#include <shared_mutex>
#include <deque>
#include <vector>
#include "externals/pybind11/include/pybind11/embed.h"
#include "state.h"
#include "src/models/firespin/rendering/firemodel_renderer.h"
#include "src/reinforcementlearning/drone_agent/drone.h"
#include "src/reinforcementlearning/drone_agent/drone_state.h"
#include "src/reinforcementlearning/drone_agent/drone_action.h"
#include "src/utils.h"

namespace py = pybind11;

class __attribute__((visibility("default"))) ReinforcementLearningHandler {

public:
    //only one instance of this class can be created
    static std::shared_ptr<ReinforcementLearningHandler> GetInstance(FireModelParameters &parameters) {
        if (instance_ == nullptr) {
            instance_ = std::shared_ptr<ReinforcementLearningHandler>(new ReinforcementLearningHandler(parameters));
        }

        return instance_;
    }

    ~ReinforcementLearningHandler() = default;
    std::vector<std::deque<std::shared_ptr<State>>> GetObservations();
    bool StepDrone(int drone_idx, double speed_x, double speed_y, int water_dispense);
    void ResetDrones(Mode mode);
    void InitFires();
    std::tuple<std::vector<std::deque<std::shared_ptr<State>>>, std::vector<double>, std::vector<bool>, std::pair<bool, bool>, double> Step(std::vector<std::shared_ptr<Action>> actions);

    void SetModelRenderer(std::shared_ptr<FireModelRenderer> model_renderer) { model_renderer_ = model_renderer; }
    void SetGridMap(std::shared_ptr<GridMap> gridmap) { gridmap_ = gridmap; }
    void SetRLStatus(py::dict status) { rl_status_ = status; }
    py::dict GetRLStatus() { return rl_status_; }

    std::shared_ptr<std::vector<std::shared_ptr<DroneAgent>>> GetDrones() { return drones_; }
    CircularBuffer<float> GetRewards() { return rewards_; }
    std::vector<float> GetAllRewards() { return all_rewards_; }

    std::function<void(int)> startFires;
private:
    static std::shared_ptr<ReinforcementLearningHandler> instance_;

    ReinforcementLearningHandler(FireModelParameters &parameters);

    std::shared_ptr<GridMap> gridmap_;
    std::shared_ptr<FireModelRenderer> model_renderer_;
    FireModelParameters& parameters_;
    std::shared_ptr<std::vector<std::shared_ptr<DroneAgent>>> drones_;

    double CalculateReward(bool drone_in_grid, bool fire_extinguished, bool drone_terminal, int out_of_area_counter, int near_fires, double max_distance, double distance_to_fire) const;

    //Dirty Variables CHANGE TO AGENT STATE OR SOMETHING
    double last_distance_to_fire_{};
    int last_near_fires_{};

    // Rewards Collection for Debugging!
    CircularBuffer<float> rewards_;
    std::vector<float> all_rewards_;

    pybind11::dict rl_status_; // Status of the current episode
};


#endif //ROSHAN_REINFORCEMENTLEARNING_HANDLER_H
