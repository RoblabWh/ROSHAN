//
// Created by nex on 04.06.24.
//

#include "reinforcementlearning_handler.h"

#include <utility>

std::shared_ptr<ReinforcementLearningHandler> ReinforcementLearningHandler::instance_ = nullptr;

ReinforcementLearningHandler::ReinforcementLearningHandler(FireModelParameters &parameters) : parameters_(parameters){
    drones_ = std::make_shared<std::vector<std::shared_ptr<DroneAgent>>>();
    agent_is_running_ = false;
    total_env_steps_ = parameters_.GetTotalEnvSteps();
}

std::vector<std::deque<std::shared_ptr<State>>> ReinforcementLearningHandler::GetObservations() {
    std::vector<std::deque<std::shared_ptr<State>>> all_drone_states;
    all_drone_states.reserve(drones_->size());
    if (gridmap_ != nullptr) {
        //Get observations
        for (auto &drone : *drones_) {
            std::deque<DroneState> drone_states = drone->GetStates();
            std::deque<std::shared_ptr<State>> shared_states;
            for (auto &state : drone_states) {
                shared_states.push_back(std::make_shared<DroneState>(state));
            }
            all_drone_states.emplace_back(shared_states);
        }

        return all_drone_states;
    }
    return {};
}

void ReinforcementLearningHandler::ResetEnvironment(Mode mode) {
    drones_->clear();
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    if (mode == Mode::GUI_RL) {
        gridmap_->SetGroundstationRenderer(model_renderer_->GetRenderer());
    }
    total_env_steps_ = parameters_.GetTotalEnvSteps();
    for (int i = 0; i < parameters_.GetNumberOfDrones(); ++i) {
        auto newDrone = std::make_shared<DroneAgent>(gridmap_, rl_status_["agent_type"].cast<std::string>(), parameters_, i);
        if (mode == Mode::GUI_RL) {
            newDrone->SetRenderer(model_renderer_->GetRenderer());
            newDrone->SetRenderer2(model_renderer_->GetRenderer());
        }
        gridmap_->UpdateExploredAreaFromDrone(newDrone);
        newDrone->SetExploreDifference(0);

        // Generate random number between 0 and 1
        double rng_number = dist(gen);
        std::pair<double, double> goal_pos = std::pair<double, double>(-1, -1);
        auto rl_mode = rl_status_["rl_mode"].cast<std::string>();
        if (rng_number < parameters_.fire_goal_percentage_ || rl_mode == "eval") {
            goal_pos = gridmap_->GetNextFire(newDrone);
        }
        if (std::pair<double, double>(-1, -1) == goal_pos) {
            goal_pos = gridmap_->GetGroundstation()->GetGridPositionDouble();
        }

        newDrone->SetGoalPosition(goal_pos);
        newDrone->Initialize(gridmap_);
        newDrone->SetLastDistanceToGoal(newDrone->GetDistanceToGoal());
        newDrone->SetLastNearFires(newDrone->DroneSeesFire());
        newDrone->SetLastDistanceToFire(newDrone->FindNearestFireDistance());
        drones_->push_back(newDrone);
    }
}

void ReinforcementLearningHandler::StepDroneManual(int drone_idx, double speed_x, double speed_y, int water_dispense) {
    if (drone_idx < drones_->size()) {
        auto drone = drones_->at(drone_idx);
        drone->Step(speed_x, speed_y, gridmap_);
        drone->DispenseWater(gridmap_, water_dispense);
        auto terminal_state = drone->IsTerminal(eval_mode_, gridmap_, total_env_steps_);
        auto nothing = drone->CalculateReward(terminal_state.first, total_env_steps_);
    }
}

void ReinforcementLearningHandler::InitFires() const {
        this->startFires(parameters_.fire_percentage_);
}

std::tuple<std::vector<std::deque<std::shared_ptr<State>>>, std::vector<double>, std::vector<bool>, std::pair<bool, bool>, double> ReinforcementLearningHandler::Step(std::vector<std::shared_ptr<Action>> actions) {

    if (gridmap_ != nullptr) {
        total_env_steps_ -= 1;
        parameters_.SetCurrentEnvSteps(parameters_.GetTotalEnvSteps() - total_env_steps_);
        std::vector<bool> terminals;
        std::vector<bool> agent_died;
//        std::vector<bool> goal_reached;
        std::vector<double> rewards;
        gridmap_->UpdateCellDiminishing();
        // First Step through all the drones and update their states, then calculate their reward
        for (int i = 0; i < (*drones_).size(); ++i) {
            actions[i]->Apply(drones_->at(i), gridmap_);
            drones_->at(i)->PolicyStep(gridmap_);
            auto terminal_state = drones_->at(i)->IsTerminal(eval_mode_, gridmap_, total_env_steps_);
            terminals.push_back(terminal_state.first);
            agent_died.push_back(terminal_state.second);
            double reward = drones_->at(i)->CalculateReward(terminal_state.first, total_env_steps_);
            drones_->at(i)->SetReward(reward);
            rewards.push_back(reward);
        }

        // Check if any element in terminals is true, if so some agent reached a terminal state
        bool resetEnv = std::any_of(terminals.begin(), terminals.end(),
                                    [](bool terminal) {return terminal;});
        bool drone_died = std::any_of(agent_died.begin(), agent_died.end(),
                                      [](bool died) {return died;});

        return {this->GetObservations(), rewards, terminals, std::make_pair(resetEnv, drone_died), gridmap_->PercentageBurned()};
    }
    std::cout << "Taking a step into nothingness..." << std::endl;
    return {};
}

void ReinforcementLearningHandler::SetRLStatus(py::dict status) {
    rl_status_ = std::move(status);
    auto rl_mode = rl_status_[py::str("rl_mode")].cast<std::string>();
    eval_mode_ = rl_mode == "eval";
}
