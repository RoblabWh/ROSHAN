//
// Created by nex on 04.06.24.
//

#include "reinforcementlearning_handler.h"

#include <utility>

ReinforcementLearningHandler::ReinforcementLearningHandler(FireModelParameters &parameters) : parameters_(parameters){

    AgentFactory::GetInstance().RegisterAgent("FlyAgent",
                                              [](auto& parameters, int drone_id, int time_steps) {
                                                  return std::make_shared<FlyAgent>(parameters, drone_id, time_steps);
                                              });
    AgentFactory::GetInstance().RegisterAgent("ExploreAgent",
                                              [](auto& parameters, int drone_id, int time_steps) {
                                                  return std::make_shared<ExploreAgent>(parameters, drone_id, time_steps);
                                              });

    agent_is_running_ = false;
    total_env_steps_ = parameters_.GetTotalEnvSteps();
}

std::unordered_map<std::string, std::vector<std::deque<std::shared_ptr<State>>>>
ReinforcementLearningHandler::GetObservations() {
    std::unordered_map<std::string, std::vector<std::deque<std::shared_ptr<State>>>> observations;
    observations.reserve(agents_by_type_.size());

    for (const auto& [key, agents] : agents_by_type_) {
        std::vector<std::deque<std::shared_ptr<State>>> agent_states;
        agent_states.reserve(agents.size());

        for (const auto& agent : agents) {
            agent_states.push_back(agent->GetObservations());
        }

        observations.emplace(key, std::move(agent_states));
    }

    return observations;
}

void ReinforcementLearningHandler::ResetEnvironment(Mode mode) {
    agents_by_type_.clear();

    if (mode == Mode::GUI_RL) {
        gridmap_->SetGroundstationRenderer(model_renderer_->GetRenderer());
    }
    auto hierarchy_type = rl_status_["hierarchy_type"].cast<std::string>();
    rl_status_["env_reset"] = true;
    parameters_.SetHierarchyType(hierarchy_type);
    total_env_steps_ = parameters_.GetTotalEnvSteps();

    // Create FlyAgents [always present]
    // Calculate the number of FlyAgents based on the chosen Hierarchy TODO currently not implemented
    int num_fly_agents = parameters_.GetNumberOfDrones();

    for (int i = 0; i < num_fly_agents; ++i){
        auto time_steps = rl_status_["flyAgentTimesteps"].cast<int>();
        auto frame_skips = rl_status_["frame_skips"].cast<int>();
        auto rl_mode = rl_status_["rl_mode"].cast<std::string>();
        auto agent = AgentFactory::GetInstance().CreateAgent("FlyAgent", parameters_, i, time_steps);
        auto fly_agent = std::dynamic_pointer_cast<FlyAgent>(agent);
        if (fly_agent == nullptr) {
            std::cerr << "Failed to create FlyAgent\n";
            continue;
        }
        // Initialize FlyAgent
        fly_agent->Initialize(mode, frame_skips, gridmap_, model_renderer_, rl_mode);
        agents_by_type_["FlyAgent"].push_back(fly_agent);
    }

    // Create ExploreAgents [only present in agent_type == "ExploreAgent"] TODO
    if (parameters_.GetHierarchyType() == "ExploreAgent") {
        auto time_steps = rl_status_["exploreAgentTimesteps"].cast<int>();
        auto agent = AgentFactory::GetInstance().CreateAgent("ExploreAgent", parameters_, 0, time_steps);
        auto explore_agent = std::dynamic_pointer_cast<ExploreAgent>(agent);
        if (explore_agent == nullptr) {
            std::cerr << "Failed to create ExploreAgent\n";
            return;
        }
        // Initialize ExploreAgent
        auto fly_agents = CastAgents<FlyAgent>(agents_by_type_["FlyAgent"]);
        explore_agent->Initialize(fly_agents, gridmap_, rl_status_["rl_mode"].cast<std::string>());
        agents_by_type_["ExploreAgent"].push_back(explore_agent);
    }

}

void ReinforcementLearningHandler::StepDroneManual(int drone_idx, double speed_x, double speed_y, int water_dispense) {
    if (drone_idx < agents_by_type_["FlyAgent"].size()) {
        auto drone = std::dynamic_pointer_cast<FlyAgent>(agents_by_type_["FlyAgent"][drone_idx]);
        drone->Step(speed_x, speed_y, gridmap_);
        drone->DispenseWater(gridmap_, water_dispense);
        auto terminal_state = drone->GetTerminalStates(eval_mode_, gridmap_, total_env_steps_);
        auto nothing = drone->CalculateReward();
    }
}

// TODO why is this in this class?
void ReinforcementLearningHandler::InitFires() const {
        this->startFires(parameters_.fire_percentage_);
}

void ReinforcementLearningHandler::SimStep(std::vector<std::shared_ptr<Action>> actions){
    if (gridmap_ == nullptr || agents_by_type_.find("FlyAgent") == agents_by_type_.end()) {
        std::cerr << "No agents of type FlyAgent or invalid GridMap.\n";
    }

    auto& agents = agents_by_type_["FlyAgent"];
    auto hierarchy_type = "Stepper";

    for (size_t i = 0; i < agents.size(); ++i) {
        agents[i]->ExecuteAction(actions[i], hierarchy_type, gridmap_);
    }
}

std::tuple<
std::unordered_map<std::string, std::vector<std::deque<std::shared_ptr<State>>>>,
std::vector<double>,
std::vector<bool>,
std::unordered_map<std::string, bool>,
double> ReinforcementLearningHandler::Step(const std::string& agent_type, std::vector<std::shared_ptr<Action>> actions) {
    //TODO return std::unordered_map<std::string, std::vector<double>> instead of vector
    if (gridmap_ == nullptr || agents_by_type_.find(agent_type) == agents_by_type_.end()) {
        std::cerr << "No agents of type " << agent_type << " or invalid GridMap.\n";
        return {};
    }

    total_env_steps_ -= 1;
    parameters_.SetCurrentEnvSteps(parameters_.GetTotalEnvSteps() - total_env_steps_);
    //init bool vector that is size of drones_
    std::vector<bool> terminals, something_failed, something_succeeded;
    std::vector<double> rewards;
    std::unordered_map<std::string, bool> agent_terminal_states;
    agent_terminal_states["EnvReset"] = false;

    // Step through all the Agents and update their states, then calculate their reward
    auto& agents = agents_by_type_[agent_type];
    auto hierarchy_type = parameters_.GetHierarchyType();

    for (size_t i = 0; i < agents.size(); ++i) {
        agents[i]->ExecuteAction(actions[i], hierarchy_type, gridmap_);

        // Check for Agent Terminal States
        auto terminal_state = agents[i]->GetTerminalStates(eval_mode_, gridmap_, total_env_steps_);
        terminals.push_back(terminal_state[0]);
        something_failed.push_back(terminal_state[1]);
        something_succeeded.push_back(terminal_state[2]);

        // Has the current agent performed a Hierarchy Action?
        if (agents[i]->GetPerformedHierarchyAction()) {
            double reward = agents[i]->CalculateReward();
            rewards.push_back(reward);
            // Should the Environment Reset
            agent_terminal_states["EnvReset"] = terminal_state[0];
            // Reset some values for the next step
            agents[i]->StepReset();
        }
    }

    // Build Terminal States
    bool one_agent_died = std::any_of(something_failed.begin(), something_failed.end(), [](bool d){ return d; });
    bool one_agent_succeeded = std::any_of(something_succeeded.begin(), something_succeeded.end(), [](bool s){ return s; });
    bool all_agents_died = std::all_of(something_failed.begin(), something_failed.end(), [](bool d){ return d; });
    bool all_agents_succeeded = std::all_of(something_succeeded.begin(), something_succeeded.end(), [](bool s){ return s; });

    agent_terminal_states["OneAgentDied"] = one_agent_died;
    agent_terminal_states["OneAgentSucceeded"] = one_agent_succeeded;
    agent_terminal_states["AllAgentsDied"] = all_agents_died;
    agent_terminal_states["AllAgentsSucceeded"] = all_agents_succeeded;

    return {this->GetObservations(), rewards, terminals, agent_terminal_states, gridmap_->PercentageBurned()};
}

void ReinforcementLearningHandler::SetRLStatus(py::dict status) {
    rl_status_ = std::move(status);
    auto rl_mode = rl_status_[py::str("rl_mode")].cast<std::string>();
    eval_mode_ = rl_mode == "eval";
}

void ReinforcementLearningHandler::UpdateReward() {
    auto intrinsic_reward = rl_status_["intrinsic_reward"].cast<std::vector<double>>();
    // TODO this very ugly decide on how to display rewards in the GUI
    for (const auto& drone: agents_by_type_["FlyAgent"]) {
        std::dynamic_pointer_cast<FlyAgent>(drone)->ModifyReward(intrinsic_reward[std::dynamic_pointer_cast<FlyAgent>(drone)->GetId()]);
    }
}
