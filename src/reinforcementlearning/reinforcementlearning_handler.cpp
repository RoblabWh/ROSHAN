//
// Created by nex on 04.06.24.
//

#include "reinforcementlearning_handler.h"

#include <utility>

ReinforcementLearningHandler::ReinforcementLearningHandler(FireModelParameters &parameters) : parameters_(parameters){

    AgentFactory::GetInstance().RegisterAgent("FlyAgent",
                                              [](auto gridmap, auto& parameters, int drone_id, int time_steps) {
                                                  return std::make_shared<FlyAgent>(gridmap, parameters, drone_id, time_steps);
                                              });
//    AgentFactory::GetInstance().RegisterAgent("ExploreAgent",
//                                              [](auto gridmap, auto& parameters, int drone_id, int time_steps) {
//                                                  return std::make_shared<ExploreAgent>(gridmap, parameters, drone_id, time_steps);
//                                              });

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
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    if (mode == Mode::GUI_RL) {
        gridmap_->SetGroundstationRenderer(model_renderer_->GetRenderer());
    }
    auto agent_type = rl_status_["agent_type"].cast<std::string>();
    parameters_.SetAgentType(agent_type);
    total_env_steps_ = parameters_.GetTotalEnvSteps();

    // Create FlyAgents [always present]
    // Calculate the number of FlyAgents based on the chosen Hierarchy TODO currently not implemented
    int num_fly_agents = parameters_.GetNumberOfDrones();

    for (int i = 0; i < num_fly_agents; ++i){
        auto time_steps = rl_status_["flyAgentTimesteps"].cast<int>();
        auto agent = AgentFactory::GetInstance().CreateAgent("FlyAgent", gridmap_, parameters_, i, time_steps);
        auto fly_agent = std::dynamic_pointer_cast<FlyAgent>(agent);
        if (fly_agent == nullptr) {
            std::cerr << "Failed to create FlyAgent\n";
            continue;
        }
        // Initialize FlyAgent TODO make this member function
        double rng_number = dist(gen);
        auto rl_mode = rl_status_["rl_mode"].cast<std::string>();
        fly_agent->Initialize(mode, gridmap_, model_renderer_, rl_mode, rng_number);
        agents_by_type_["FlyAgent"].push_back(fly_agent);
    }

    // Create ExploreAgents [only present in agent_type == "ExploreAgent"] TODO

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

std::tuple<std::unordered_map<std::string, std::vector<std::deque<std::shared_ptr<State>>>>, std::vector<double>, std::vector<bool>, std::vector<bool>, double> ReinforcementLearningHandler::Step(const std::string& agent_type, std::vector<std::shared_ptr<Action>> actions) {
    //TODO return std::unordered_map<std::string, std::vector<double>> instead of vector
    if (gridmap_ == nullptr || agents_by_type_.find(agent_type) == agents_by_type_.end()) {
        std::cerr << "No agents of type " << agent_type << " or invalid GridMap.\n";
        return {};
    }

    total_env_steps_ -= 1;
    parameters_.SetCurrentEnvSteps(parameters_.GetTotalEnvSteps() - total_env_steps_);
    //init bool vector that is size of drones_ TODO make pretty
    std::vector<bool> terminals, agent_died, agent_succeeded;
    std::vector<double> rewards;

    // Step through all the Agents and update their states, then calculate their reward
    auto& agents = agents_by_type_[agent_type];
    auto hierarchy_type = rl_status_["agent_type"].cast<std::string>();

    for (size_t i = 0; i < agents.size(); ++i) {
        agents[i]->ExecuteAction(actions[i], hierarchy_type, gridmap_);


        // Check for Agent Terminal States
        auto terminal_state = agents[i]->GetTerminalStates(eval_mode_, gridmap_, total_env_steps_);
        terminals.push_back(terminal_state[0]);
        agent_died.push_back(terminal_state[1]);
        agent_succeeded.push_back(terminal_state[2]);

        // Confusing Logic to determine Hierarchy Steps TODO possibly make not as confusing
        if (agents[i]->GetPerformedHierarchyAction()) {
            double reward = agents[i]->CalculateReward();
            rewards.push_back(reward);
            // Reset some values for the next step
            agents[i]->StepReset();
        }
    }

    // Build Terminal States
    bool resetEnv = std::any_of(terminals.begin(), terminals.end(), [](bool t){ return t; });
    bool drone_died = std::any_of(agent_died.begin(), agent_died.end(), [](bool d){ return d; });
    bool drone_succeeded = std::any_of(agent_succeeded.begin(), agent_succeeded.end(), [](bool s){ return s; });

    std::vector<bool> terminal_states{resetEnv, drone_died, drone_succeeded};

    return {this->GetObservations(), rewards, terminals, terminal_states, gridmap_->PercentageBurned()};
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
