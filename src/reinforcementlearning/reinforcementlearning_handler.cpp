//
// Created by nex on 04.06.24.
//

#include "reinforcementlearning_handler.h"

#include <utility>

ReinforcementLearningHandler::ReinforcementLearningHandler(FireModelParameters &parameters) : parameters_(parameters){

    AgentFactory::GetInstance().RegisterAgent("fly_agent",
                                              [](auto& parameters, int drone_id, int time_steps) {
                                                  return std::make_shared<FlyAgent>(parameters, drone_id, time_steps);
                                              });
    AgentFactory::GetInstance().RegisterAgent("explore_agent",
                                              [](auto& parameters, int id, int time_steps) {
                                                  return std::make_shared<ExploreAgent>(parameters, id, time_steps);
                                              });

    AgentFactory::GetInstance().RegisterAgent("planner_agent",
                                              [](auto& parameters, int id, int time_steps) {
                                                  return std::make_shared<PlannerAgent>(parameters, id, time_steps);
                                              });

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
            const bool no_skip = last_tick_was_terminal_ || (agent->GetFrameCtrl() % agent->GetFrameSkips() == 0);
            if (no_skip) agent->UpdateStates(gridmap_);
            agent_states.push_back(agent->GetObservations());
        }
        observations.emplace(key, std::move(agent_states));
    }

    return observations;
}

void ReinforcementLearningHandler::ResetEnvironment(Mode mode) {
    if (mode == Mode::GUI_RL) {
        gridmap_->SetGroundstationRenderer(model_renderer_->GetRenderer());
    }
    auto hierarchy_type = rl_status_["hierarchy_type"].cast<std::string>();
    rl_status_["env_reset"] = true;
    parameters_.SetHierarchyType(hierarchy_type);
    total_env_steps_ = parameters_.GetTotalEnvSteps();
    auto rl_mode = rl_status_["rl_mode"].cast<std::string>();

    if (parameters_.GetHierarchyType() == "fly_agent") {
        int desired_agents = rl_mode == "eval" ? 1 : parameters_.GetNumberOfFlyAgents();
        auto &agents = agents_by_type_["fly_agent"];
        if (agents.capacity() < parameters_.GetNumberOfFlyAgents()) {
            agents.reserve(parameters_.GetNumberOfFlyAgents());
        }
        // Reset existing agents if the number matches, otherwise create them
        if (agents.size() == desired_agents) {
            for (auto &agent : agents) {
                auto fly_agent = std::dynamic_pointer_cast<FlyAgent>(agent);
                if (fly_agent) {
                    fly_agent->SetAgentType("fly_agent");
                    fly_agent->Reset(mode, gridmap_, model_renderer_, rl_mode);
                }
            }
        }
        else {
            agents.clear();
            for (int i = 0; i < desired_agents; ++i) {
                auto time_steps = parameters_.fly_agent_time_steps_;
                auto agent = AgentFactory::GetInstance().CreateAgent("fly_agent", parameters_, i, time_steps);
                auto fly_agent = std::dynamic_pointer_cast<FlyAgent>(agent);
                if (!fly_agent) { std::cerr << "Failed to create fly_agent\n"; continue; }
                auto agent_speed = parameters_.fly_agent_speed_;
                auto view_range = parameters_.fly_agent_view_range_;
                fly_agent->SetAgentType("fly_agent");
                fly_agent->Initialize(mode, agent_speed, view_range, gridmap_, model_renderer_, rl_mode);
                agents.push_back(fly_agent);
            }
        }
        // clear other agent types
        agents_by_type_["ExploreFlyAgent"].clear();
        agents_by_type_["PlannerFlyAgent"].clear();
        agents_by_type_["explore_agent"].clear();
        agents_by_type_["planner_agent"].clear();
    }
    else {
        // If the hierarchy type is not FlyAgent, we create ExploreFlyAgents and an explore_agent first
        int num_explore_agents = parameters_.GetNumberOfExplorers();
        auto &explore_fly_agents = agents_by_type_["ExploreFlyAgent"];
        if (explore_fly_agents.capacity() < num_explore_agents) {
            explore_fly_agents.reserve(num_explore_agents);
        }
        // Same logic as above, reset if the number matches, otherwise create them
        if (explore_fly_agents.size() == num_explore_agents) {
            for (auto &agent : explore_fly_agents) {
                auto fly_agent = std::dynamic_pointer_cast<FlyAgent>(agent);
                if (fly_agent) {
                    fly_agent->SetAgentType("ExploreFlyAgent");
                    fly_agent->Reset(mode, gridmap_, model_renderer_, rl_mode);
                }
            }
        }
        // Create ExploreFlyAgents
        else {
            explore_fly_agents.clear();
            for (int i = 0; i < num_explore_agents; ++i) {
                auto time_steps = parameters_.fly_agent_time_steps_;
                auto agent = AgentFactory::GetInstance().CreateAgent("fly_agent", parameters_, i, time_steps);
                auto fly_agent = std::dynamic_pointer_cast<FlyAgent>(agent);
                if (!fly_agent) { std::cerr << "Failed to create fly_agent\n"; continue; }
                auto agent_speed = parameters_.explore_agent_speed_;
                auto view_range = parameters_.explore_agent_view_range_;
                fly_agent->SetAgentType("ExploreFlyAgent");
                fly_agent->Initialize(mode, agent_speed, view_range, gridmap_, model_renderer_, rl_mode);
                explore_fly_agents.push_back(fly_agent);
            }
        }
        // Reset or create explore_agent
        auto &explore_agents = agents_by_type_["explore_agent"];
        // Only one explore_agent is allowed, if it exists we reset it
        if (!explore_agents.empty() && explore_agents.size() == 1) {
            auto explore_agent = std::dynamic_pointer_cast<ExploreAgent>(explore_agents[0]);
            if (explore_agent) {
                explore_agent->Reset(mode, gridmap_, model_renderer_, rl_mode);
            }
        }
        // Otherwise we create it
        else {
            explore_agents.clear();
            auto time_steps = parameters_.explore_agent_time_steps_;
            auto agent = AgentFactory::GetInstance().CreateAgent("explore_agent", parameters_, 0, time_steps);
            auto explore_agent = std::dynamic_pointer_cast<ExploreAgent>(agent);
            if (!explore_agent) { std::cerr << "Failed to create explore_agent\n"; return; }
            auto fly_agents = CastAgents<FlyAgent>(explore_fly_agents);
            explore_agent->Initialize(fly_agents, gridmap_, rl_mode);
            explore_agents.push_back(explore_agent);
        }
        // If the hierarchy type is planner_agent, we create PlannerFlyAgents and a planner_agent
        if (parameters_.GetHierarchyType() == "planner_agent") {
            int num_extinguishers = parameters_.GetNumberOfExtinguishers();
            auto &planner_fly_agents = agents_by_type_["PlannerFlyAgent"];
            if (planner_fly_agents.capacity() < num_extinguishers) {
                planner_fly_agents.reserve(num_extinguishers);
            }
            // Reset existing agents if the number matches, otherwise create them
            if (planner_fly_agents.size() == num_extinguishers) {
                for (auto &agent : planner_fly_agents) {
                    auto fly_agent = std::dynamic_pointer_cast<FlyAgent>(agent);
                    if (fly_agent) {
                        fly_agent->SetAgentType("PlannerFlyAgent");
                        fly_agent->Reset(mode, gridmap_, model_renderer_, rl_mode);
                    }
                }
            }
            // Otherwise we create PlannerFlyAgents
            else {
                planner_fly_agents.clear();
                for (int i = 0; i < num_extinguishers; ++i) {
                    auto time_steps = parameters_.fly_agent_time_steps_;
                    auto agent = AgentFactory::GetInstance().CreateAgent("fly_agent", parameters_, i, time_steps);
                    auto fly_agent = std::dynamic_pointer_cast<FlyAgent>(agent);
                    if (!fly_agent) { std::cerr << "Failed to create FlyAgent\n"; continue; }
                    auto agent_speed = parameters_.extinguisher_speed_;
                    auto view_range = parameters_.extinguisher_view_range_;
                    fly_agent->SetAgentType("PlannerFlyAgent");
                    fly_agent->Initialize(mode, agent_speed, view_range, gridmap_, model_renderer_, rl_mode);
                    planner_fly_agents.push_back(fly_agent);
                }
            }
            // Create or Reset planner_agent
            auto &planner_agents = agents_by_type_["planner_agent"];
            // Only one planner_agent is allowed, if it exists we reset it
            if (!planner_agents.empty() && planner_agents.size() == 1) {
                auto planner = std::dynamic_pointer_cast<PlannerAgent>(planner_agents[0]);
                if (planner) {
                    planner->Reset(mode, gridmap_, model_renderer_, rl_mode);
                }
            }
            // Otherwise we create planner_agent
            else {
                planner_agents.clear();
                auto planner_agent = AgentFactory::GetInstance().CreateAgent("planner_agent", parameters_, 0, 1);
                auto planner = std::dynamic_pointer_cast<PlannerAgent>(planner_agent);
                if (!planner) { std::cerr << "Failed to create planner_agent\n"; return; }
                planner->SetGridMap(gridmap_);
                auto explore_agent = std::dynamic_pointer_cast<ExploreAgent>(agents_by_type_["explore_agent"].front());
                auto fly_agents = CastAgents<FlyAgent>(planner_fly_agents);
                planner->Initialize(explore_agent, fly_agents, gridmap_, rl_mode);
                planner_agents.push_back(planner);
            }
        }
        // Clear unused agent types
        else {
            agents_by_type_["PlannerFlyAgent"].clear();
            agents_by_type_["planner_agent"].clear();
        }
        agents_by_type_["fly_agent"].clear();
    }
}

void ReinforcementLearningHandler::StepDroneManual(int drone_idx, double speed_x, double speed_y, int water_dispense) {
    if (drone_idx < agents_by_type_["fly_agent"].size()) {
        auto drone = std::dynamic_pointer_cast<FlyAgent>(agents_by_type_["fly_agent"][drone_idx]);
        drone->Step(speed_x, speed_y, gridmap_);
        drone->DispenseWater(gridmap_, water_dispense);
        auto terminal_state = drone->GetTerminalStates(eval_mode_, gridmap_, total_env_steps_);
        auto nothing = drone->CalculateReward();
    }
}

StepResult ReinforcementLearningHandler::Step(const std::string& agent_type, std::vector<std::shared_ptr<Action>> actions) {

    if (gridmap_ == nullptr || agents_by_type_.find(agent_type) == agents_by_type_.end()) {
        std::cerr << "No agents of type " << agent_type << " or invalid GridMap.\n";
        return {};
    }

    StepResult result;

    total_env_steps_ -= 1;
    parameters_.SetCurrentEnvSteps(parameters_.total_env_steps_ - total_env_steps_);

    auto &agents = agents_by_type_[agent_type];
    auto hierarchy_type = parameters_.GetHierarchyType();

    result.rewards.reserve(agents.size());
    result.terminals.resize(agents.size());

    // Step through all the Agents and update their states, then calculate their reward
    for (size_t i = 0; i < agents.size(); ++i) {
        auto &agent = agents[i];
        agent->ExecuteAction(actions[i], hierarchy_type, gridmap_);
        // Increase the frame control for frame skipping
        agent->SetFrameControl(agent->GetFrameCtrl() + 1);

        // Get Terminal State from the Agent
        auto terminal_state = agent->GetTerminalStates(eval_mode_, gridmap_, total_env_steps_);
        result.terminals[i] = terminal_state;

        if (agent->GetPerformedHierarchyAction()) {
            result.rewards.push_back(agent->CalculateReward());
            // Summary is only relevant for the highest Hierarchy Agent
            // (e.g. PlannerAgent -> Environment Reset doesn't trigger when FlyAgents reach their GoalPos)
            result.summary.env_reset = result.summary.env_reset || terminal_state.is_terminal;
            result.summary.any_failed |= terminal_state.kind == TerminationKind::Failed;
            result.summary.reason = terminal_state.reason;
            result.summary.any_succeeded |= terminal_state.kind == TerminationKind::Succeeded;
            agent->StepReset();
        }
    }

    // Could now add other metrics here like Extinguished Fires or Explored Percentage
    last_tick_was_terminal_ = result.summary.env_reset;
    result.observations = this->GetObservations();
    result.percent_burned = gridmap_->PercentageBurned();
    return result;
}

void ReinforcementLearningHandler::SetRLStatus(py::dict status) {
    rl_status_ = std::move(status);
    auto rl_mode = rl_status_[py::str("rl_mode")].cast<std::string>();
    eval_mode_ = rl_mode == "eval";
    // Find ExplorerFlyAgents and set their render mode
    auto explore_fly_agents = CastAgents<FlyAgent>(agents_by_type_["ExploreFlyAgent"]);
    for (const auto& agent : explore_fly_agents) {
        agent->SetRender(eval_mode_);
    }
    this->onUpdateRLStatus();
}

void ReinforcementLearningHandler::UpdateReward() {
    auto intrinsic_reward = rl_status_["intrinsic_reward"].cast<std::vector<double>>();
    // TODO this very ugly decide on how to display rewards in the GUI
    for (const auto& drone: agents_by_type_["fly_agent"]) {
        std::dynamic_pointer_cast<FlyAgent>(drone)->ModifyReward(intrinsic_reward[std::dynamic_pointer_cast<FlyAgent>(drone)->GetId()]);
    }
}
