//
// Created by nex on 04.06.24.
//

#include "rl_handler.h"
#include "externals/pybind11/include/pybind11/numpy.h"

#include <utility>

ReinforcementLearningHandler::ReinforcementLearningHandler(FireModelParameters &parameters) : parameters_(parameters){

    AgentFactory::GetInstance().RegisterAgent("fly_agent",
                                              [](auto& parameters, int total_id, int drone_id, int time_steps) {
                                                  return std::make_shared<FlyAgent>(parameters, total_id, drone_id, time_steps);
                                              });
    AgentFactory::GetInstance().RegisterAgent("explore_agent",
                                              [](auto& parameters, int total_id, int id, int time_steps) {
                                                  return std::make_shared<ExploreAgent>(parameters, total_id, id, time_steps);
                                              });

    AgentFactory::GetInstance().RegisterAgent("planner_agent",
                                              [](auto& parameters, int total_id, int id, int time_steps) {
                                                  return std::make_shared<PlannerAgent>(parameters, total_id, id, time_steps);
                                              });

    total_env_steps_ = 1;
}

std::unordered_map<std::string, std::vector<std::deque<std::shared_ptr<State>>>>
ReinforcementLearningHandler::GetObservations() {
    std::unordered_map<std::string, std::vector<std::deque<std::shared_ptr<State>>>> observations;
    observations.reserve(agents_by_type_.size());

    for (const auto& [key, agents] : agents_by_type_) {
        std::vector<std::deque<std::shared_ptr<State>>> agent_states;
        agent_states.reserve(agents.size());

        for (const auto& agent : agents) {
            // Use ref to avoid copying shared_ptrs, then construct the State deque
            const auto& obs_ref = agent->GetObservationsRef();
            std::deque<std::shared_ptr<State>> state_deque(obs_ref.begin(), obs_ref.end());
            agent_states.push_back(std::move(state_deque));
        }
        observations.emplace(key, std::move(agent_states));
    }

    return observations;
}

py::tuple ReinforcementLearningHandler::GetBatchedFlyObservations(const std::string& agent_type) {
    auto it = agents_by_type_.find(agent_type);
    if (it == agents_by_type_.end() || it->second.empty()) {
        return py::make_tuple(py::none(), py::none(), py::none(), py::none(),
                              py::none(), py::none(), py::none(), py::none());
    }

    const auto& agents = it->second;
    const int N = static_cast<int>(agents.size());

    // Collect AgentState pointers per agent and determine dimensions
    std::vector<std::vector<AgentState*>> agent_states(N);
    int max_T = 0;
    for (int i = 0; i < N; ++i) {
        const auto& obs_ref = agents[i]->GetObservationsRef();
        agent_states[i].reserve(obs_ref.size());
        for (const auto& state_ptr : obs_ref) {
            // GetObservationsRef returns deque<shared_ptr<AgentState>> directly
            agent_states[i].push_back(state_ptr.get());
        }
        max_T = std::max(max_T, static_cast<int>(agent_states[i].size()));
    }

    if (max_T == 0) {
        return py::make_tuple(py::none(), py::none(), py::none(), py::none(),
                              py::none(), py::none(), py::none(), py::none());
    }

    // Determine distance slots from first available state
    int dist_slots = 0;
    int dist_features = 0;
    for (int i = 0; i < N && dist_slots == 0; ++i) {
        if (!agent_states[i].empty()) {
            const auto& dists = agent_states[i][0]->GetDistancesToOtherAgentsRef();
            dist_slots = static_cast<int>(dists.size());
            if (!dists.empty()) dist_features = static_cast<int>(dists[0].size());
        }
    }

    // Allocate numpy arrays — single contiguous allocation per field
    py::array_t<double> ids({N, max_T});
    py::array_t<double> velocities({N, max_T, 2});
    py::array_t<double> delta_goals({N, max_T, 2});
    py::array_t<double> cos_sin({N, max_T, 2});
    py::array_t<double> speed({N, max_T});
    py::array_t<double> dist_to_goal({N, max_T});
    py::array_t<double> dists_others({N, max_T, dist_slots, dist_features});
    // Use uint8 for mask (pybind11 doesn't support bool arrays directly with unchecked)
    py::array_t<bool> dists_mask({N, max_T, dist_slots});

    auto id_buf = ids.mutable_unchecked<2>();
    auto vel_buf = velocities.mutable_unchecked<3>();
    auto dg_buf = delta_goals.mutable_unchecked<3>();
    auto cs_buf = cos_sin.mutable_unchecked<3>();
    auto sp_buf = speed.mutable_unchecked<2>();
    auto dtg_buf = dist_to_goal.mutable_unchecked<2>();
    auto dto_buf = dists_others.mutable_unchecked<4>();
    auto dm_buf = dists_mask.mutable_unchecked<3>();

    // Fill arrays in one pass — all 8 fields extracted per state
    for (int i = 0; i < N; ++i) {
        const int T_i = static_cast<int>(agent_states[i].size());
        for (int t = 0; t < max_T; ++t) {
            if (t < T_i) {
                auto* as = agent_states[i][t];
                auto vel = as->GetVelocityNorm();
                vel_buf(i, t, 0) = vel.first;
                vel_buf(i, t, 1) = vel.second;

                auto dg = as->GetDeltaGoal();
                dg_buf(i, t, 0) = dg.first;
                dg_buf(i, t, 1) = dg.second;

                auto cs_ = as->GetCosSinToGoal();
                cs_buf(i, t, 0) = cs_.first;
                cs_buf(i, t, 1) = cs_.second;

                sp_buf(i, t) = as->GetSpeed();
                dtg_buf(i, t) = as->GetDistanceToGoal();
                id_buf(i, t) = static_cast<double>(as->GetID());

                const auto& dists = as->GetDistancesToOtherAgentsRef();
                const auto& mask = as->GetDistancesMaskRef();
                for (int s = 0; s < dist_slots; ++s) {
                    dm_buf(i, t, s) = (s < static_cast<int>(mask.size())) && mask[s];
                    if (s < static_cast<int>(dists.size())) {
                        const auto& row = dists[s];
                        for (int f = 0; f < dist_features; ++f) {
                            dto_buf(i, t, s, f) = (f < static_cast<int>(row.size())) ? row[f] : 0.0;
                        }
                    } else {
                        for (int f = 0; f < dist_features; ++f) {
                            dto_buf(i, t, s, f) = 0.0;
                        }
                    }
                }
            } else {
                // Pad with zeros for agents with fewer time steps
                vel_buf(i, t, 0) = 0.0; vel_buf(i, t, 1) = 0.0;
                dg_buf(i, t, 0) = 0.0; dg_buf(i, t, 1) = 0.0;
                cs_buf(i, t, 0) = 0.0; cs_buf(i, t, 1) = 0.0;
                sp_buf(i, t) = 0.0;
                dtg_buf(i, t) = 0.0;
                id_buf(i, t) = 0.0;
                for (int s = 0; s < dist_slots; ++s) {
                    dm_buf(i, t, s) = false;
                    for (int f = 0; f < dist_features; ++f) {
                        dto_buf(i, t, s, f) = 0.0;
                    }
                }
            }
        }
    }

    return py::make_tuple(ids, velocities, delta_goals, cos_sin, speed, dist_to_goal, dists_others, dists_mask);
}

py::tuple ReinforcementLearningHandler::GetBatchedPlannerObservations() {
    auto it = agents_by_type_.find("planner_agent");
    if (it == agents_by_type_.end() || it->second.empty()) {
        return py::make_tuple(py::none(), py::none(), py::none());
    }

    const auto& agents = it->second;
    const int N = static_cast<int>(agents.size());

    // Collect AgentState pointers per agent and determine dimensions
    std::vector<std::vector<AgentState*>> agent_states(N);
    int max_T = 0;
    for (int i = 0; i < N; ++i) {
        const auto& obs_ref = agents[i]->GetObservationsRef();
        agent_states[i].reserve(obs_ref.size());
        for (const auto& state_ptr : obs_ref) {
            agent_states[i].push_back(state_ptr.get());
        }
        max_T = std::max(max_T, static_cast<int>(agent_states[i].size()));
    }

    if (max_T == 0) {
        return py::make_tuple(py::none(), py::none(), py::none());
    }

    // Determine num_drones from first available state and max_fires across all states
    int num_drones = 0;
    int max_fires = 0;
    for (int i = 0; i < N; ++i) {
        for (auto* as : agent_states[i]) {
            if (num_drones == 0) {
                num_drones = static_cast<int>(as->GetDronePositionsRef().size());
            }
            max_fires = std::max(max_fires, static_cast<int>(as->GetFirePositionsRef().size()));
        }
    }

    // Allocate numpy arrays
    py::array_t<double> drone_positions({N, max_T, num_drones, 2});
    py::array_t<double> goal_positions({N, max_T, num_drones, 2});
    py::array_t<double> fire_positions({N, max_T, max_fires, 2});

    auto dp_buf = drone_positions.mutable_unchecked<4>();
    auto gp_buf = goal_positions.mutable_unchecked<4>();
    auto fp_buf = fire_positions.mutable_unchecked<4>();

    // Fill arrays in one pass
    for (int i = 0; i < N; ++i) {
        const int T_i = static_cast<int>(agent_states[i].size());
        for (int t = 0; t < max_T; ++t) {
            if (t < T_i) {
                auto* as = agent_states[i][t];

                const auto& drones = as->GetDronePositionsRef();
                for (int d = 0; d < num_drones; ++d) {
                    dp_buf(i, t, d, 0) = drones[d].first;
                    dp_buf(i, t, d, 1) = drones[d].second;
                }

                const auto& goals = as->GetGoalPositionsRef();
                for (int d = 0; d < num_drones; ++d) {
                    gp_buf(i, t, d, 0) = goals[d].first;
                    gp_buf(i, t, d, 1) = goals[d].second;
                }

                const auto& fires = as->GetFirePositionsRef();
                const int n_fires = static_cast<int>(fires.size());
                for (int f = 0; f < max_fires; ++f) {
                    if (f < n_fires) {
                        fp_buf(i, t, f, 0) = fires[f].first;
                        fp_buf(i, t, f, 1) = fires[f].second;
                    } else {
                        fp_buf(i, t, f, 0) = 0.0;
                        fp_buf(i, t, f, 1) = 0.0;
                    }
                }
            } else {
                // Zero-pad shorter timesteps
                for (int d = 0; d < num_drones; ++d) {
                    dp_buf(i, t, d, 0) = 0.0; dp_buf(i, t, d, 1) = 0.0;
                    gp_buf(i, t, d, 0) = 0.0; gp_buf(i, t, d, 1) = 0.0;
                }
                for (int f = 0; f < max_fires; ++f) {
                    fp_buf(i, t, f, 0) = 0.0; fp_buf(i, t, f, 1) = 0.0;
                }
            }
        }
    }

    return py::make_tuple(drone_positions, goal_positions, fire_positions);
}

void ReinforcementLearningHandler::ResetEnvironment(Mode mode) {
    if (mode == Mode::GUI_RL) {
        gridmap_->SetGroundstationRenderer(model_renderer_->GetRenderer());
    }

    StartPlanner start_planner(parameters_, gridmap_);
    GoalSelector goal_selector(gridmap_);

    SpawnConfig spawn_config;
    parameters_.groundstation_start_percentage_ = parameters_.adaptive_start_position_ ? rl_status_["objective"].cast<double>() : parameters_.groundstation_start_percentage_;
    spawn_config.drone_radius_m = parameters_.drone_size_;
    spawn_config.groundstation_start_percentage_ = parameters_.groundstation_start_percentage_;

    GoalConfig goal_config;
    parameters_.fire_goal_percentage_ = parameters_.adaptive_goal_position_ ? 1.0 - rl_status_["objective"].cast<double>() : parameters_.fire_goal_percentage_;
    goal_config.fire_goal_pct = parameters_.fire_goal_percentage_;

    auto hierarchy_type = rl_status_["hierarchy_type"].cast<std::string>();
    rl_status_["env_reset"] = true;
    parameters_.SetHierarchyType(hierarchy_type);
    const auto rl_mode = rl_status_["rl_mode"].cast<std::string>();
    total_env_steps_ = parameters_.GetTotalEnvSteps(rl_mode == "eval");

    // --- Helpers -------------------------------------------------------------

    auto assign_start_and_goal = [&](FlyAgent& fly, int id) {
        auto start_pos = start_planner.assign_start(spawn_config, id);
        fly.SetPosition(start_pos.pos_grid_double);

        auto goal = goal_selector.pick_goal(
                goal_config,
                rl_mode,
                fly.IsExplorerBySubtype(),
                fly.GetGridPositionDouble(),
                parameters_.gen_
        );
        fly.SetGoalPosition(goal);
    };

    // Ensure a vector<T> has exactly count agents of T derived from fly_agent bucket,
    // either resetting existing or recreating from scratch.
    auto ensure_fly_group =
            [&](const std::string& bucket_key,
                int count,
                int id_offset,
                const std::string& agent_type_label,
                double speed,
                int view_range,
                int time_steps) -> std::vector<std::shared_ptr<FlyAgent>>
            {
                auto& bucket = agents_by_type_[bucket_key];

                // Reserve for target size (so we don’t reallocate in loops)
                if (bucket.capacity() < static_cast<size_t>(count)) {
                    bucket.reserve(static_cast<size_t>(count));
                }

                std::vector<std::shared_ptr<FlyAgent>> out; out.reserve(static_cast<size_t>(count));

                const bool sizes_match = (static_cast<int>(bucket.size()) == count);

                if (sizes_match) {
                    // Reset path — single cast pass for setup + reset
                    auto fly_agents = CastAgents<FlyAgent>(bucket);
                    for (auto& fly : fly_agents) {
                        fly->SetAgentSubType(agent_type_label);
                        assign_start_and_goal(*fly, fly->GetTotalId());
                    }
                    findCollisions(fly_agents, gridmap_, view_range, true);
                    for (auto& fly : fly_agents) {
                        fly->Reset(mode, gridmap_, model_renderer_);
                        out.push_back(fly);
                    }
                } else {
                    // Recreate path — push FlyAgent directly, no second CastAgents needed
                    bucket.clear();
                    std::vector<std::shared_ptr<FlyAgent>> fly_agents;
                    fly_agents.reserve(static_cast<size_t>(count));
                    for (int i = 0; i < count; ++i) {
                        const int total_id = id_offset + i;
                        const int agent_id = i;
                        auto agent = AgentFactory::GetInstance().CreateAgent("fly_agent", parameters_, total_id, agent_id, time_steps);
                        auto fly   = std::dynamic_pointer_cast<FlyAgent>(agent);
                        if (!fly) { std::cerr << "Failed to create fly_agent\n"; continue; }

                        fly->SetAgentSubType(agent_type_label);
                        assign_start_and_goal(*fly, total_id);

                        bucket.push_back(fly);
                        fly_agents.push_back(fly);
                    }
                    findCollisions(fly_agents, gridmap_, view_range, true);
                    for (auto& fly : fly_agents) {
                        fly->Initialize(mode, speed, view_range, gridmap_, model_renderer_);
                        out.push_back(fly);
                    }
                }
                return out;
            };

    // Reset or create exactly one manager agent in bucket_key.
    // init_fn is called only on creation. reset_fn only on reset.
    auto ensure_singleton =
            [&](const std::string& bucket_key,
                const std::function<void(std::shared_ptr<Agent>&)>& init_fn,
                const std::function<void(std::shared_ptr<Agent>&)>& reset_fn)
            {
                auto& bucket = agents_by_type_[bucket_key];
                if (bucket.size() == 1) {
                    // Reset existing
                    auto& a = bucket[0];
                    reset_fn(a);
                    return;
                }

                // (Re)create 1
                bucket.clear();
                auto agent = AgentFactory::GetInstance().CreateAgent(bucket_key, parameters_, 0, 0, 1);
                if (!agent) { std::cerr << "Failed to create " << bucket_key << "\n"; return; }
                init_fn(agent);
                bucket.push_back(agent);
            };

    // small helper to clear buckets don't needed in current mode
    auto clear_buckets = [&](std::initializer_list<const char*> keys) {
        for (auto* k : keys) { agents_by_type_[k].clear(); }
    };

    // --- Flows ---------------------------------------------------------------
    // (1) Pure fly_agent mode: We don’t need any manager agents
    // (2) Evaluation of fly_agent with explore_agent: We don’t need planner_agent
    // (3) Hierarchical mode with planner_agent and explore_agent: We don’t need fly_agent
    // (4) only explore_agent: We don’t need planner_agent and fly_agent (debug for now)

    auto pure_fly_agent_mode = parameters_.GetHierarchyType() == "fly_agent" && !parameters_.eval_fly_policy_;
    auto eval_fly_policy_mode = parameters_.GetHierarchyType() == "fly_agent" && parameters_.eval_fly_policy_;
    auto hierarchical_planner_mode = parameters_.GetHierarchyType() == "planner_agent";

    if (pure_fly_agent_mode) {
        const int desired_fly = parameters_.GetNumberOfFlyAgents();
        spawn_config.group_size = desired_fly;

        // Ensure the flat fly_agent group
        (void) ensure_fly_group(
                "fly_agent",
                desired_fly,
                /*id_offset*/ 0,
                /*label*/ "fly_agent",
                parameters_.fly_agent_speed_,
                parameters_.fly_agent_view_range_,
                parameters_.fly_agent_time_steps_
        );

        // Clear others we don’t use in this mode
        clear_buckets({"ExploreFlyAgent","PlannerFlyAgent","explore_agent","planner_agent"});
    }
    else if (eval_fly_policy_mode) {
        spawn_config.group_size = parameters_.GetNumberOfFlyAgents() + parameters_.GetNumberOfExplorers();

        // Ensure the flat fly_agent group
        (void) ensure_fly_group(
                "fly_agent",
                parameters_.GetNumberOfFlyAgents(),
                /*id_offset*/ 0,
                /*label*/ "fly_agent",
                parameters_.fly_agent_speed_,
                parameters_.fly_agent_view_range_,
                parameters_.fly_agent_time_steps_
        );

        // ExploreFlyAgent group
        auto explore_fly_agents = ensure_fly_group(
                "ExploreFlyAgent",
                parameters_.GetNumberOfExplorers(),
                /*id_offset*/ parameters_.GetNumberOfFlyAgents(),
                /*label*/ "ExploreFlyAgent",
                parameters_.explore_agent_speed_,
                parameters_.explore_agent_view_range_,
                parameters_.explore_agent_time_steps_
        );

        // explore_agent singleton
        ensure_singleton(
                "explore_agent",
                // init
                [&](std::shared_ptr<Agent>& a){
                    auto explore = std::dynamic_pointer_cast<ExploreAgent>(a);
                    if (!explore) { std::cerr << "Failed to cast explore_agent\n"; return; }
                    auto flys = CastAgents<FlyAgent>(agents_by_type_["ExploreFlyAgent"]);
                    explore->Initialize(flys, gridmap_);
                },
                // reset
                [&](std::shared_ptr<Agent>& a){
                    auto explore = std::dynamic_pointer_cast<ExploreAgent>(a);
                    if (explore) explore->Reset(mode, gridmap_, model_renderer_);
                }
        );

        // Clear others we don’t use in this mode
        clear_buckets({"PlannerFlyAgent","planner_agent"});
    }
    else {
        // We’re in hierarchical mode: explore + maybe planner
        const int num_explore = parameters_.GetNumberOfExplorers();
        int total_group = num_explore;

        if (hierarchical_planner_mode) {
            total_group += parameters_.GetNumberOfExtinguishers();
        }
        spawn_config.group_size = total_group;

        // ExploreFlyAgent group
        auto explore_fly_agents = ensure_fly_group(
                "ExploreFlyAgent",
                num_explore,
                /*id_offset*/ 0,
                /*label*/ "ExploreFlyAgent",
                parameters_.explore_agent_speed_,
                parameters_.explore_agent_view_range_,
                parameters_.explore_agent_time_steps_
        );

        // explore_agent singleton
        ensure_singleton(
                "explore_agent",
                // init
                [&](std::shared_ptr<Agent>& a){
                    auto explore = std::dynamic_pointer_cast<ExploreAgent>(a);
                    if (!explore) { std::cerr << "Failed to cast explore_agent\n"; return; }
                    auto flys = CastAgents<FlyAgent>(agents_by_type_["ExploreFlyAgent"]);
                    explore->Initialize(flys, gridmap_);
                },
                // reset
                [&](std::shared_ptr<Agent>& a){
                    auto explore = std::dynamic_pointer_cast<ExploreAgent>(a);
                    if (explore) explore->Reset(mode, gridmap_, model_renderer_);
                }
        );

        // If we are in planner mode, ensure the planner buckets
        if (hierarchical_planner_mode) {
            // PlannerFlyAgent group (extinguishers), with ID offset after explorers
            const int num_ext = parameters_.GetNumberOfExtinguishers();
            auto planner_fly_agents = ensure_fly_group(
                    "PlannerFlyAgent",
                    num_ext,
                    /*id_offset*/ num_explore,
                    /*label*/ "PlannerFlyAgent",
                    parameters_.extinguisher_speed_,
                    parameters_.extinguisher_view_range_,
                    parameters_.fly_agent_time_steps_
            );

            // planner_agent singleton
            ensure_singleton(
                    "planner_agent",
                    // init
                    [&](std::shared_ptr<Agent>& a){
                        auto planner = std::dynamic_pointer_cast<PlannerAgent>(a);
                        if (!planner) { std::cerr << "Failed to cast planner_agent\n"; return; }
                        auto explore_mgr = std::dynamic_pointer_cast<ExploreAgent>(agents_by_type_["explore_agent"].front());
                        auto fly_agents = CastAgents<FlyAgent>(agents_by_type_["PlannerFlyAgent"]);
                        planner->Initialize(explore_mgr, fly_agents, gridmap_);
                    },
                    // reset
                    [&](std::shared_ptr<Agent>& a){
                        auto planner = std::dynamic_pointer_cast<PlannerAgent>(a);
                        if (planner) planner->Reset(mode, gridmap_, model_renderer_);
                    }
            );
        } else {
            // Not planner mode: clear planner buckets if they exist
            clear_buckets({"PlannerFlyAgent","planner_agent"});
        }

        // We never keep the flat "fly_agent" bucket in planning hierarchical flows
        clear_buckets({"fly_agent"});
    }
}

void ReinforcementLearningHandler::StepDroneManual(int drone_idx, double speed_x, double speed_y, int water_dispense) {
    // Find fly agents from any of the possible buckets (same logic as GetDrones)
    std::vector<std::shared_ptr<FlyAgent>> fly_agents;

    auto add_fly_agents_from = [&](const std::string& type_key) {
        if (agents_by_type_.find(type_key) != agents_by_type_.end()) {
            for (const auto& agent : agents_by_type_[type_key]) {
                auto fly_agent = std::dynamic_pointer_cast<FlyAgent>(agent);
                if (fly_agent) {
                    fly_agents.push_back(fly_agent);
                }
            }
        }
    };

    add_fly_agents_from("fly_agent");
    add_fly_agents_from("ExploreFlyAgent");
    add_fly_agents_from("PlannerFlyAgent");

    if (fly_agents.empty()) {
        std::cerr << "[StepDroneManual] No fly agents found in any bucket.\n";
        return;
    }

    if (drone_idx < static_cast<int>(fly_agents.size())) {
        auto& agent = fly_agents[drone_idx];
        agent->Step(speed_x, speed_y, gridmap_);
        agent->DispenseWater(gridmap_, water_dispense);
        // Always Update States. Manual control doesn't have frame skipping and will most likely botch training
        // So just use it for Debugging!
        findCollisions(fly_agents, gridmap_);
        agent->UpdateStates(gridmap_);
    } else {
        std::cerr << "[StepDroneManual] drone_idx " << drone_idx << " >= agent count " << fly_agents.size() << std::endl;
    }
}

StepResult ReinforcementLearningHandler::Step(const std::string& agent_type, std::vector<std::shared_ptr<Action>> actions, bool skip_observations) {

    if (gridmap_ == nullptr || agents_by_type_.find(agent_type) == agents_by_type_.end()) {
        std::cerr << "No agents of type " << agent_type << " or invalid GridMap.\n";
        return {};
    }

    StepResult result;

    parameters_.SetCurrentEnvSteps(parameters_.total_env_steps_ - total_env_steps_);

    auto &agents = agents_by_type_[agent_type];
    auto hierarchy_type = parameters_.GetHierarchyType();
    if (agent_type == hierarchy_type) {
        total_env_steps_ -= static_cast<int>(1 * parameters_.hierarchy_time_steps_);
    }

    result.rewards.reserve(agents.size());
    result.terminals.resize(agents.size());
    gridmap_->SetTerminals(false);

    // Step through all the Agents and update their states, then calculate their reward
    for (size_t i = 0; i < agents.size(); ++i) {
        auto &agent = agents[i];
        agent->ExecuteAction(actions[i], hierarchy_type, gridmap_);
        // Increase the frame control for frame skipping
        agent->SetFrameControl(agent->GetFrameCtrl() + 1);

        // Get Terminal State from the Agent
        auto terminal_state = agent->GetTerminalStates(eval_mode_, gridmap_, total_env_steps_);
        result.terminals[i] = terminal_state;
        result.summary.explorers_reached_goal &= terminal_state.kind == TerminationKind::Succeeded;

        if (agent->GetPerformedHierarchyAction()) {
            result.rewards.push_back(agent->CalculateReward(gridmap_));
            // Summary is only relevant for the highest Hierarchy Agent
            // (e.g. PlannerAgent -> Environment Reset doesn't trigger when FlyAgents reach their GoalPos)
            result.summary.env_reset = result.summary.env_reset || terminal_state.is_terminal;
            result.summary.any_failed |= terminal_state.kind == TerminationKind::Failed;
            if (terminal_state.is_terminal) {
                result.summary.reason = terminal_state.reason;
            }
            result.summary.any_succeeded |= terminal_state.kind == TerminationKind::Succeeded;
            agent->StepReset();
        }
    }

    // This should probably go right after the step and the reward should be calculated after collisions are found
    if (agents[0]->GetAgentType() == FLY_AGENT) {
        auto fly_agents = CastAgents<FlyAgent>(agents);
        findCollisions(fly_agents, gridmap_);
    }

    for (const auto &agent : agents) {
        // Update the Agent States for the next observation
        if (result.summary.env_reset || (agent->GetFrameCtrl() % agent->GetFrameSkips() == 0)) agent->UpdateStates(gridmap_);
    }

    // Could now add other metrics here like Extinguished Fires or Explored Percentage
    if (!skip_observations) {
        result.observations = this->GetObservations();
    }
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
    // Find PlannerAgents and set their eval mode
    auto planner_agents = CastAgents<PlannerAgent>(agents_by_type_["planner_agent"]);
    for (const auto& agent : planner_agents) {
        agent->SetEvalMode(eval_mode_);
    }
    this->onUpdateRLStatus();
}

void ReinforcementLearningHandler::UpdateReward() {
    auto intrinsic_reward = rl_status_["intrinsic_reward"].cast<std::vector<double>>();
    // TODO this very ugly decide on how to display rewards in the GUI
    for (const auto& drone: agents_by_type_["fly_agent"]) {
        auto fly = std::dynamic_pointer_cast<FlyAgent>(drone);
        if (fly) fly->ModifyReward(intrinsic_reward[fly->GetId()]);
    }
}
