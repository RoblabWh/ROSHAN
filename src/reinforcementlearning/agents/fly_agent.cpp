//
// Created by nex on 08.04.25.
//

#include "fly_agent.h"

FlyAgent::FlyAgent(FireModelParameters &parameters, int total_id, int id, int time_steps) :
Agent(parameters, 300){
    total_id_ = total_id;
    id_ = id;
    agent_sub_type_ = "fly_agent";
    agent_type_ = FLY_AGENT;
    time_steps_ = time_steps;
    water_capacity_ = parameters_.GetWaterCapacity();
    frame_skips_ = parameters_.fly_agent_frame_skips_;
    frame_ctrl_ = 0;
    out_of_area_counter_ = 0;
    vel_vector_ = {0.0, 0.0};
}

void FlyAgent::Initialize(int mode,
                          double speed,
                          int view_range,
                          const std::shared_ptr<GridMap>& grid_map,
                          const std::shared_ptr<FireModelRenderer>& model_renderer) {

    if (mode == GUI_RL) {
        auto asset_path = "../assets/ext_drone.png";
        if (agent_sub_type_ != "PlannerFlyAgent") {
            asset_path = "../assets/looker_drone.png";
        }
        if (parameters_.cia_mode_){
            asset_path = "../assets/pidgeon.png";
        }
        this->SetDroneTextureRenderer(model_renderer->GetRenderer(), asset_path);
        this->SetGoalTextureRenderer(model_renderer->GetRenderer());
    }
    if (agent_sub_type_ == "ExploreFlyAgent") {
        is_explorer_ = true;
    }
    if (agent_sub_type_ == "PlannerFlyAgent") {
        is_planner_agent_ = true;
    }
    max_speed_ = std::make_pair(speed, speed);
    max_acceleration_ = std::make_pair(speed / 2, speed / 2);
    view_range_ = view_range;
    norm_scale_ = view_range_ / 2;

    this->last_distance_to_goal_ = this->GetDistanceToGoal();
    this->InitializeFlyAgentStates(grid_map);
}

void FlyAgent::Reset(Mode mode,
                     const std::shared_ptr<GridMap>& grid_map,
                     const std::shared_ptr<FireModelRenderer>& model_renderer) {
    objective_reached_ = false;
    agent_terminal_state_ = false;
    did_hierarchy_step = false;
    collision_occurred_ = false;
    extinguished_fire_ = false;
    planner_commanded_recharge_ = false;
    still_charging_ = false;
    reward_components_.clear();
    num_extinguished_fires_ = 0;
    vel_vector_ = {0.0, 0.0};
    // DONT clear this here, because we must calculate the distances before we call this function!
    // distance_to_other_agents_.clear();
    trail_.clear();
    newly_explored_cells_ = 0;
    frame_ctrl_ = 0;
    out_of_area_counter_ = 0;
    extinguished_last_fire_ = false;
    water_capacity_ = parameters_.GetWaterCapacity();
    agent_states_.clear();
    Initialize(mode, max_speed_.first, view_range_, grid_map, model_renderer);
}

std::shared_ptr<AgentState> FlyAgent::BuildAgentState(const std::shared_ptr<GridMap>& grid_map) {
    auto state = std::make_shared<AgentState>();
    state->SetID(this->GetId());
    state->SetNormScale(norm_scale_);
    state->SetMaxSpeed(max_speed_);
    state->SetVelocity(this->vel_vector_);
    state->SetPosition(position_);
    state->SetGoalPosition(goal_position_);
    state->SetCellSize(parameters_.GetCellSize());
    state->SetDistancesToOtherAgents(std::make_shared<std::vector<std::vector<double>>>(distance_to_other_agents_));
    state->SetDistancesMask(std::make_shared<std::vector<bool>>(distance_mask_));
//    state->SetDroneView(grid_map->GetDroneView(GetGridPosition(), GetViewRange()));
//    state->SetTotalDroneView(grid_map->GetInterpolatedDroneView(GetGridPosition(), GetViewRange()));
//    state->SetExplorationMap(grid_map->GetExploredMap());
//    state->SetFireMap(grid_map->GetFireMap());
//    state->SetMapDimensions({grid_map->GetRows(), grid_map->GetCols()});
    return state;
}

void FlyAgent::InitializeFlyAgentStates(const std::shared_ptr<GridMap>& grid_map) {
    for(int i = 0; i < time_steps_; ++i) {
        agent_states_.push_front(BuildAgentState(grid_map));
    }
}

void FlyAgent::PerformFly(FlyAction* action, const std::string& hierarchy_type, const std::shared_ptr<GridMap>& gridMap) {

    this->Step(action->GetSpeedX(), action->GetSpeedY(), gridMap);
    // Four different modes for low level agents
    // (1) fly_agent with complex policy: extinguish fires until none are left (no_explorers should be used here)
    // (2) fly_agent with simple policy: just reach the goal (no_explorers should be used here)
    // (3) fly_agent EvaluationPolicy: Behaviour described by the FlyPolicy function (no_explorers should be used here)
    // (4) explore_agents and planner_agents fly_agents: just reach the goal (the explore agents don't set hierarchy steps and don't care about selected policies)
    // (hierarchy_type == "fly_agent" && !parameters_.use_simple_policy_ && !is_explorer_);
    // (hierarchy_type == "fly_agent" && parameters_.use_simple_policy_ && !is_explorer_);
    // (hierarchy_type == "fly_agent" && parameters_.eval_fly_policy_ && !is_explorer_);

    if (hierarchy_type == "fly_agent" && !parameters_.use_simple_policy_ && !is_explorer_ && !parameters_.eval_fly_policy_) {
        if (almostEqual(this->GetGoalPosition(), this->GetGridPositionDouble())) {
            this->DispenseWaterCertain(gridMap);
            gridMap->RemoveReservation(this->GetGoalPositionInt());
            this->SetGoalPosition(gridMap->GetNextFire(this->GetGridPositionDouble()));
        }
        did_hierarchy_step = true;
    } else if (hierarchy_type == "fly_agent" && parameters_.use_simple_policy_ && !is_explorer_ && !parameters_.eval_fly_policy_) {
        if (almostEqual(this->GetGoalPosition(), this->GetGridPositionDouble())) {
            objective_reached_ = true;
        }
        did_hierarchy_step = true;
    } else if (hierarchy_type == "fly_agent" && parameters_.eval_fly_policy_ && !is_explorer_) {
        FlyPolicy(gridMap);
        did_hierarchy_step = true;
    } else {
        if (almostEqual(this->GetGoalPosition(), this->GetGridPositionDouble())) {
            objective_reached_ = true;
        }
        if (is_planner_agent_) {
            this->DispenseWaterCertain(gridMap);
            if (planner_commanded_recharge_){
                if(this->GetGoalPositionInt() == this->GetGridPosition()){
                    if (this->water_capacity_ < parameters_.GetWaterCapacity()) {
                        this->water_capacity_ += parameters_.GetWaterRefillDt();
                    } else {
                        planner_commanded_recharge_ = false;
                        still_charging_ = false;
                    }
                }
            }
        }
    }
}

double FlyAgent::CalculateReward(const std::shared_ptr<GridMap>& grid_map) {
    double distance_to_goal = GetDistanceToGoal();
    double delta_distance = last_distance_to_goal_ - distance_to_goal;
    bool drone_in_grid = GetDroneInGrid();

    std::unordered_map<std::string, double> reward_components;
    double total_reward = 0;

    // Terminals get computed earlier, they set the Flag in gridmap to true in the "ComplexPolicy"
    // that also sets the current objective to true, current objectives get set to true in PerformAction(simplePolicy) OR GetTerminals(ComplexPolicy) noodle code wtf
    // Terminal Reward should go to all agents when the objective is reached (For the simple policy that does not apply here, but for the complex case it does)
    if (objective_reached_ || (grid_map->GetTerminalOccured())) {
        reward_components["GoalReached"] = parameters_.FlyGoalReached_;
    }

    if (!drone_in_grid && agent_terminal_state_) {
        reward_components["BoundaryTerminal"] = parameters_.FlyBoundaryTerminal_;
    }

    if (agent_terminal_state_ && (env_steps_remaining_ <= 0)) {
        reward_components["TimeOut"] = parameters_.FlyTimeOut_;
    }

    // Use the extinguishing code only in the complex policy since the simple one only cares about reaching the goal
    if (extinguished_fire_ && !parameters_.use_simple_policy_) {
        //auto bonus_log = 0.05 * std::log(double(grid_map->GetNRefFires() + 1) / double(grid_map->GetNumBurningCells() + 1));
        reward_components["Extinguish"] = parameters_.FlyExtinguish_; // + bonus_log;
    }

    if (collision_occurred_ && parameters_.fly_agent_collision_) {
        reward_components["Collision"] = parameters_.FlyCollision_;
    }

    if (delta_distance > 0) {
//        auto safety_factor = std::tanh(10 * distance_to_boundary);
        reward_components["DistanceImprovement"] = parameters_.FlyDistanceImprovement_ * delta_distance; // * safety_factor;
    }

    if (parameters_.fly_agent_collision_){
        auto distances_to_objects = this->GetLastState().GetDistancesToOtherAgents();
        for (auto & dist : distances_to_objects) {
            double dx = std::abs(dist[0]);
            double dy = std::abs(dist[1]);
            if (dx < 0.1) {
                reward_components["ProximityPenalty"] += parameters_.FlyProximityPenalty_ * dx;
            }
            if (dy < 0.1) {
                reward_components["ProximityPenalty"] += parameters_.FlyProximityPenalty_ * dy;
            }
        }
    }

    total_reward = ComputeTotalReward(reward_components);
    LogRewards(reward_components);
    last_distance_to_goal_ = distance_to_goal;
    reward_components_ = reward_components;
    this->SetReward(total_reward);
    return total_reward;
}

AgentTerminal FlyAgent::GetTerminalStates(bool eval_mode, const std::shared_ptr<GridMap>& grid_map, int env_steps_remaining) {
    std::vector<bool> terminal_states;
    // bool eval = eval_mode && parameters_.extinguish_all_fires_;

    AgentTerminal t;

    if(!is_explorer_){
        // If the agent has flown out of the grid it has reached a terminal state and died
        if (GetOutOfAreaCounter() > 1 && !parameters_.eval_fly_policy_) {
            t.is_terminal = true;
            t.reason = FailureReason::BoundaryExit;
        }

        if (collision_occurred_ && parameters_.fly_agent_collision_ && !parameters_.eval_fly_policy_) {
            t.is_terminal = true;
            t.reason = FailureReason::Collision;
        }

        // If the agent has taken too long it has reached a terminal state and died
        if (env_steps_remaining <= 0) {
            t.is_terminal = true;
            t.reason = FailureReason::Timeout;
        }
    }

    // If the drone has reached the goal and it is not in evaluation mode
    // the goal is reached because the fly agent is trained that way
    if (!parameters_.use_simple_policy_ && !is_explorer_){
        if (extinguished_last_fire_) {
            t.is_terminal = true;
            objective_reached_ = true;
            grid_map->SetTerminals(true);
        }

        if (!grid_map->IsBurning()) {
            t.is_terminal = true;
        }
    } else {
        if (objective_reached_) {
            t.is_terminal = true;
        }
        if (parameters_.eval_fly_policy_ && !grid_map->HasBurningFires()){
            t.is_terminal = true;
        }
    }

    if (t.is_terminal && t.reason != FailureReason::None) { t.kind = TerminationKind::Failed; }
    else if (t.is_terminal) { t.kind = TerminationKind::Succeeded; }
    else { t.kind = TerminationKind::None; }

    agent_terminal_state_ = t.is_terminal;
    env_steps_remaining_ = env_steps_remaining;

    return t;
}

void FlyAgent::Render(const FireModelCamera& camera) {
    if (!should_render_){
        return;
    }
    const auto cell_size = static_cast<int>(camera.GetCellSize());
    std::pair<double, double> agent_position = this->GetGridPositionDouble();
    const auto drone_size = !parameters_.show_small_drones_ ? cell_size : (cell_size * ((parameters_.drone_size_ * parameters_.drone_size_) / parameters_.GetCellSize()));
    const auto drone_half = static_cast<int>(std::ceil(drone_size / 2));

    const auto screen_position = camera.GridToScreenPosition(agent_position.first,
                                                             agent_position.second);
    const auto drone_position = std::make_pair(screen_position.first - drone_half,
                                               screen_position.second - drone_half);
    const auto view_range_position = std::make_pair(drone_position.first - 0.5,
                                                    drone_position.second - 0.5);

    std::pair<int, int> goal_screen_position = camera.GridToScreenPosition(goal_position_.first -0.5,
                                                                           goal_position_.second - 0.5);

    const auto fast_drone = this->GetAgentSubType() == "ExploreFlyAgent";
    if (!fast_drone) {
        goal_texture_renderer_.RenderGoal(goal_screen_position, cell_size);
    }
    // Render a glowing circle beneath the drone if active
    if (active_ || parameters_.show_drone_circles_) {
        // Glowing white color
        auto color = SDL_Color{255, 255, 255, 190};
        auto renderer = drone_texture_renderer_.GetRenderer();
        auto radius = drone_half;
        auto x = drone_position.first + drone_half;
        auto y = drone_position.second + drone_half;
        SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);
        SDL_SetRenderDrawColor(renderer, color.r, color.g, color.b, color.a);

        for(int w = 0; w < radius * 2; w++) {
            for(int h = 0; h < radius * 2; h++) {
                int dx = radius - w; // horizontal offset
                int dy = radius - h; // vertical offset
                if((dx*dx + dy*dy) <= (radius * radius)) {
                    SDL_RenderDrawPoint(renderer, x + dx, y + dy);
                }
            }
        }
    }
    drone_texture_renderer_.RenderDrone(drone_position, static_cast<int>(drone_size), fast_drone ? 150 : 255);
    drone_texture_renderer_.RenderViewRange(view_range_position, cell_size, view_range_, fast_drone ? 150 : 255);
}

void FlyAgent::AppendTrail(const std::pair<int, int> &position) {
    trail_.push_back(std::make_pair<double, double>(position.first / parameters_.GetCellSize(), position.second / parameters_.GetCellSize()));
    if (trail_.size() > static_cast<size_t>(trail_length_)) {
        trail_.pop_front();
    }
}

std::deque<std::pair<double, double>> FlyAgent::GetCameraTrail(const FireModelCamera& camera) const {
    std::deque<std::pair<double, double>> screen_trail;
    if (trail_.size() > 1) {
        for (const auto&[fst, snd] : trail_) {
            auto screen_trail_pos = camera.GridToScreenPosition(fst, snd);
            screen_trail.emplace_back(screen_trail_pos.first, screen_trail_pos.second);
        }
    }
    return screen_trail;
}

void FlyAgent::Step(double speed_x, double speed_y, const std::shared_ptr<GridMap>& gridmap) {
    this->vel_vector_ = this->MovementStep(speed_x, speed_y);
    this->newly_explored_cells_ += gridmap->UpdateExploredAreaFromDrone(this->GetGridPosition(), this->GetViewRange());
    // Calculates if the Drone is in the grid and if not how far it is away from the grid
    // These values are used to calculate the reward
    std::pair<int, int> drone_position = GetGridPosition();
    drone_in_grid_ = gridmap->IsPointInGrid(drone_position.first, drone_position.second);
    if (!drone_in_grid_) {out_of_area_counter_++;}
    else {out_of_area_counter_ = 0;}
    //    CalcMaxDistanceFromMap();
}

void FlyAgent::DispenseWater(const std::shared_ptr<GridMap>& grid_map, int water_dispense) {
    // Returns true if fire was extinguished
    if (water_dispense == 1) {
        dispensed_water_ = true;
        std::pair<int, int> grid_position = GetGridPosition();
        extinguished_fire_ = grid_map->WaterDispension(grid_position.first, grid_position.second);
        if (extinguished_fire_) {
            if (grid_map->GetNumBurningCells() == 0) {
                extinguished_last_fire_ = true;
            }
            num_extinguished_fires_++;
        }
    } else {
        dispensed_water_ = false;
        extinguished_fire_ = false;
    }
}

bool FlyAgent::DispenseWaterCertain(const std::shared_ptr<GridMap>& grid_map) {
    std::pair<int, int> grid_position = GetGridPosition();
    bool cell_is_burning = false;
    if (grid_map->IsPointInGrid(grid_position.first, grid_position.second)){
        cell_is_burning = grid_map->At(grid_position.first, grid_position.second).IsBurning();
    }
    if (cell_is_burning) {
        dispensed_water_ = true;
        water_capacity_ -= parameters_.use_water_limit_ ? 1 : 0;
        extinguished_fire_ = grid_map->WaterDispension(grid_position.first, grid_position.second);
        if (extinguished_fire_) {
            if (grid_map->GetNumBurningCells() == 0) {
                extinguished_last_fire_ = true;
            }
            num_extinguished_fires_++;
        }
        return extinguished_fire_;
    } else {
        dispensed_water_ = false;
        extinguished_fire_ = false;
        return false;
    }
}

std::pair<double, double> FlyAgent::CalculateLocalGoal(double global_x, double global_y) {
    double vision_radius = this->GetViewRange();
    auto current_pos = this->GetGridPositionDouble();

    // Compute vector towards global goal
    double dx = global_x - current_pos.first;
    double dy = global_y - current_pos.second;
    double distance = sqrt(dx*dx + dy*dy);

    // Clamp the distance to within vision radius
    if (distance <= vision_radius) {
        return {global_x, global_y}; // goal within vision
    } else {
        // Move vision_radius steps towards global goal
        double scale = vision_radius / distance;
        return {current_pos.first + dx * scale, current_pos.second + dy * scale};
    }
}

std::pair<int, int> FlyAgent::GetGridPosition() {
    int x, y;
    parameters_.ConvertRealToGridCoordinates(position_.first, position_.second, x, y);
    return std::make_pair(x, y);
}

double FlyAgent::GetDistanceToGoal() {
    return sqrt(pow(this->GetGoalPosition().first - this->GetGridPositionDouble().first, 2) +
                pow(this->GetGoalPosition().second - this->GetGridPositionDouble().second, 2)
    );
}

void FlyAgent::FlyPolicy(const std::shared_ptr<GridMap>& gridmap){
    if(this->policy_type_ == EXTINGUISH_FIRE) {
        // If policy is to extinguish fire and the goal is set to groundstation(start of mission), set new goal to next fire
        // as soon as some fire is explored
        if (almostEqual(this->GetGoalPosition(), gridmap->GetGroundstation()->GetGridPositionDouble())) {
            auto fire_map_empty = gridmap->GetRawFirePositionsFromFireMap().empty();
            if (!fire_map_empty) {
                this->SetGoalPosition(gridmap->GetNextFire(this->GetGridPosition()));
                return;
            }
        }
        // If at goal position, dispense water and set new goal
        if (almostEqual(this->GetGoalPosition(), this->GetGridPositionDouble())) {
            this->DispenseWaterCertain(gridmap);
            gridmap->RemoveReservation(this->GetGoalPositionInt());
            if (this->water_capacity_ <= 0) {
                this->SetGoalPosition(gridmap->GetGroundstation()->GetGridPositionDouble());
                this->policy_type_ = FLY_TO_GROUNDSTATION;
            } else {
                this->SetGoalPosition(gridmap->GetNextFire(this->GetGridPosition()));
            }
        }
    }
    else if (this->policy_type_ == FLY_TO_GROUNDSTATION) {
        if (this->GetGoalPositionInt() == this->GetGridPosition()) {
            this->policy_type_ = RECHARGE;
        }
    }
    else if (this->policy_type_ == RECHARGE) {
        if (parameters_.use_water_limit_) {
            if (this->water_capacity_ < parameters_.GetWaterCapacity()) {
                this->water_capacity_ += parameters_.GetWaterRefillDt();
            } else {
                this->policy_type_ = EXTINGUISH_FIRE;
                this->SetGoalPosition(gridmap->GetNextFire(this->GetGridPosition()));
            }
        } else {
            this->water_capacity_ = parameters_.GetWaterCapacity();
            this->policy_type_ = EXTINGUISH_FIRE;
            this->SetGoalPosition(gridmap->GetNextFire(this->GetGridPosition()));
        }
    }
}

std::pair<double, double> FlyAgent::MovementStep(double netout_x, double netout_y) {
    std::pair<double, double> velocity_vector = this->GetNewVelocity(netout_x, netout_y);
    position_.first += velocity_vector.first * parameters_.GetDt();
    position_.second += velocity_vector.second * parameters_.GetDt();
    this->AppendTrail(std::make_pair(static_cast<int>(position_.first), static_cast<int>(position_.second)));
    return velocity_vector;
}

bool FlyAgent::GetDistanceToNearestBoundaryNorm(int rows, int cols, double view_range, std::vector<double>& out_norm) {
    auto position = this->GetGridPositionDouble();
    double x = position.first;
    double y = position.second;
    auto view_range_half = view_range * 0.5;

    // Displacements to each boundary line (only one axis nonzero)
    const double dxL = -x;          // to left   (x=0)
    const double dxR =  rows - x;   // to right  (x=cols)
    const double dyT = -y;          // to top    (y=0)
    const double dyB =  cols - y;   // to bottom (y=rows)

    // Pick the nearest by absolute distance
    double cand_vals[4] = { std::fabs(dxL), std::fabs(dxR), std::fabs(dyT), std::fabs(dyB) };
    int argmin = 0;
    for (int i = 1; i < 4; ++i) if (cand_vals[i] < cand_vals[argmin]) argmin = i;

    std::pair<double,double> v; // unnormalized displacement
    switch (argmin) {
        case 0: v = { dxL, 0.0 }; break;
        case 1: v = { dxR, 0.0 }; break;
        case 2: v = { 0.0, dyT }; break;
        default:v = { 0.0, dyB }; break;
    }

    // Visibility inside rectangular view (axis-aligned), half-size = view_range_half
    if (std::fabs(v.first)  <= view_range_half && std::fabs(v.second) <= view_range_half) {
        // Don't use internal view_range here, because at some points this function is called, it's not initialized
        out_norm = { v.first / view_range_half, v.second / view_range_half, 0.0, 0.0 }; // same scaling as neighbors
        return true;
    }
    out_norm = { 0.0, 0.0, 0.0, 0.0 };
    return false;
}

void FlyAgent::CalcMaxDistanceFromMap() {
    max_distance_from_map_ = 0;
    if (!drone_in_grid_) {
        // TODO: This calculation really should be put in it's separate function, but for now it works here
        auto cell_size = parameters_.GetCellSize();
        auto norm_x = position_.first / cell_size;
        auto norm_y = position_.second / cell_size;
        auto map_dims = this->GetLastState().get_map_dimensions();
        norm_x = (2 * norm_x / map_dims.first) - 1;
        norm_y = (2 * norm_y / map_dims.second) - 1;
        std::pair<double, double> pos = std::make_pair(norm_x, norm_y);
        double max_distance1 = 0;
        double max_distance2 = 0;
        if (pos.first < 0 || pos.second < 0) {
            max_distance1 = abs(std::min(pos.first, pos.second));
        } else if (pos.first > 1 || pos.second > 1) {
            max_distance2 = std::max(pos.first, pos.second) - 1;
        }
        max_distance_from_map_ = std::max(max_distance1, max_distance2);
    }
}

double FlyAgent::FindNearestFireDistance() {
    std::pair<int, int> drone_grid_position = GetGridPosition();
    double min_distance = std::numeric_limits<double>::max();
    std::vector<std::vector<int>> fire_status = this->GetLastState().GetFireView();

    for (int y = 0; y <= view_range_; ++y) {
        for (int x = 0; x <= view_range_; ++x) {
            if (fire_status[x][y] == 1) { // Assuming 1 indicates fire
                std::pair<int, int> fire_grid_position = std::make_pair(
                    drone_grid_position.first + y - (view_range_ / 2),
                    drone_grid_position.second + x - (view_range_ / 2)
                );

                double real_x, real_y;
                parameters_.ConvertGridToRealCoordinates(fire_grid_position.first, fire_grid_position.second, real_x, real_y);
                double distance = sqrt(
                    pow(real_x - position_.first, 2) +
                    pow(real_y - position_.second, 2)
                );

                if (distance < min_distance) {
                    min_distance = distance;
                }
            }
        }
    }

    return min_distance;
}

// Checks whether the drone sees fire in the current fire status and return how much
int FlyAgent::DroneSeesFire() {
    std::vector<std::vector<int>> fire_status = this->GetLastState().GetFireView();
    int count = std::accumulate(fire_status.begin(), fire_status.end(), 0,
                                [](int acc, const std::vector<int>& vec) {
                                    return acc + std::count(vec.begin(), vec.end(), 1);
                                }
    );
    return count;
}

double DiscretizeOutput(double netout, double bin_size) {
    double clamped = std::clamp(netout, -1.0, 1.0);
    double discrete = std::round(clamped / bin_size) * bin_size;

    return discrete;
}

std::pair<double, double> FlyAgent::GetNewVelocity(double next_speed_x, double next_speed_y) const {
    double new_speed_x, new_speed_y;
    if (parameters_.use_vel_bins_) {
        next_speed_x = DiscretizeOutput(next_speed_x, 0.05);
        next_speed_y = DiscretizeOutput(next_speed_y, 0.05);
    }
    if (parameters_.use_velocity_change_){
        // Acceleration model: Netout determines the CHANGE in velocity (treat max_speed_ as max_acceleration)
        new_speed_x = vel_vector_.first + ((next_speed_x * max_acceleration_.first) * parameters_.GetDt());
        new_speed_y = vel_vector_.second + ((next_speed_y * max_acceleration_.second) * parameters_.GetDt());
    } else {
        // Direct velocity model: Netout determines the new velocity directly
        new_speed_x = (next_speed_x * max_speed_.first);
        new_speed_y = (next_speed_y * max_speed_.second);
    }
    // Clamp new_speed between -max_speed_.first and max_speed_.first
    new_speed_x = std::clamp(new_speed_x, -max_speed_.first, max_speed_.first);
    new_speed_y = std::clamp(new_speed_y, -max_speed_.second, max_speed_.second);

    return std::make_pair(new_speed_x, new_speed_y);
}


