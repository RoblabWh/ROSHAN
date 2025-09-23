//
// Created by nex on 08.04.25.
//

#include "fly_agent.h"

FlyAgent::FlyAgent(FireModelParameters &parameters, int id, int time_steps) :
Agent(parameters, 300){
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
        this->SetDroneTextureRenderer(model_renderer->GetRenderer(), asset_path);
        this->SetGoalTextureRenderer(model_renderer->GetRenderer());
    }

    max_speed_ = std::make_pair(speed, speed);
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
    reward_components_.clear();
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
    state->SetNormScale(norm_scale_);
    state->SetMaxSpeed(max_speed_);
    state->SetVelocity(this->vel_vector_);
    state->SetDroneView(grid_map->GetDroneView(GetGridPosition(), GetViewRange()));
    state->SetTotalDroneView(grid_map->GetInterpolatedDroneView(GetGridPosition(), GetViewRange()));
    state->SetExplorationMap(grid_map->GetExploredMap());
    state->SetFireMap(grid_map->GetFireMap());
    state->SetPosition(position_);
    state->SetGoalPosition(goal_position_);
    state->SetMapDimensions({grid_map->GetRows(), grid_map->GetCols()});
    state->SetCellSize(parameters_.GetCellSize());
    state->SetDistancesToOtherAgents(std::make_shared<std::vector<std::vector<double>>>(distance_to_other_agents_));
    state->SetDistancesMask(std::make_shared<std::vector<bool>>(distance_mask_));

    return state;
}

void FlyAgent::InitializeFlyAgentStates(const std::shared_ptr<GridMap>& grid_map) {
    for(int i = 0; i < time_steps_; ++i) {
        agent_states_.push_front(BuildAgentState(grid_map));
    }
}

void FlyAgent::PerformFly(FlyAction* action, const std::string& hierarchy_type, const std::shared_ptr<GridMap>& gridMap) {

    this->Step(action->GetSpeedX(), action->GetSpeedY(), gridMap);

    if (hierarchy_type == "fly_agent") {
        this->FlyPolicy(gridMap);
        did_hierarchy_step = true;
    } else {
        objective_reached_ = false;
        if (almostEqual(this->GetGoalPosition(), this->GetGridPositionDouble())) {
            objective_reached_ = true;
        }
    }
}

double FlyAgent::CalculateReward() {
    double distance_to_goal = GetDistanceToGoal();
    double delta_distance = last_distance_to_goal_ - distance_to_goal;
    bool drone_in_grid = GetDroneInGrid();

    std::unordered_map<std::string, double> reward_components;
    double total_reward = 0;

    if (objective_reached_) {
        reward_components["GoalReached"] = 1;
    }

    if (!drone_in_grid && agent_terminal_state_) {
        reward_components["BoundaryTerminal"] = -1;
    }

    if (agent_terminal_state_ && (env_steps_remaining_ <= 0)) {
        reward_components["TimeOut"] = -1;
    }

    if (collision_occurred_) {
        reward_components["Collision"] = -1;
    }

    if (delta_distance > 0) {
//        auto safety_factor = std::tanh(10 * distance_to_boundary);
        reward_components["DistanceImprovement"] = 0.1 * delta_distance; // * safety_factor;
    }

//    if (distance_to_boundary < 0.125) {
//        reward_components["Boundary"] = -0.5 * distance_to_boundary;
//    }

    total_reward = ComputeTotalReward(reward_components);
    LogRewards(reward_components);
    last_distance_to_goal_ = distance_to_goal;
    reward_components_ = reward_components;
    this->SetReward(total_reward);
    return total_reward;
}

//TODO TIDY UP !!!!

AgentTerminal FlyAgent::GetTerminalStates(bool eval_mode, const std::shared_ptr<GridMap>& grid_map, int env_steps_remaining) {
    std::vector<bool> terminal_states;
    bool eval = eval_mode && parameters_.extinguish_all_fires_;

    AgentTerminal t;

    // If the agent has flown out of the grid it has reached a terminal state and died
    if (GetOutOfAreaCounter() > 1) {
        t.is_terminal = true;
        t.reason = FailureReason::BoundaryExit;
    }

    if (collision_occurred_) {
        t.is_terminal = true;
        t.reason = FailureReason::Collision;
    }

    // If the drone has reached the goal and it is not in evaluation mode
    // the goal is reached because the fly agent is trained that way
    if (objective_reached_ && !eval) {
        t.is_terminal = true;
    }
    // If the agent has taken too long it has reached a terminal state and died
    if (env_steps_remaining <= 0) {
        t.is_terminal = true;
        t.reason = FailureReason::Timeout;
    }

    // TODO CHANGE LATER
    // Terminals only for evaluation lustiges loeschverhalten
    if (eval){
        if (grid_map->PercentageBurned() > 0.30) {
            t.is_terminal = true;
            t.reason = FailureReason::Burnout;
        }
        if (extinguished_last_fire_) {
            //  Don't use gridmap_->IsBurning() because it is not reliable since it returns false when there
            //  are particles in the air. Instead, check if the drone has extinguished the last fire on the map.
            //  This also makes sure that only the drone that actually extinguished the fire gets the reward
            t.is_terminal = true;
        }
        if (!grid_map->IsBurning()) {
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
    drone_texture_renderer_.RenderDrone(drone_position, static_cast<int>(drone_size), fast_drone ? 190 : 255);
    drone_texture_renderer_.RenderViewRange(view_range_position, cell_size, view_range_, fast_drone ? 190 : 255);
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
        bool fire_extinguished = grid_map->WaterDispension(grid_position.first, grid_position.second);
        if (fire_extinguished) {
            if (grid_map->GetNumBurningCells() == 0) {
                extinguished_last_fire_ = true;
            }
        }
        extinguished_fire_ = fire_extinguished;
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
        water_capacity_ -= 1;
        bool fire_extinguished = grid_map->WaterDispension(grid_position.first, grid_position.second);
        if (fire_extinguished) {
            if (grid_map->GetNumBurningCells() == 0) {
                extinguished_last_fire_ = true;
            }
        }
        extinguished_fire_ = fire_extinguished;
        return fire_extinguished;
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
    objective_reached_ = false;
    if(this->policy_type_ == EXTINGUISH_FIRE) {
        if (almostEqual(this->GetGoalPosition(), this->GetGridPositionDouble())) {
            objective_reached_ = true;
            this->DispenseWaterCertain(gridmap);
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
        if (parameters_.recharge_time_active_) {
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
    auto adjusted_vel_vector = std::make_pair(velocity_vector.first * parameters_.GetDt(), velocity_vector.second * parameters_.GetDt());
    position_.first += adjusted_vel_vector.first;
    position_.second += adjusted_vel_vector.second;
    this->AppendTrail(std::make_pair(static_cast<int>(position_.first), static_cast<int>(position_.second)));
    return adjusted_vel_vector;
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

std::pair<double, double> FlyAgent::GetNewVelocity(double next_speed_x, double next_speed_y) const {
    // Next Speed determines the velocity CHANGE
    //    next_speed_x = DiscretizeOutput(next_speed_x, 0.05);
    //    next_speed_y = DiscretizeOutput(next_speed_y, 0.05);
    //    double new_speed_x = velocity_.first + next_speed_x * max_speed_.first;
    //    double new_speed_y = velocity_.second + next_speed_y * max_speed_.second;

    // Netout determines the velocity DIRECTLY #TODO: Why does this perform worse??
    double new_speed_x = next_speed_x * max_speed_.first;
    double new_speed_y = next_speed_y * max_speed_.second;

    // Clamp new_speed between -max_speed_.first and max_speed_.first
    new_speed_x = std::clamp(new_speed_x, -max_speed_.first, max_speed_.first);
    new_speed_y = std::clamp(new_speed_y, -max_speed_.second, max_speed_.second);

    return std::make_pair(new_speed_x, new_speed_y);
}


