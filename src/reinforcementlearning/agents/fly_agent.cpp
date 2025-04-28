//
// Created by nex on 08.04.25.
//

#include "fly_agent.h"

FlyAgent::FlyAgent(FireModelParameters &parameters, int id, int time_steps) : Agent(parameters, 300) {
    id_ = id;
    agent_type_ = "FlyAgent";
    time_steps_ = time_steps;
    view_range_ = FireModelParameters::GetViewRange(agent_type_);
    water_capacity_ = parameters_.GetWaterCapacity();
    max_speed_ = parameters_.GetMaxVelocity();
    out_of_area_counter_ = 0;
//    if (agent_type_ == "FlyAgent") {
//        if (rng_number <= parameters_.corner_start_percentage_) {
//            point = grid_map->GetNonGroundStationCorner();
//        }
//        else if (rng_number <= parameters_.groundstation_start_percentage_) {
//            point = grid_map->GetGroundstation()->GetGridPosition();
//        }
//        else {
//            point = grid_map->GetRandomPointInGrid();
//        }
//    } else {
//        int view_range_off = view_range_ / 2;
//        point = grid_map->GetGroundstation()->GetGridPosition();
//        int off_x = point.first <= grid_map->GetRows() / 2 ? view_range_off - 1 : -view_range_off + 1;
//        int off_y = point.second <= grid_map->GetCols() / 2 ? view_range_off - 1: -view_range_off + 1;
//        point.first += off_x;
//        point.second += off_y;
//    }
//
//    double x = (point.first + 0.5); // position in grid + offset to center
//    double y = (point.second + 0.5); // position in grid + offset to center
//    position_ = std::make_pair(x * parameters_.GetCellSize(), y * parameters_.GetCellSize());
//    goal_position_ = std::make_pair(0, 0);
//    view_range_ = FireModelParameters::GetViewRange(agent_type_);
//    water_capacity_ = parameters_.GetWaterCapacity();
//    out_of_area_counter_ = 0;
}

void FlyAgent::Initialize(int mode, const std::shared_ptr<GridMap>& grid_map, const std::shared_ptr<FireModelRenderer>& model_renderer, const std::string& rl_mode) {
    if (mode == Mode::GUI_RL) {
        this->SetDroneTextureRenderer(model_renderer->GetRenderer());
        this->SetGoalTextureRenderer(model_renderer->GetRenderer());
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double rng_number = dist(gen);
    std::pair<int, int> point;

    if (parameters_.GetHierarchyType() == "FlyAgent") {
        if (rng_number <= parameters_.corner_start_percentage_) {
            point = grid_map->GetNonGroundStationCorner();
        }
        else if (rng_number <= parameters_.groundstation_start_percentage_) {
            point = grid_map->GetGroundstation()->GetGridPosition();
        }
        else {
            point = grid_map->GetRandomPointInGrid();
        }
        // Generate random number between 0 and 1
        std::pair<double, double> goal_pos = std::pair<double, double>(-1, -1);
        if (rng_number < parameters_.fire_goal_percentage_ || rl_mode == "eval") {
            goal_pos = grid_map->GetNextFire(this->GetGridPosition());
        }
        if (std::pair<double, double>(-1, -1) == goal_pos) {
            goal_pos = grid_map->GetGroundstation()->GetGridPositionDouble();
        }

        this->SetGoalPosition(goal_pos);
    } else {
        point = grid_map->GetGroundstation()->GetGridPosition();
        SetGoalPosition(std::make_pair(grid_map->GetRows() / 2, grid_map->GetCols() / 2));
    }
    SetPosition(point);

//    grid_map->UpdateExploredAreaFromDrone(this->GetGridPosition(), this->GetViewRange());
//    this->newly_explored_cells_ = (this->view_range_ + 1) * (this->view_range_ + 1);

    this->InitializeFlyAgentStates(grid_map);
    this->last_distance_to_goal_ = this->GetDistanceToGoal();
}

std::shared_ptr<AgentState> FlyAgent::BuildAgentState(const std::shared_ptr<GridMap>& grid_map,
                                                      std::pair<double, double> velocity,
                                                      int water_dispense) {
    auto state = std::make_shared<AgentState>();

    state->SetVelocity(velocity);
    state->SetWaterDispense(water_dispense);
    state->SetDroneView(grid_map->GetDroneView(GetGridPosition(), GetViewRange()));
    state->SetTotalDroneView(grid_map->GetInterpolatedDroneView(GetGridPosition(), GetViewRange()));
    state->SetExplorationMap(grid_map->GetExploredMap());
    state->SetFireMap(grid_map->GetFireMap());
    state->SetPosition(position_);
    state->SetGoalPosition(goal_position_);
    state->SetMapDimensions({grid_map->GetRows(), grid_map->GetCols()});
    state->SetCellSize(parameters_.GetCellSize());

    return state;
}

void FlyAgent::InitializeFlyAgentStates(const std::shared_ptr<GridMap>& grid_map) {
    for(int i = 0; i < time_steps_; ++i) {
        agent_states_.push_front(BuildAgentState(grid_map, {0, 0}, 0));
    }
}

void FlyAgent::UpdateStates(const std::shared_ptr<GridMap>& grid_map,
                            std::pair<double, double> velocity_vector,
                            int water_dispense) {
    agent_states_.push_front(BuildAgentState(grid_map, velocity_vector, water_dispense));

    // Maximum number of states i.e. memory
    if (agent_states_.size() > time_steps_) {
        agent_states_.pop_back();
    }
}

void FlyAgent::PerformFly(FlyAction* action, const std::string& hierarchy_type, const std::shared_ptr<GridMap>& gridMap) {
    // Get the speed and water dispense from the action
    double speed_x = action->GetSpeedX(); // change this to "real" speed
    double speed_y = action->GetSpeedY();
    this->Step(speed_x, speed_y, gridMap);
    if (hierarchy_type == "FlyAgent") {
        this->FlyPolicy(gridMap);
        did_hierarchy_step = true;
    } else {
        objective_reached_ = false;
        if (this->almostEqual(this->GetGoalPosition(), this->GetGridPositionDouble())) {
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
        reward_components["GoalReached"] = 5;
    }

    if (!drone_in_grid && agent_terminal_state_) {
        reward_components["BoundaryTerminal"] = -2;
    }

    if (agent_terminal_state_ && (env_steps_remaining_ <= 0)) {
        reward_components["TimeOut"] = -2;
    }

    if (delta_distance > 0) {
//        auto safety_factor = std::tanh(10 * distance_to_boundary);
        reward_components["DistanceImprovement"] = 0.2 * delta_distance; // * safety_factor;
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

void FlyAgent::Render(std::pair<int, int> position, std::pair<int, int> goal_position_screen, int size) {
    drone_texture_renderer_.Render(position, size, view_range_, 0, active_);
    goal_texture_renderer_.RenderGoal(goal_position_screen, size);
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

std::pair<double, double> FlyAgent::MovementStep(double netout_x, double netout_y) {
    std::pair<double, double> velocity_vector = this->GetNewVelocity(netout_x, netout_y);
    auto adjusted_vel_vector = std::make_pair(velocity_vector.first * parameters_.GetDt(), velocity_vector.second * parameters_.GetDt());
    position_.first += adjusted_vel_vector.first;
    position_.second += adjusted_vel_vector.second;
    return adjusted_vel_vector;
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

std::pair<int, int> FlyAgent::GetGridPosition() {
    int x, y;
    parameters_.ConvertRealToGridCoordinates(position_.first, position_.second, x, y);
    return std::make_pair(x, y);
}

// Checks whether the drone sees fire in the current fire status and return how much
int FlyAgent::DroneSeesFire() {
    std::vector<std::vector<int>> fire_status = GetLastState().GetFireView();
    int count = std::accumulate(fire_status.begin(), fire_status.end(), 0,
                                [](int acc, const std::vector<int>& vec) {
                                    return acc + std::count(vec.begin(), vec.end(), 1);
                                }
    );
    return count;
}

double FlyAgent::FindNearestFireDistance() {
    std::pair<int, int> drone_grid_position = GetGridPosition();
    double min_distance = std::numeric_limits<double>::max();
    std::vector<std::vector<int>> fire_status = GetLastState().GetFireView();

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

void FlyAgent::CalcMaxDistanceFromMap() {
    max_distance_from_map_ = 0;
    if (!drone_in_grid_) {
        out_of_area_counter_++;
        std::pair<double, double> pos = GetLastState().GetGridPositionDoubleNorm();
        double max_distance1 = 0;
        double max_distance2 = 0;
        if (pos.first < 0 || pos.second < 0) {
            max_distance1 = abs(std::min(pos.first, pos.second));
        } else if (pos.first > 1 || pos.second > 1) {
            max_distance2 = std::max(pos.first, pos.second) - 1;
        }
        max_distance_from_map_ = std::max(max_distance1, max_distance2);
    } else {
        out_of_area_counter_ = 0;
    }
}

double FlyAgent::GetDistanceToGoal() {
    return sqrt(pow(this->GetGoalPosition().first - this->GetGridPositionDouble().first, 2) +
                pow(this->GetGoalPosition().second - this->GetGridPositionDouble().second, 2)
    );
}

void FlyAgent::Step(double speed_x, double speed_y, const std::shared_ptr<GridMap>& gridmap) {
    std::pair<double, double> vel_vector = this->MovementStep(speed_x, speed_y);
    this->newly_explored_cells_ += gridmap->UpdateExploredAreaFromDrone(this->GetGridPosition(), this->GetViewRange());
    this->UpdateStates(gridmap, vel_vector, 0);
    // Calculates if the Drone is in the grid and if not how far it is away from the grid
    // These values are used to calculate the reward
    std::pair<int, int> drone_position = GetGridPosition();
    drone_in_grid_ = gridmap->IsPointInGrid(drone_position.first, drone_position.second);
    CalcMaxDistanceFromMap();
}

void FlyAgent::FlyPolicy(const std::shared_ptr<GridMap>& gridmap){
    objective_reached_ = false;
    if(this->policy_type_ == 0) {
        if (this->almostEqual(this->GetGoalPosition(), this->GetGridPositionDouble())) {
            objective_reached_ = true;
            this->DispenseWaterCertain(gridmap);
            if (this->water_capacity_ <= 0) {
                auto groundstation_position = gridmap->GetGroundstation()->GetGridPositionDouble();
                this->SetGoalPosition(groundstation_position);
                this->policy_type_ = 1;
            } else {
                auto next_fire = gridmap->GetNextFire(this->GetGridPosition());
                this->SetGoalPosition(next_fire);
            }
        }
    } else if (this->policy_type_ == 1) {
        if (this->GetGoalPositionInt() == this->GetGridPosition()) {
            this->policy_type_ = 2;
        }
    } else if (this->policy_type_ == 2) {
        if (parameters_.recharge_time_active_) {
            if (this->water_capacity_ < parameters_.GetWaterCapacity()) {
                this->water_capacity_ += parameters_.GetWaterRefillDt();
            } else {
                this->policy_type_ = 0;
                this->SetGoalPosition(gridmap->GetNextFire(this->GetGridPosition()));
            }
        } else {
            this->water_capacity_ = parameters_.GetWaterCapacity();
            this->policy_type_ = 0;
            this->SetGoalPosition(gridmap->GetNextFire(this->GetGridPosition()));
        }
    }
}

std::vector<bool> FlyAgent::GetTerminalStates(bool eval_mode, const std::shared_ptr<GridMap>& grid_map, int env_steps_remaining) {
    std::vector<bool> terminal_states;
    bool terminal_state = false;
    bool drone_died = false;
    bool drone_succeeded = false;

    // If the agent has flown out of the grid it has reached a terminal state and died
    if (GetOutOfAreaCounter() > 1) {
        terminal_state = true;
        drone_died = true;
    }
    // If the drone has reached the goal and it is not in evaluation mode
    // the goal is reached because the fly agent is trained that way
    if (objective_reached_) {
        if (!eval_mode){
            terminal_state = true;
        }
        drone_succeeded = true;
    }
    // If the agent has taken too long it has reached a terminal state and died
    if (env_steps_remaining <= 0 && !eval_mode) {
        terminal_state = true;
        drone_died = true;
    }

    // TODO CHANGE LATER
    // Terminals only for evaluation
    if (eval_mode){
        if (grid_map->PercentageBurned() > 0.30) {
            terminal_state = true;
            drone_died = true;
        }
        if (extinguished_last_fire_) {
            //  Don't use gridmap_->IsBurning() because it is not reliable since it returns false when there
            //  are particles in the air. Instead, check if the drone has extinguished the last fire on the map.
            //  This also makes sure that only the drone that actually extinguished the fire gets the reward
            terminal_state = true;
        }
        if (!grid_map->IsBurning()) {
            terminal_state = true;
        }
    }

    terminal_states.push_back(terminal_state);
    terminal_states.push_back(drone_died);
    terminal_states.push_back(drone_succeeded);
    agent_terminal_state_ = terminal_state;
    env_steps_remaining_ = env_steps_remaining;
    return terminal_states;
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


