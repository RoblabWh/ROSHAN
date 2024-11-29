//
// Created by nex on 04.06.24.
//

#include "reinforcementlearning_handler.h"

std::shared_ptr<ReinforcementLearningHandler> ReinforcementLearningHandler::instance_ = nullptr;

ReinforcementLearningHandler::ReinforcementLearningHandler(FireModelParameters &parameters) :
                                                           parameters_(parameters),
                                                           rewards_(parameters_.GetRewardsBufferSize()){
    drones_ = std::make_shared<std::vector<std::shared_ptr<DroneAgent>>>();
    groundstation_ = std::make_shared<Groundstation>(parameters_.GetGroundstationPosition(), parameters_);
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

void ReinforcementLearningHandler::ResetDrones(Mode mode) {
    drones_->clear();
    auto corner = gridmap_->GetRandomCorner();
    groundstation_ = std::make_shared<Groundstation>(corner, parameters_);
    if (mode == Mode::GUI_RL) {
        groundstation_->SetRenderer(model_renderer_->GetRenderer());
    }
    total_env_steps_ = parameters_.GetTotalEnvSteps();
    for (int i = 0; i < parameters_.GetNumberOfDrones(); ++i) {
        auto newDrone = std::make_shared<DroneAgent>(gridmap_->GetRandomPointInGrid(), parameters_, i);
        if (mode == Mode::GUI_RL) {
            newDrone->SetRenderer(model_renderer_->GetRenderer());
        }
        gridmap_->UpdateExploredAreaFromDrone(newDrone);
        newDrone->SetExploreDifference(0);

        // Generate random number between -1 and 1
        double dicider = -1 + (rand() % 200) / 100.0;
        std::pair<double, double> goal_pos;
        auto rl_mode = rl_status_["rl_mode"].cast<std::string>();
        if (dicider < 0.89 || rl_mode == "eval") {
            goal_pos = gridmap_->GetNextFire(newDrone);
        } else {
            goal_pos = groundstation_->GetGridPosition();
        }
        newDrone->SetGoalPosition(goal_pos);
        newDrone->SetLastDistanceToGoal(newDrone->GetDistanceToGoal());
        newDrone->Initialize(*gridmap_);
        newDrone->SetLastNearFires(newDrone->DroneSeesFire());
        newDrone->SetLastDistanceToFire(newDrone->FindNearestFireDistance());
        drones_->push_back(newDrone);
    }
}

void ReinforcementLearningHandler::StepDrone(int drone_idx, double speed_x, double speed_y, int water_dispense) {
    drones_->at(drone_idx)->SetReachedGoal(false);
    std::pair<double, double> vel_vector;
    if(drones_->at(drone_idx)->GetPolicyType() == 0) {
        vel_vector = drones_->at(drone_idx)->Step(speed_x, speed_y);
        if (drones_->at(drone_idx)->GetGoalPositionInt() == drones_->at(drone_idx)->GetGridPosition()) {
            drones_->at(drone_idx)->SetReachedGoal(true);
            drones_->at(drone_idx)->DispenseWaterCertain(*gridmap_);
            if (drones_->at(drone_idx)->GetWaterCapacity() <= 0) {
                auto groundstation_position = groundstation_->GetGridPosition();
                drones_->at(drone_idx)->SetGoalPosition(groundstation_position);
                drones_->at(drone_idx)->SetPolicyType(1);
            } else {
                auto next_fire = gridmap_->GetNextFire(drones_->at(drone_idx));
                drones_->at(drone_idx)->SetGoalPosition(next_fire);
            }
        }
    } else if (drones_->at(drone_idx)->GetPolicyType() == 1) {
        vel_vector = drones_->at(drone_idx)->Step(speed_x, speed_y);
        if (drones_->at(drone_idx)->GetGoalPositionInt() == drones_->at(drone_idx)->GetGridPosition()) {
            drones_->at(drone_idx)->SetPolicyType(2);
        }
    } else {
        vel_vector = std::make_pair(0, 0);
        if (drones_->at(drone_idx)->GetWaterCapacity() <= parameters_.GetWaterCapacity()) {
            drones_->at(drone_idx)->SetWaterCapacity(drones_->at(drone_idx)->GetWaterCapacity() + parameters_.GetWaterRefillDt());
        } else {
            drones_->at(drone_idx)->SetPolicyType(0);
            drones_->at(drone_idx)->SetGoalPosition(gridmap_->GetNextFire(drones_->at(drone_idx)));
        }
    }
//    drones_->at(drone_idx)->DispenseWater(*gridmap_, water_dispense);
    auto drone_view = gridmap_->GetDroneView(drones_->at(drone_idx));
    gridmap_->UpdateExploredAreaFromDrone(drones_->at(drone_idx));
    // TODO consider not only adding the current velocity, but the last netoutputs (these are two potential dimensions)
    drones_->at(drone_idx)->UpdateStates(*gridmap_, vel_vector, drone_view, water_dispense);
}

void ReinforcementLearningHandler::InitFires() {
        this->startFires(parameters_.fire_percentage_);
}

double ReinforcementLearningHandler::FindNearestFireDistanceGlobal(std::shared_ptr<DroneAgent> drone) const {
    double min_distance = std::numeric_limits<double>::max();
    auto fire_map = gridmap_->GetFireMapRef();
    std::pair<double, double> position = drone->GetRealPosition();

    for (int x = 0; x < fire_map.size(); ++x) {
        for (int y = 0; y < fire_map[0].size(); ++y) {
            if (fire_map[x][y] == 1) { // Assuming 1 indicates fire
                double real_x, real_y;
                parameters_.ConvertGridToRealCoordinates(x, y, real_x, real_y);
                double distance = sqrt(
                        pow(real_x - position.first, 2) +
                        pow(real_y - position.second, 2)
                );

                if (distance < min_distance) {
                    min_distance = distance;
                }
            }
        }
    }

    return min_distance;
}

double ReinforcementLearningHandler::CalculateReward(std::shared_ptr<DroneAgent> drone, bool terminal_state) const {
    // Calculate distance to goal
    double distance_to_goal = drone->GetDistanceToGoal();
    double last_distance_to_goal = drone->GetLastDistanceToGoal();
    double delta_distance = last_distance_to_goal - distance_to_goal;
    bool drone_in_grid = drone->GetDroneInGrid();
    bool fires = gridmap_->GetNumBurningCells() > 0;
    std::string debug_str;

    double reward = 0;

    if (drone->GetReachedGoal()) {
        reward += 1;
        if(terminal_state){
            reward += 9;
            debug_str += "Goal Reached Reward: " + std::to_string(reward) + "\n";
#ifdef DEBUG_REWARD_YES
            std::cout << debug_str;
#endif
            return reward;
        }
        debug_str += "Goal Reached Reward: " + std::to_string(reward) + "\n";
    }

    // Check the boundaries of the map
    if (!drone_in_grid && terminal_state) {
            reward += -10;
            debug_str += "Boundary Terminal Reward: " + std::to_string(-1) + "\n";
#ifdef DEBUG_REWARD_YES
            std::cout << debug_str;
#endif
            return reward;
    }

    // The environment reached terminal state, this means the map is burned too much
    if(terminal_state && (total_env_steps_ <= 0)){
        reward += -10;
        debug_str += "Terminal State(Took too long) Reward: " + std::to_string(-1) + "\n";
        return reward;
    }

    if(terminal_state && fires){
        reward += -10;
        debug_str += "Terminal State(Map Burned) Reward: " + std::to_string(-1) + "\n";
        return reward;
    }

    if (delta_distance > 0) {
        reward += 0.1;
        debug_str += "Distance Reward: " + std::to_string(0.1) + "\n";
    }
//    else {
//        reward += -0.5;
//        debug_str += "Distance Reward: " + std::to_string(-0.2) + "\n";
//    }

    // Set some drone variables now so we don't need to compute them again

    debug_str += "Total Reward: " + std::to_string(reward) + "\n\n";
#ifdef DEBUG_REWARD_YES
    std::cout << debug_str;
#endif
    drone->SetLastDistanceToGoal(distance_to_goal);
    return reward;
}

double ReinforcementLearningHandler::CalculateReward2(std::shared_ptr<DroneAgent> drone, bool terminal_state) const {
    // Calculate distance to nearest fire, dirty maybe change that later(lol never gonna happen)
    //double distance_to_fire = drone->FindNearestFireDistance();
    double distance_to_fire = FindNearestFireDistanceGlobal(drone);
    // Get the last distance to the nearest fire
    double last_distance_to_fire_ = drone->GetLastDistanceToFire();
    // Check if the Fire Extinquished flag is set
    bool fire_extinguished = drone->GetExtinguishedFire();
    // Calculate the number of fires in the last states drone view
    int near_fires = drone->DroneSeesFire();
    // Get the last near fires
    double last_near_fires_ = drone->GetLastNearFires();
    // Get the last outside area counter of the drone
    //int out_of_area_counter = drone->GetLastState().CountOutsideArea();
    // Get the max_distance from map
    //double max_distance = drone->GetMaxDistanceFromMap();
    bool drone_in_grid = drone->GetDroneInGrid();
    // Did the drone dispense water?
    bool water_dispensed = drone->GetDispensedWater();
    // Get a value for how much the Drone explored in the last time step
    int explore_difference = drone->GetExploreDifference() / parameters_.GetExplorationTime();
    // How many cells are burned down
    int burned_cells = gridmap_->GetNumBurnedCells();
    // How many cells are burning, not used in reward but in checking terminal states
    bool fires = gridmap_->GetNumBurningCells() > 0;
    std::vector<std::vector<double>> exploration_map = drone->GetLastState().GetExplorationMapNorm();
    std::string debug_str;

    double reward = 0;

    if (fire_extinguished) {
        reward += 1;
        if(terminal_state){
            reward += 5;
            debug_str += "Fire extinguished Reward: " + std::to_string(reward) + "\n";
#ifdef DEBUG_REWARD_YES
            std::cout << debug_str;
#endif
            return reward;
        }
        debug_str += "Fire extinguished Reward: " + std::to_string(reward) + "\n";
    }

    // Boundary Penalty
//    double ooa_factor = -0.005 * out_of_area_counter;
//    debug_str += "OOA Factor: " + std::to_string(ooa_factor) + "\n";
//    reward += ooa_factor;

    // Check the boundaries of the map
    if (!drone_in_grid) {
//        reward += (-0.2 + ooa_factor) * max_distance;
//        debug_str += "Boundary Reward:" + std::to_string((-0.02 + ooa_factor) * max_distance) + "\n";
        if(terminal_state) {
            reward += -5;
            debug_str += "Boundary Terminal Reward: " + std::to_string(-1) + "\n";
#ifdef DEBUG_REWARD_YES
            std::cout << debug_str;
#endif
            return reward;
        }
    }

    // Burned Area Penalty
//    reward += -0.01 * burned_cells;
//    debug_str += "Burned Area Penalty: " + std::to_string(-0.01 * burned_cells) + "\n";

    // Exploration Reward
//    double exploration_reward = 0.05 * explore_difference;
//    debug_str += "Exploration Reward: " + std::to_string(exploration_reward) + "\n";
//    reward += exploration_reward;

    // Calculating the reward for the whole exploration map
    double freshness = 0;
    for (auto &row : exploration_map) {
        for (double value : row) {
            freshness += value;
//            if (value == 0.0) {
//                freshness -= 1.0;
//            }
        }
    }
    freshness /= static_cast<int>(exploration_map.size() * exploration_map[0].size());
    debug_str += "Freshness: " + std::to_string(0.1*freshness) + "\n";
    reward += 0.001 * freshness;

    // Staying Alive Reward
//    reward += pow(0.1, (0.000000000000000000001+gridmap_->PercentageBurned()/0.05));
//    debug_str += "Staying Alive Reward: " + std::to_string(pow(0.1, (0.000000000000000000001+gridmap_->PercentageBurned()/0.05))) + "\n";

    // Fire Proximity Reward
//    if (last_near_fires_ == 0 && near_fires > 0) {
//        reward += 0.2; // TODO we need to check for firespread here
//        debug_str += "Fire Proximity 1:" + std::to_string(0.2) + "\n";
//    }
//    if (near_fires > 0) {
//        reward += 0.5 * near_fires;
//        debug_str += "Fire Proximity 2:" + std::to_string(0.05 * near_fires) + "\n";
//    }
//    if ((near_fires < last_near_fires_) && !fire_extinguished){
//        reward += -0.5;
//        debug_str += "Fire Proximity 3:" + std::to_string(-0.5) + "\n";
//    }
//    if (near_fires > last_near_fires_){
//        reward += 0.2;
//        debug_str += "Fire Proximity 4:" + std::to_string(0.2) + "\n";
//    }

    // Reward for moving towards the fire
    // If last_distance or last_distance_to_fire_ is very large, dismiss the reward
    // These high values occure when fire spreads and gets extinguished
    if (distance_to_fire <= 1000000)
    {
        auto epsilon_ = 1e-6;
        double span = sqrt(pow(gridmap_->GetRows(), 2) + pow(gridmap_->GetCols(), 2)) * parameters_.GetCellSize();
        auto dist_to_fire_reward = std::max(0.0, 0.05 * log(1 / ((distance_to_fire / span) + epsilon_)));
        reward += dist_to_fire_reward;
//        debug_str += "span: " + std::to_string(span) + "\n";
//        debug_str += "distance to fire: " + std::to_string(distance_to_fire) + "\n";
        debug_str += "distance to fire REWARD: " + std::to_string(dist_to_fire_reward) + "\n";
    }
    
//     if (water_dispensed && !fire_extinguished) {
//        reward += -0.3;
//        debug_str += "Water Dispensed Reward: " + std::to_string(-0.01) + "\n";
//     }
    // The environment reached terminal state, this means the map is burned too much
    if(terminal_state && fires){
        reward += -5;
        debug_str += "Terminal State Reward: " + std::to_string(-1) + "\n";
    }
    if(terminal_state && !fires){
        reward += 0;
        debug_str += "Terminal State Reward: " + std::to_string(1) + "\n";
    }

    // Set some drone variables now so we don't need to compute them again
    drone->SetLastDistanceToFire(distance_to_fire);
    drone->SetLastNearFires(near_fires);
    debug_str += "Total Reward: " + std::to_string(reward) + "\n\n";
#ifdef DEBUG_REWARD_YES
    std::cout << debug_str;
#endif
    return reward;
}

std::tuple<std::vector<std::deque<std::shared_ptr<State>>>, std::vector<double>, std::vector<bool>, std::pair<bool, bool>, double> ReinforcementLearningHandler::Step(std::vector<std::shared_ptr<Action>> actions) {

    if (gridmap_ != nullptr) {
        total_env_steps_ -= 1;
        std::vector<bool> terminals;
        std::vector<double> rewards;
        // TODO this is dirty, but maybe ok since it's only used for logging rn
        bool drone_died = false;
        bool all_fires_ext = false;
        gridmap_->UpdateCellDiminishing();
        // First Step through all the Drones and update their states
        for (int i = 0; i < (*drones_).size(); ++i) {
            // Get the speed and water dispense from the action
            double speed_x = std::dynamic_pointer_cast<DroneAction>(
                    actions[i])->GetSpeedX(); // change this to "real" speed
            double speed_y = std::dynamic_pointer_cast<DroneAction>(actions[i])->GetSpeedY();
            int water_dispense = std::dynamic_pointer_cast<DroneAction>(actions[i])->GetWaterDispense();
            StepDrone(i, speed_x, speed_y, water_dispense);
        }

        // Now we calculate the rewards for each drone
        for (int i = 0; i < (*drones_).size(); ++i) {
            terminals.push_back(false);

            // Check if drone is out of area for too long
            if (drones_->at(i)->GetOutOfAreaCounter() > 1) {
                terminals[i] = true;
                drone_died = true;
            } else if(drones_->at(i)->GetReachedGoal() && !eval_mode_) {
                terminals[i] = true;
            } else if(gridmap_->PercentageBurned() > 0.30) {
//                std::cout << "Percentage burned: " << gridmap_->PercentageBurned() << " resetting GridMap" << std::endl;
                terminals[i] = true;
                drone_died = true;
            } else if (drones_->at(i)->GetExtinguishedLastFire()) {
//                std::cout << "Fire is extinguished, resetting GridMap" << std::endl;
                // TODO Don't use gridmap_->IsBurning() because it is not reliable since it returns false when there
                //  are particles in the air. Instead, check if the drone has extinguished the last fire on the map.
                //  This also makes sure that only the drone that actually extinguished the fire gets the reward
                terminals[i] = true;
                all_fires_ext = true;
            } else if (!gridmap_->IsBurning()) {
                terminals[i] = true;
            } else if (total_env_steps_ <= 0) {
                terminals[i] = true;
            }

            double reward = CalculateReward(drones_->at(i), terminals[i]);

            rewards.push_back(reward);

            // For displaying the rewards in the GUI
            rewards_.put(static_cast<float>(reward));

            if (terminals[i]) {
                auto buffer = rewards_.getBuffer();
                float sum = std::accumulate(buffer.begin(), buffer.end(), 0.0f);
                all_rewards_.push_back(sum / buffer.size());
                rewards_.reset();
            }
        }
        // Reset the Environment if all drones are in a terminal state

        double percentage_burned = gridmap_->PercentageBurned();
        std::vector<std::deque<std::shared_ptr<State>>> next_observations = this->GetObservations();
        return {next_observations, rewards, terminals, std::make_pair(drone_died, all_fires_ext), percentage_burned};

    }
    std::cout << "Taking a step into nothingness..." << std::endl;
    return {};
}

void ReinforcementLearningHandler::SetRLStatus(py::dict status) {
    rl_status_ = status;
    auto rl_mode = rl_status_[py::str("rl_mode")].cast<std::string>();
    if (rl_mode == "eval") {
        eval_mode_ = true;
    } else {
        eval_mode_ = false;
    }
}
