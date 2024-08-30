//
// Created by nex on 04.06.24.
//

#include "reinforcementlearning_handler.h"

std::shared_ptr<ReinforcementLearningHandler> ReinforcementLearningHandler::instance_ = nullptr;

ReinforcementLearningHandler::ReinforcementLearningHandler(FireModelParameters &parameters) :
                                                           parameters_(parameters),
                                                           rewards_(parameters_.GetRewardsBufferSize()){
    drones_ = std::make_shared<std::vector<std::shared_ptr<DroneAgent>>>();
    agent_is_running_ = false;
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
    for (int i = 0; i < parameters_.GetNumberOfDrones(); ++i) {
        auto newDrone = std::make_shared<DroneAgent>(gridmap_->GetRandomPointInGrid(), parameters_, i);
        if (mode == Mode::GUI_RL) {
            newDrone->SetRenderer(model_renderer_->GetRenderer());
        }
        auto drone_view = gridmap_->GetDroneView(newDrone);
        newDrone->Initialize(drone_view.first, drone_view.second, std::make_pair(gridmap_->GetCols(), gridmap_->GetRows()));
        newDrone->SetLastNearFires(newDrone->DroneSeesFire());
        newDrone->SetLastDistanceToFire(newDrone->FindNearestFireDistance());
        drones_->push_back(newDrone);
    }
}

void ReinforcementLearningHandler::StepDrone(int drone_idx, double speed_x, double speed_y, int water_dispense) {
    std::pair<double, double> vel_vector = drones_->at(drone_idx)->Step(speed_x, speed_y);
    drones_->at(drone_idx)->DispenseWater(*gridmap_, water_dispense);
    std::pair<std::vector<std::vector<int>>, std::vector<std::vector<int>>> drone_view = gridmap_->GetDroneView(drones_->at(drone_idx));
    std::vector<std::vector<int>> updated_map = gridmap_->GetUpdatedMap(drones_->at(drone_idx), drone_view.second);
    drones_->at(drone_idx)->UpdateStates(*gridmap_, vel_vector, drone_view.first, drone_view.second, updated_map);
}

void ReinforcementLearningHandler::InitFires() {
        this->startFires(parameters_.fire_percentage_);
}

double ReinforcementLearningHandler::CalculateReward(std::shared_ptr<DroneAgent> drone, bool terminal_state) const {
    // Calculate distance to nearest fire, dirty maybe change that later(lol never gonna happen)
    double distance_to_fire = drone->FindNearestFireDistance();
    // Get the last distance to the nearest fire
    double last_distance_to_fire_ = drone->GetLastDistanceToFire();
    // Check if the Fire Extinquished flag is set
    bool fire_extinguished = drone->GetExtinguishedFire();
    // Calculate the number of fires in the last states drone view
    int near_fires = drone->DroneSeesFire();
    // Get the last near fires
    double last_near_fires_ = drone->GetLastNearFires();
    // Get the last outside area counter of the drone
    int out_of_area_counter = drone->GetLastState().CountOutsideArea();
    // Get the max_distance from map
    double max_distance = drone->GetMaxDistanceFromMap();
    // Is the Drone in the Gridmap?
    bool drone_in_grid = drone->GetDroneInGrid();
    std::string debug_str = "";

    double reward = 0;
    if (fire_extinguished) {
        if (near_fires == 0) {
            // all fires in sight were extinguished
            reward += 1;
        } else {
            // a fire was extinguished
            reward += 0.5;
        }
        if(terminal_state)
            reward += 5;

        debug_str += "Fire extinguished Reward: " + std::to_string(reward) + "\n";
    }

    // Boundary Penalty
    double ooa_factor = -0.0005 * out_of_area_counter;
    debug_str += "OOA Factor: " + std::to_string(ooa_factor) + "\n";
    reward += ooa_factor;

    // Check the boundaries of the network
    if (!drone_in_grid) {
        reward += (-0.02 + ooa_factor) * max_distance;
        debug_str += "Boundary Reward:" + std::to_string((-0.02 + ooa_factor) * max_distance) + "\n";
        if(terminal_state) {
            reward += -0.1;
        }
        debug_str += "Boundary Terminal Reward: " + std::to_string(-0.1) + "\n";
    }

    // Staying Alive Reward
//    reward += pow(0.1, (0.000000000000000000001+gridmap_->PercentageBurned()/0.05));
//    debug_str += "Staying Alive Reward: " + std::to_string(pow(0.1, (0.000000000000000000001+gridmap_->PercentageBurned()/0.05))) + "\n";

    // Fire Proximity Reward
    if (last_near_fires_ == 0 && near_fires > 0) {
        reward += 0.2; // TODO we need to check for firespread here
        debug_str += "Fire Proximity 1:" + std::to_string(0.2) + "\n";
    }
    if (near_fires > 0) {
        reward += 0.005 * near_fires;
        debug_str += "Fire Proximity 2:" + std::to_string(0.005 * near_fires) + "\n";
    }
    if ((near_fires < last_near_fires_) && !fire_extinguished){
        reward += -0.01;
        debug_str += "Fire Proximity 3:" + std::to_string(-0.01) + "\n";
    }
    if (near_fires > last_near_fires_){
        reward += 0.2;
        debug_str += "Fire Proximity 4:" + std::to_string(0.2) + "\n";
    }

    // Reward for moving towards the fire
    // If last_distance or last_distance_to_fire_ is very large, dismiss the reward
    // These high values occure when fire spreads and gets extinguished
    if (!(last_distance_to_fire_ > 1000000 || distance_to_fire > 1000000))
    {
        double delta_distance = last_distance_to_fire_ - distance_to_fire;

        if(delta_distance > 0) {
            reward += 0.0125 * delta_distance;
        }
        debug_str += "Moving towards fire Reward: " + std::to_string(0.0125 * delta_distance) + "\n";
    }
    
    // if (water_dispensed)
    //     reward += -0.1;

    // The environment reached terminal state, this means the map is burned too much
    if(terminal_state && !fire_extinguished && drone_in_grid){
        reward += -0.1;
        debug_str += "Terminal State Reward: " + std::to_string(-0.1) + "\n";
    }

    // Set some drone variables now so we don't need to compute them again
    drone->SetLastDistanceToFire(distance_to_fire);
    drone->SetLastNearFires(near_fires);
    debug_str += "Total Reward: " + std::to_string(reward) + "\n\n\n";
#ifdef DEBUG_REWARD_YES
    std::cout << debug_str;
#endif
    return reward;
}

std::tuple<std::vector<std::deque<std::shared_ptr<State>>>, std::vector<double>, std::vector<bool>, std::pair<bool, bool>, double> ReinforcementLearningHandler::Step(std::vector<std::shared_ptr<Action>> actions) {

    if (gridmap_ != nullptr) {
        std::vector<bool> terminals;
        std::vector<double> rewards;
        // TODO this is dirty
        bool drone_died = false;
        bool all_fires_ext = false;

        // Move the drones and get the next_observation
        for (int i = 0; i < (*drones_).size(); ++i) {
            // Get the speed and water dispense from the action
            double speed_x = std::dynamic_pointer_cast<DroneAction>(actions[i])->GetSpeedX(); // change this to "real" speed
            double speed_y = std::dynamic_pointer_cast<DroneAction>(actions[i])->GetSpeedY();
            int water_dispense = std::dynamic_pointer_cast<DroneAction>(actions[i])->GetWaterDispense();

            StepDrone(i, speed_x, speed_y, water_dispense);

            terminals.push_back(false);

            // Check if drone is out of area for too long
            if (drones_->at(i)->GetOutOfAreaCounter() > 5) {
                terminals[i] = true;
                drone_died = true;
            } else if(gridmap_->PercentageBurned() > 0.30) {
//                std::cout << "Percentage burned: " << gridmap_->PercentageBurned() << " resetting GridMap" << std::endl;
                terminals[i] = true;
                drone_died = true;
            } else if (!gridmap_->IsBurning()) {
//                std::cout << "Fire is extinguished, resetting GridMap" << std::endl;
                terminals[i] = true;
                all_fires_ext = true;
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
