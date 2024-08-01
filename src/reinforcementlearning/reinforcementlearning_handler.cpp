//
// Created by nex on 04.06.24.
//

#include "reinforcementlearning_handler.h"

std::shared_ptr<ReinforcementLearningHandler> ReinforcementLearningHandler::instance_ = nullptr;

ReinforcementLearningHandler::ReinforcementLearningHandler(FireModelParameters &parameters) :
                                                           parameters_(parameters),
                                                           rewards_(parameters_.GetRewardsBufferSize()){
    drones_ = std::make_shared<std::vector<std::shared_ptr<DroneAgent>>>();
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
        drones_->push_back(newDrone);
    }
}

bool ReinforcementLearningHandler::StepDrone(int drone_idx, double speed_x, double speed_y, int water_dispense) {
    std::pair<double, double> vel_vector = drones_->at(drone_idx)->Step(speed_x, speed_y);
    bool dispensed = false;
    if (water_dispense == 1) {
        dispensed = drones_->at(drone_idx)->DispenseWater(*gridmap_);
    }

    std::pair<std::vector<std::vector<int>>, std::vector<std::vector<int>>> drone_view = gridmap_->GetDroneView(drones_->at(drone_idx));
    std::vector<std::vector<int>> updated_map = gridmap_->GetUpdatedMap(drones_->at(drone_idx), drone_view.second);
    drones_->at(drone_idx)->UpdateStates(vel_vector, drone_view.first, drone_view.second, updated_map);

    return dispensed;
}

void ReinforcementLearningHandler::InitFires() {
        last_distance_to_fire_ = std::numeric_limits<double>::max();

        this->startFires(parameters_.fire_percentage_);
        for(auto &drone : *drones_) {
            std::pair<std::vector<std::vector<int>>, std::vector<std::vector<int>>> drone_view = gridmap_->GetDroneView(drone);
            // Calculate the number of fires in the drone's view
            int fires = drone->DroneSeesFire();
            last_near_fires_ = fires;
        }
}

double ReinforcementLearningHandler::CalculateReward(bool drone_in_grid, bool fire_extinguished, bool drone_terminal, int out_of_area_counter, int near_fires, double max_distance, double distance_to_fire) const {
    double reward = 0;
    if (fire_extinguished) {
        if (near_fires == 0) {
            // all fires in sight were extinguished
            reward += 1;
        } else {
            // a fire was extinguished
            reward += 0.5;
        }
        if(drone_terminal)
            reward += 5;
    }

    // Boundary Penalty
    double ooa_factor = -0.0005 * out_of_area_counter;
    reward += ooa_factor;
    // Check the boundaries of the network
    if (!drone_in_grid) {
        reward += (-0.02 + ooa_factor) * max_distance;
        if(drone_terminal) {
            reward += -0.1;
        }
    }

    // Staying Alive Reward
    reward += pow(0.1, (0.000000000000000000001+gridmap_->PercentageBurned()/0.05));

    // Fire Proximity Reward
    if (last_near_fires_ == 0 && near_fires > 0) {
        reward += 0.2; // TODO we need to check for firespread here
    }
    if (near_fires > 0) {
        reward += 0.005 * near_fires;
    }
    if ((near_fires < last_near_fires_) && !fire_extinguished){
        reward += -0.01;
    }
    if (near_fires > last_near_fires_){
        reward += 0.2;
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
    }
    
    // if (water_dispensed)
    //     reward += -0.1;

    // The environment reached terminal state, this means the map is burned too much
    if(drone_terminal && !fire_extinguished && drone_in_grid){
        reward += -0.1;
    }
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

            bool drone_dispensed_water = StepDrone(i, speed_x, speed_y, water_dispense);

            std::pair<int, int> drone_position = drones_->at(i)->GetGridPosition();
            bool drone_in_grid = gridmap_->IsPointInGrid(drone_position.first, drone_position.second);

            // Calculate distance to nearest fire, dirty maybe change that later(lol never gonna happen)
            double distance_to_fire = drones_->at(i)->FindNearestFireDistance();

            double max_distance = 0;
            if (!drone_in_grid) {
                drones_->at(i)->IncrementOutOfAreaCounter();
                std::pair<double, double> pos = drones_->at(i)->GetLastState().GetPositionNorm();
                double max_distance1 = 0;
                double max_distance2 = 0;
                if (pos.first < 0 || pos.second < 0) {
                    max_distance1 = abs(std::min(pos.first, pos.second));
                } else if (pos.first > 1 || pos.second > 1) {
                    max_distance2 = std::max(pos.first, pos.second) - 1;
                }
                max_distance = std::max(max_distance1, max_distance2);
            } else {
                drones_->at(i)->ResetOutOfAreaCounter();
            }
            int out_of_area_counter = drones_->at(i)->GetLastState().CountOutsideArea();

            int near_fires = drones_->at(i)->DroneSeesFire();
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

            double reward = CalculateReward(drone_in_grid, drone_dispensed_water, terminals[i],
                                            out_of_area_counter, near_fires, max_distance, distance_to_fire);

            rewards.push_back(reward);
            // For displaying the rewards in the GUI
            rewards_.put(static_cast<float>(reward));

            if (!terminals[i]) {
                last_distance_to_fire_ = distance_to_fire;
                last_near_fires_ = near_fires;
            } else {
                auto buffer = rewards_.getBuffer();
                float sum = std::accumulate(buffer.begin(), buffer.end(), 0.0f);
                all_rewards_.push_back(sum / buffer.size());
                last_distance_to_fire_ = std::numeric_limits<double>::max();
                last_near_fires_ = 0;
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
