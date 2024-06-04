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

void ReinforcementLearningHandler::ResetDrones() {
    drones_->clear();
    for (int i = 0; i < parameters_.GetNumberOfDrones(); ++i) {
        auto newDrone = std::make_shared<DroneAgent>(model_renderer_->GetRenderer(), gridmap_->GetRandomPointInGrid(), parameters_, i);
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
        int cells = gridmap_->GetNumCells();
        int fires = static_cast<int>(cells * 0.01);
        std::pair<int, int> drone_position = drones_->at(0)->GetGridPosition();
        if(gridmap_->CellCanIgnite(drone_position.first, drone_position.second))
            gridmap_->IgniteCell(drone_position.first, drone_position.second);
        for(int i = 0; i < fires;) {
            std::pair<int, int> point = gridmap_->GetRandomPointInGrid();
            if(gridmap_->CellCanIgnite(point.first, point.second)){
                gridmap_->IgniteCell(point.first, point.second);
                i++;
            }
        }
        last_near_fires_ = fires + 1;
}

double ReinforcementLearningHandler::CalculateReward(bool drone_in_grid, bool fire_extinguished, bool drone_terminal, int water_dispensed, int near_fires, double max_distance, double distance_to_fire) {
    double reward = 0;

    // check the boundaries of the network
    if (!drone_in_grid) {
        reward += -50 * max_distance;
        if(drone_terminal) {
            reward -= 10;
            return reward;
        }
    }

    // Fire was discovered
    if (last_near_fires_ == 0 && near_fires > 0) {
        reward += 2;
    }

    if (fire_extinguished) {
        if (drone_terminal) {
            reward += 10;
            return reward;
        }
            // all fires in sight were extinguished
        else if (near_fires == 0) {
            reward += 5;
        }
            // a fire was extinguished
        else {
            reward += 1;
        }
    } else {
        // if last_distance or last_distance_to_fire_ is very large, dismiss the reward
        if (!(last_distance_to_fire_ > 1000000 || distance_to_fire > 1000000))
        {
            double delta_distance = last_distance_to_fire_ - distance_to_fire;
            //These high values occure when fire spreads and gets extinguished
//            if (delta_distance < -10 || delta_distance > 10) {
//                std::cout << "Delta distance: " << delta_distance << std::endl;
//                std::cout << "Last distance: " << last_distance_to_fire_ << std::endl;
//                std::cout << "Current distance: " << distance_to_fire << std::endl;
//                std::cout << "" << std::endl;
//            }
            reward += 0.05 * delta_distance;
        }
        // if (water_dispensed)
        //     reward += -0.1;
    }
    return reward;
}

std::tuple<std::vector<std::deque<std::shared_ptr<State>>>, std::vector<double>, std::vector<bool>, std::pair<bool, bool>> ReinforcementLearningHandler::Step(std::vector<std::shared_ptr<Action>> actions) {

    if (gridmap_ != nullptr) {
        std::vector<std::deque<std::shared_ptr<State>>> next_observations;
        std::vector<bool> terminals;
        std::vector<double> rewards;
        // TODO this is dirty
        bool drone_died = false;
        bool all_fires_ext = false;
        // Move the drones and get the next_observation
        for (int i = 0; i < (*drones_).size(); ++i) {
            double speed_x = std::dynamic_pointer_cast<DroneAction>(actions[i])->GetSpeedX(); // change this to "real" speed
            double speed_y = std::dynamic_pointer_cast<DroneAction>(actions[i])->GetSpeedY();
//            std::cout << "Drone " << i << " is moving with speed: " << speed_x << ", " << speed_y << std::endl;
            int water_dispense = std::dynamic_pointer_cast<DroneAction>(actions[i])->GetWaterDispense();
            bool drone_dispensed_water = StepDrone(i, speed_x, speed_y, water_dispense);
            // bool drone_dispensed_water = MoveDroneByAngle(i, speed_x, speed_y, water_dispense);

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
            std::deque<DroneState> drone_states = drones_->at(i)->GetStates();
            std::deque<std::shared_ptr<State>> shared_states;
            for (auto &state : drone_states) {
                shared_states.push_back(std::make_shared<DroneState>(state));
            }
            next_observations.push_back(shared_states);

            int near_fires = drones_->at(i)->DroneSeesFire();
            terminals.push_back(false);

            // Check if drone is out of area for too long, if so, reset it
            if (drones_->at(i)->GetOutOfAreaCounter() > 15) {
                terminals[i] = true;
                drone_died = true;
                // Delete Drone and create new one
                // drones_->erase(drones_->begin() + i);
                // auto newDrone = std::make_shared<DroneAgent>(model_renderer_->GetRenderer(),gridmap_->GetRandomPointInGrid(), parameters_, i);
                // std::pair<std::vector<std::vector<int>>, std::vector<std::vector<int>>> drone_view = gridmap_->GetDroneView(newDrone);
                // newDrone->Initialize(drone_view.first, drone_view.second, std::make_pair(gridmap_->GetCols(), gridmap_->GetRows()), parameters_.GetCellSize());
                // // Insert new drone at the same position
                // drones_->insert(drones_->begin() + i, newDrone);
            }

            if(gridmap_->PercentageBurned() > 0.30) {
//                std::cout << "Percentage burned: " << gridmap_->PercentageBurned() << " resetting GridMap" << std::endl;
                terminals[i] = true;
                drone_died = true;
            } else if (!gridmap_->IsBurning()) {
//                std::cout << "Fire is extinguished, resetting GridMap" << std::endl;
                terminals[i] = true;
                all_fires_ext = true;
            }
            double reward = CalculateReward(drone_in_grid, drone_dispensed_water, terminals[i],
                                            water_dispense, near_fires, max_distance, distance_to_fire);
            rewards.push_back(reward);
            rewards_.put(static_cast<float>(reward));

            if (!terminals[i]) {
                last_distance_to_fire_ = distance_to_fire;
                last_near_fires_ = near_fires;
            } else {
                std::cout << "Drone " << i << " terminated" << std::endl;
                auto buffer = rewards_.getBuffer();
                float sum = std::accumulate(buffer.begin(), buffer.end(), 0.0f);
                all_rewards_.push_back(sum / buffer.size());
                rewards_.reset();
            }
        }

        return {next_observations, rewards, terminals, std::make_pair(drone_died, all_fires_ext)};

    }
    std::cout << "Taking a step into nothingness..." << std::endl;
    return {};
}
