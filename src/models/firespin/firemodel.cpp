//
// Created by nex on 08.06.23.
//

#include "firemodel.h"

std::shared_ptr<FireModel> FireModel::instance_ = nullptr;

FireModel::FireModel(std::shared_ptr<SDL_Renderer> renderer, int mode) {
    // Find the project root directory
    auto start_path = std::filesystem::current_path();
    auto project_root = find_project_root(start_path);
    std::filesystem::path  dataset_path;
    if (project_root) {
        dataset_path = *project_root / "assets" / "dataset" / "CLMS_CLCplus_RASTER_2018_010m_eu_03035_V1_1.tif";
        std::cout << "Project root found: " << project_root->string() << std::endl;
        std::cout << "Dataset path: " << dataset_path << std::endl;
    } else {
        std::cerr << "Project root not found starting from: " << start_path << std::endl;
    }
    dataset_handler_ = std::make_shared<DatasetHandler>(dataset_path);
    drones_ = std::make_shared<std::vector<std::shared_ptr<DroneAgent>>>();

    if (mode == 1) {
        parameters_.SetNumberOfDrones(0);
        python_code_ = false;
    }

    model_renderer_ = FireModelRenderer::GetInstance(renderer, parameters_);
    wind_ = std::make_shared<Wind>(parameters_);
    this->setupImGui();
    gridmap_ = nullptr;
    running_time_ = 0;
}

void FireModel::ResetGridMap(std::vector<std::vector<int>>* rasterData) {
    gridmap_ = std::make_shared<GridMap>(wind_, parameters_, rasterData);
    model_renderer_->SetGridMap(gridmap_);

    // Init drones
    ResetDrones();

    if (python_code_){
        last_distance_to_fire_ = std::numeric_limits<double>::max();
        int fires = 4;
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

//    model_renderer_->CheckCamera();
    running_time_ = 0;
}

void FireModel::SetUniformRasterData() {
    current_raster_data_.clear();
    current_raster_data_ = std::vector<std::vector<int>>(parameters_.GetGridNx(), std::vector<int>(parameters_.GetGridNy(), GENERIC_UNBURNED));
    parameters_.map_is_uniform_ = true;
}

void FireModel::FillRasterWithEnum() {
    current_raster_data_.clear();
    current_raster_data_ = std::vector<std::vector<int>>(parameters_.GetGridNx(), std::vector<int>(parameters_.GetGridNy(), GENERIC_UNBURNED));
    int total_cells = parameters_.GetGridNx() * parameters_.GetGridNy();
    // int num_classes = CELL_STATE_COUNT - 1;
    int num_classes = 11;

    int per_class = total_cells / num_classes;
    int extra = total_cells % num_classes;

    int current_class = 1;
    int count = 0;

    for(int i = 0; i < parameters_.GetGridNx(); ++i) {
        for(int j = 0; j < parameters_.GetGridNy(); ++j) {
            current_raster_data_[i][j] = current_class;
            if (++count == per_class + (current_class < extra)) {
                count = 0;
                ++current_class;
            }
        }
    }
}

void FireModel::Update() {
    // Simulation time step update
    running_time_ += parameters_.GetDt();
    // Update the fire particles and the cell states
    gridmap_->UpdateParticles();
    gridmap_->UpdateCells();
}

std::vector<std::deque<std::shared_ptr<State>>> FireModel::GetObservations() {
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

bool FireModel::MoveDroneByAngle(int drone_idx, double netout_speed, double netout_angle, int water_dispense) {
    drones_->at(drone_idx)->MoveByAngle(netout_speed, netout_angle);
    bool dispensed = false;
    if (water_dispense == 1) {
        dispensed = drones_->at(drone_idx)->DispenseWater(*gridmap_);
    }

    std::pair<std::vector<std::vector<int>>, std::vector<std::vector<int>>> drone_view = gridmap_->GetDroneView(drones_->at(drone_idx));
    std::vector<std::vector<int>> updated_map = gridmap_->GetUpdatedMap(drones_->at(drone_idx), drone_view.second);
    drones_->at(drone_idx)->Update(netout_speed, netout_angle, drone_view.first, drone_view.second, updated_map);

    return dispensed;
}

bool FireModel::MoveDrone(int drone_idx, double speed_x, double speed_y, int water_dispense) {
    drones_->at(drone_idx)->Move(speed_x, speed_y);
    bool dispensed = false;
    if (water_dispense == 1) {
        dispensed = drones_->at(drone_idx)->DispenseWater(*gridmap_);
    }

    std::pair<std::vector<std::vector<int>>, std::vector<std::vector<int>>> drone_view = gridmap_->GetDroneView(drones_->at(drone_idx));
    std::vector<std::vector<int>> updated_map = gridmap_->GetUpdatedMap(drones_->at(drone_idx), drone_view.second);
    drones_->at(drone_idx)->Update(speed_x, speed_y, drone_view.first, drone_view.second, updated_map);

    return dispensed;
}

bool FireModel::AgentIsRunning() {
    return agent_is_running_;
}

void FireModel::ResetDrones() {
    drones_->clear();
    for (int i = 0; i < parameters_.GetNumberOfDrones(); ++i) {
        auto newDrone = std::make_shared<DroneAgent>(model_renderer_->GetRenderer(), gridmap_->GetRandomPointInGrid(), parameters_, i);
        std::pair<std::vector<std::vector<int>>, std::vector<std::vector<int>>> drone_view = gridmap_->GetDroneView(newDrone);
        newDrone->Initialize(drone_view.first, drone_view.second, std::make_pair(gridmap_->GetCols(), gridmap_->GetRows()), parameters_.GetCellSize());
        drones_->push_back(newDrone);
    }
}

double FireModel::CalculateReward(bool drone_in_grid, bool fire_extinguished, bool drone_terminal, int water_dispensed, int near_fires, double max_distance, double distance_to_fire) {
    double reward = 0;

    // check the boundaries of the network
    if (!drone_in_grid) {
        reward += -2 * max_distance;
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
        //int dist_fires = last_near_fires_ - near_fires; 
        // tricky because if the agent flies towards a big fire thats desireable but we dont
        // want the agent to wait for fires to spawn

        // if (water_dispensed)
        //     reward += -0.1;
    }
    return reward;
}

std::tuple<std::vector<std::deque<std::shared_ptr<State>>>, std::vector<double>, std::vector<bool>, std::pair<bool, bool>> FireModel::Step(std::vector<std::shared_ptr<Action>> actions) {

    if (gridmap_ != nullptr) {
        std::vector<std::deque<std::shared_ptr<State>>> next_observations;
        std::vector<bool> terminals;
        rewards_.clear();
        // TODO this is dirty
        bool drone_died = false;
        bool all_fires_ext = false;
        // Move the drones and get the next_observation
        for (int i = 0; i < (*drones_).size(); ++i) {
            double speed_x = std::dynamic_pointer_cast<DroneAction>(actions[i])->GetSpeedX(); // change this to "real" speed
            double speed_y = std::dynamic_pointer_cast<DroneAction>(actions[i])->GetSpeedY();
//            std::cout << "Drone " << i << " is moving with speed: " << speed_x << ", " << speed_y << std::endl;
            int water_dispense = std::dynamic_pointer_cast<DroneAction>(actions[i])->GetWaterDispense();
            bool drone_dispensed_water = MoveDrone(i, speed_x, speed_y, water_dispense);
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
                ResetGridMap(&current_raster_data_);
            }

            if(gridmap_->PercentageBurned() > 0.30) {
//                std::cout << "Percentage burned: " << gridmap_->PercentageBurned() << " resetting GridMap" << std::endl;
                ResetGridMap(&current_raster_data_);
                terminals[i] = true;
                drone_died = true;
            } else if (!gridmap_->IsBurning()) {
//                std::cout << "Fire is extinguished, resetting GridMap" << std::endl;
                ResetGridMap(&current_raster_data_);
                terminals[i] = true;
                all_fires_ext = true;
            }
            double reward = CalculateReward(drone_in_grid, drone_dispensed_water, terminals[i],
                                water_dispense, near_fires, max_distance, distance_to_fire);
            rewards_.push_back(reward);
            if (!terminals[i]) {
                last_distance_to_fire_ = distance_to_fire;
                last_near_fires_ = near_fires;
            }
        }

        return {next_observations, rewards_, terminals, std::make_pair(drone_died, all_fires_ext)};

    }
    return {};
}

void FireModel::Render() {
    model_renderer_->Render(drones_);
    model_renderer_->DrawArrow(wind_->GetCurrentAngle() * 180 / M_PI + 45);
}

void FireModel::Initialize() {
    // Outdated code
}

void FireModel::SetWidthHeight(int width, int height) {
    // Outdated code
}

void FireModel::Reset() {
    // Outdated code
}

//** ########################################################################
//                                 ImGui Stuff
//   ######################################################################
// **//

FireModel::~FireModel() {

}

void FireModel::setupImGui() {
    imgui_handler_ = std::make_shared<ImguiHandler>(python_code_, parameters_);
    imgui_handler_->onResetDrones = [this]() {ResetDrones();};
    imgui_handler_->onResetGridMap = [this](std::vector<std::vector<int>>* rasterData) {ResetGridMap(rasterData);};
    imgui_handler_->onFillRasterWithEnum = [this]() {FillRasterWithEnum();};
    imgui_handler_->onSetUniformRasterData = [this]() {SetUniformRasterData();};
    imgui_handler_->onMoveDrone = [this](int drone_idx, double speed_x, double speed_y, int water_dispense) {return MoveDrone(drone_idx, speed_x, speed_y, water_dispense);};
}

void FireModel::ImGuiRendering(std::function<void(bool &, bool &, int &)> controls, bool &update_simulation,
                             bool &render_simulation, int &delay) {
    imgui_handler_->ShowControls(controls, update_simulation, render_simulation, delay);
    imgui_handler_->ImGuiModelMenu(model_renderer_, current_raster_data_);
    imgui_handler_->Config(gridmap_, drones_, model_renderer_, dataset_handler_,
                           current_raster_data_, running_time_, rewards_, wind_, agent_is_running_);
    imgui_handler_->ShowPopups(gridmap_, current_raster_data_);
}

void FireModel::HandleEvents(SDL_Event event, ImGuiIO *io) {
    imgui_handler_->HandleEvents(event, io, gridmap_, model_renderer_, dataset_handler_, current_raster_data_, agent_is_running_);
}

void FireModel::ImGuiSimulationSpeed() {
    imgui_handler_->ImGuiSimulationSpeed();
}

