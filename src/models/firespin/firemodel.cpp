//
// Created by nex on 08.06.23.
//

#include "firemodel.h"

std::shared_ptr<FireModel> FireModel::instance_ = nullptr;

std::optional<std::filesystem::path> find_project_root(const std::filesystem::path& start) {
    std::filesystem::path current = start;

    // Search for .git directory, which is a good indicator of a project root
    while (current.has_relative_path()) {
        if (std::filesystem::exists(current / ".git")) {
            return current;
        }
        if (current.parent_path() == current) {
            break;
        }
        current = current.parent_path();
    }
    return std::nullopt;
}

FireModel::FireModel(std::shared_ptr<SDL_Renderer> renderer, int mode) {
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

void FireModel::HandleEvents(SDL_Event event, ImGuiIO *io) {
    // SDL Events
    if (event.type == SDL_MOUSEBUTTONDOWN && !io->WantCaptureMouse && event.button.button == SDL_BUTTON_LEFT) {
        int x, y;
        SDL_GetMouseState(&x, &y);
        std::pair<int, int> gridPos = model_renderer_->ScreenToGridPosition(x, y);
        x = gridPos.first;
        y = gridPos.second;
        if (x >= 0 && x < gridmap_->GetCols() && y >= 0 && y < gridmap_->GetRows()) {
            if(gridmap_->At(x, y).CanIgnite())
                gridmap_->IgniteCell(x, y);
            else if(gridmap_->GetCellState(x, y) == CellState::GENERIC_BURNING)
                gridmap_->ExtinguishCell(x, y);
        }
    } else if (event.type == SDL_MOUSEWHEEL && !io->WantCaptureMouse) {
        if (event.wheel.y > 0) // scroll up
        {
            model_renderer_->ApplyZoom(1.1);
        }
        else if (event.wheel.y < 0) // scroll down
        {
            model_renderer_->ApplyZoom(0.9);
        }
    } else if (event.type == SDL_MOUSEMOTION && !io->WantCaptureMouse) {
        int x, y;
        Uint32 mouseState = SDL_GetMouseState(&x, &y);
        if (mouseState & SDL_BUTTON(SDL_BUTTON_RIGHT)) // Middle mouse button is pressed
        {
            model_renderer_->ChangeCameraPosition(-event.motion.xrel, -event.motion.yrel);
        }
    } else if (event.type == SDL_MOUSEBUTTONDOWN && event.button.button == SDL_BUTTON_MIDDLE && !io->WantCaptureMouse) {
        int x, y;
        SDL_GetMouseState(&x, &y);
        std::pair<int, int> cell_pos = model_renderer_->ScreenToGridPosition(x, y);
        if (cell_pos.first >= 0 && cell_pos.first < gridmap_->GetCols() && cell_pos.second >= 0 && cell_pos.second < gridmap_->GetRows()) {
            popups_.insert(cell_pos);
            popup_has_been_opened_.insert({cell_pos, false});
        }
    } else if (event.type == SDL_WINDOWEVENT) {
        if (event.window.event == SDL_WINDOWEVENT_SIZE_CHANGED) {
            model_renderer_->ResizeEvent();
        }
    } else if (event.type == SDL_KEYDOWN && parameters_.GetNumberOfDrones() == 1 && !agent_is_running_) {
        if (event.key.keysym.sym == SDLK_w)
            MoveDrone(0, 0, parameters_.GetDroneSpeed(0.1), 0);
            // MoveDroneByAngle(0, 0.25, 0, 0);
        if (event.key.keysym.sym == SDLK_s)
            MoveDrone(0, 0, -parameters_.GetDroneSpeed(0.1), 0);
            // MoveDroneByAngle(0, -0.25, 0, 0);
        if (event.key.keysym.sym == SDLK_a)
            MoveDrone(0, -parameters_.GetDroneSpeed(0.1), 0, 0);
            // MoveDroneByAngle(0, 0, -0.25, 0);
        if (event.key.keysym.sym == SDLK_d)
            MoveDrone(0, parameters_.GetDroneSpeed(0.1), 0, 0);
            // MoveDroneByAngle(0, 0, 0.25, 0);
        if (event.key.keysym.sym == SDLK_SPACE)
            MoveDrone(0, 0, 0, 1);
            // MoveDroneByAngle(0, 0, 0, 1);
    }
    // Browser Events
    // TODO: Eventloop only gets executed when Application is in Focus. Fix this.
    if (dataset_handler_ != nullptr) {
        if (dataset_handler_->NewDataPointExists() && browser_selection_flag_) {
            std::vector<std::vector<int>> rasterData;
            dataset_handler_->LoadRasterDataFromJSON(rasterData);
            current_raster_data_.clear();
            current_raster_data_ = rasterData;
            browser_selection_flag_ = false;
            parameters_.map_is_uniform_ = false;
            ResetGridMap(&current_raster_data_);
        }
    }
}

void FireModel::OpenBrowser(std::string url) {
    std::string command;

#if defined(_WIN32)
    command = std::string("start ");
#elif defined(__APPLE__)
    command = std::string("open ");
#elif defined(__linux__)
    command = std::string("xdg-open ");
#endif

    // system call
    if(system((command + url).c_str()) == -1) {
        std::cerr << "Error opening URL: " << url << std::endl;
    }
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

void FireModel::ShowControls(std::function<void(bool &, bool &, int &)> controls, bool &update_simulation, bool &render_simulation, int &delay) {
    if (show_controls_)
        controls(update_simulation, render_simulation, delay);
}

void FireModel::ImGuiSimulationSpeed(){
    ImGui::Text("Simulation Speed");
    ImGui::SliderScalar("dt", ImGuiDataType_Double, &parameters_.dt_, &parameters_.min_dt_, &parameters_.max_dt_, "%.8f", 1.0f);
    ImGui::Spacing();
}

void FireModel::ImGuiModelMenu() {
    if (model_startup_) {
        if (ImGui::BeginMainMenuBar()) {
            if (ImGui::BeginMenu("File")) {
                if (ImGui::MenuItem("Open Browser")) {
                    std::string url = "http://localhost:3000/map.html";
                    OpenBrowser(url);
                }
                if (ImGui::MenuItem("Load Map from Disk")) {
                    // Open File Dialog
                    open_file_dialog_ = true;
                    // Load Map
                    load_map_from_disk_ = true;
                }
                if (ImGui::MenuItem("Load Map from Browser Selection"))
                    browser_selection_flag_ = true;
                if (ImGui::MenuItem("Load Uniform Map")) {
                    SetUniformRasterData();
                    ResetGridMap(&current_raster_data_);
                }
                if (ImGui::MenuItem("Load Map with Classes")) {
                    FillRasterWithEnum();
                    ResetGridMap(&current_raster_data_);
                }
                if (ImGui::MenuItem("Save Map")) {
                    // Open File Dialog
                    open_file_dialog_ = true;
                    // Save Map
                    save_map_to_disk_ = true;
                }
                if (ImGui::MenuItem("Reset GridMap"))
                    ResetGridMap(&current_raster_data_);
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("Model Controls")) {
                ImGui::MenuItem("Emit Convective Particles", NULL, &parameters_.emit_convective_);
                ImGui::MenuItem("Emit Radiation Particles", NULL, &parameters_.emit_radiation_);
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("View")) {
                ImGui::MenuItem("Show Controls", NULL, &show_controls_);
                ImGui::MenuItem("Show Simulation Analysis", NULL, &show_model_analysis_);
                ImGui::MenuItem("Show Parameter Config", NULL, &show_model_parameter_config_);
                if(ImGui::MenuItem("Render Grid", NULL, &parameters_.render_grid_))
                    model_renderer_->SetFullRedraw();
                ImGui::EndMenu();
            }
            if (python_code_) {
                if (ImGui::BeginMenu("RL Controls")) {
                    ImGui::MenuItem("Show RL Controls", NULL, &show_rl_controls_);
                    ImGui::MenuItem("Show Drone Analysis", NULL, &show_drone_analysis_);
                    ImGui::EndMenu();
                }
            }
            if (ImGui::BeginMenu("Help")) {
                ImGui::MenuItem("ImGui Help", NULL, &show_demo_window_);
                ImGui::EndMenu();
            }
            ImGui::EndMainMenuBar();
        }
    }
}

void FireModel::Config() {
    if(show_demo_window_)
        ImGui::ShowDemoWindow(&show_demo_window_);

    if (!ImGuiOnStartup()) {

        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(5, 7));

        if(show_model_parameter_config_)
            ShowParameterConfig();

        if (show_model_analysis_) {
            ImGui::Begin("Simulation Analysis");
            ImGui::Spacing();
            ImGui::Text("Number of particles: %d", gridmap_->GetNumParticles());
            ImGui::Text("Number of cells: %d", gridmap_->GetNumCells());
            ImGui::Text("Running Time: %s", formatTime(running_time_).c_str());
            ImGui::Text("Height: %.2fkm | Width: %.2fkm", gridmap_->GetRows() * parameters_.GetCellSize() / 1000,
                                                              gridmap_->GetCols() * parameters_.GetCellSize() / 1000);
            ImGui::End();
        }

        if (show_rl_controls_ && python_code_) {
            ImGui::Begin("RL Controls");
            bool button_color = false;
            if (agent_is_running_) {
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.35f, 0.6f, 0.85f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.45f, 0.7f, 0.95f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.25f, 0.5f, 0.75f, 1.0f));
                button_color = true;
            }
            if (ImGui::Button(agent_is_running_ ? "Stop Training" : "Start Training")) {
                agent_is_running_ = !agent_is_running_;
            }
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Click to %s Reinforcement Learning.", agent_is_running_ ? "stop" : "start");
            if (button_color) {
                ImGui::PopStyleColor(3);
            }
            ImGui::SameLine();
            if (ImGui::Button("Reset Drones")) {
                ResetDrones();
            }
            ImGui::End();
        }

        if (show_drone_analysis_ && python_code_) {
            ImGuiWindowFlags window_flags = ImGuiWindowFlags_HorizontalScrollbar | ImGuiWindowFlags_AlwaysVerticalScrollbar;
            ImGui::Begin("Drone Analysis", NULL, window_flags);
            ImGui::Spacing();

            static int current_drone_index = 0;
            if (drones_->size() > 0) {
                // Create an array of drone names
                std::vector<const char*> drone_names;
                for (int i = 0; i < drones_->size(); ++i) {
                    drone_names.push_back(std::to_string((*drones_)[i]->GetId()).c_str());
                }

                // Create a combo box for selecting a drone
                ImGui::Combo("Select Drone", &current_drone_index, &drone_names[0], drones_->size());

                // Get the currently selected drone
                auto& selected_drone = (*drones_)[current_drone_index];

                ImGui::Text("Drone %d", selected_drone->GetId());
                ImGui::Text("Grid Position: (%d, %d)", selected_drone->GetGridPosition().first,
                            selected_drone->GetGridPosition().second);
                ImGui::Text("Out of Area Counter: %d", selected_drone->GetOutOfAreaCounter());
                ImGui::Text("Real Position: (%.2f, %.2f)", selected_drone->GetRealPosition().first,
                            selected_drone->GetRealPosition().second);
                ImGui::Text("Network Input:");
                ImGui::Text("Relative Position: %.2f, %.2f", selected_drone->GetLastState().GetPositionNorm().first,
                                                             selected_drone->GetLastState().GetPositionNorm().second);
                ImGui::Text("Velocity (x,y) m/s: %.2f, %.2f", selected_drone->GetLastState().GetVelocityNorm().first,
                                                                    selected_drone->GetLastState().GetVelocityNorm().second);
                ImGui::Text("Terrain");
                // Calculate the size and position of each cell
                float cell_size = 15.0f;
                ImVec2 cursor_pos = ImGui::GetCursorScreenPos();
                ImVec2 first_map_origin = cursor_pos;

                std::vector<std::vector<int>> terrain = selected_drone->GetLastState().get_terrain();


                // Draw each cell
                for (int y = 0; y < terrain.size(); ++y) {
                    for (int x = 0; x < terrain[y].size(); ++x) {
                        ImVec4 color = model_renderer_->GetMappedColor(terrain[y][x]);
                        ImVec2 p_min = ImVec2(cursor_pos.x + x * cell_size, cursor_pos.y + y * cell_size);
                        ImVec2 p_max = ImVec2(cursor_pos.x + (x + 1) * cell_size, cursor_pos.y + (y + 1) * cell_size);
                        ImGui::GetWindowDrawList()->AddRectFilled(p_min, p_max, IM_COL32(color.x * 255, color.y * 255, color.z * 255, color.w * 255));
                    }
                }

                std::vector<std::vector<int>> fire_status = selected_drone->GetLastState().get_fire_status();

                // Set the cursor to the position of the second map
                ImVec2 second_map_origin = ImVec2(first_map_origin.x + terrain[0].size() * cell_size + 10, first_map_origin.y);
                ImGui::SetCursorScreenPos(second_map_origin);
                cursor_pos = ImGui::GetCursorScreenPos();

                ImGui::Text("FireStatus");
                for (int y = 0; y < fire_status.size(); ++y) {
                    for (int x = 0; x < fire_status[y].size(); ++x) {
                        ImVec4 color = fire_status[y][x] == 1 ? ImVec4(255 / 255.f, 0, 0, 255 / 255.f) : ImVec4(0, 0, 0, 255 / 255.f);
                        ImVec2 p_min = ImVec2(cursor_pos.x + x * cell_size, cursor_pos.y + y * cell_size);
                        ImVec2 p_max = ImVec2(cursor_pos.x + (x + 1) * cell_size, cursor_pos.y + (y + 1) * cell_size);
                        ImGui::GetWindowDrawList()->AddRectFilled(p_min, p_max, IM_COL32(color.x * 255, color.y * 255, color.z * 255, color.w * 255));
                    }
                }

                std::vector<std::vector<int>> map = selected_drone->GetLastState().get_map();

                ImVec2 third_map_origin = ImVec2(first_map_origin.x, first_map_origin.y + terrain.size() * cell_size + 10);
                ImGui::SetCursorScreenPos(third_map_origin);
                cell_size = 5.0f;
                cursor_pos = ImGui::GetCursorScreenPos();

                ImGui::Text("Map");
                for (int x = 0; x < map.size(); ++x) {
                    for (int y = 0; y < map[x].size(); ++y) {
                        ImVec4 color = map[x][y] == 1 ? ImVec4(255 / 255.f, 0, 0, 255 / 255.f) : ImVec4(0, 0, 0, 255 / 255.f);
                        ImVec2 p_min = ImVec2(cursor_pos.x + x * cell_size, cursor_pos.y + y * cell_size);
                        ImVec2 p_max = ImVec2(cursor_pos.x + (x + 1) * cell_size, cursor_pos.y + (y + 1) * cell_size);
                        ImGui::GetWindowDrawList()->AddRectFilled(p_min, p_max, IM_COL32(color.x * 255, color.y * 255, color.z * 255, color.w * 255));
                    }
                }
                if (ImGui::Begin("Rewards List")) {
                    if (ImGui::BeginChild("RewardChild", ImVec2(0, 200), true)) {  // 200 is the height; adjust as required
                        for (size_t i = 0; i < rewards_.size(); ++i) {
                            ImGui::Text("Reward %zu: %f", i, rewards_[i]);
                        }
                    }
                    ImGui::EndChild();
                }
                ImGui::End();

            } else {
                ImGui::Text("No drones available.");
            }

            ImGui::Spacing();
            ImGui::End();
        }

        if (open_file_dialog_) {
            std::string filePathName;
            // open Dialog Simple
            IGFD::FileDialogConfig config;
	        config.path = "../maps";
            ImGuiFileDialog::Instance()->OpenDialog("ChooseFileDlgKey", "Choose File", ".tif", config);

            // display
            if (ImGuiFileDialog::Instance()->Display("ChooseFileDlgKey")) {
                // action if OK
                if (ImGuiFileDialog::Instance()->IsOk()) {
                    filePathName = ImGuiFileDialog::Instance()->GetFilePathName();
                    std::string filePath = ImGuiFileDialog::Instance()->GetCurrentPath();
                    if (load_map_from_disk_){
                        dataset_handler_->LoadMap(filePathName);
                        std::vector<std::vector<int>> rasterData;
                        dataset_handler_->LoadMapDataset(rasterData);
                        current_raster_data_.clear();
                        current_raster_data_ = rasterData;
                        parameters_.map_is_uniform_ = false;
                        load_map_from_disk_ = false;
                        init_gridmap_ = true;
                    }
                    else if (save_map_to_disk_) {
                        dataset_handler_->SaveRaster(filePathName);
                        save_map_to_disk_ = false;
                    }
                }
                // close
                ImGuiFileDialog::Instance()->Close();
                open_file_dialog_ = false;
            }
        }
        ImGui::PopStyleVar();
    }
}

bool FireModel::ImGuiOnStartup() {
    if (!model_startup_) {
        int width, height;
        SDL_GetRendererOutputSize(model_renderer_->GetRenderer().get(), &width, &height);
        ImVec2 window_size = ImVec2(400, 100);
        ImGui::SetNextWindowSize(window_size);
        ImVec2 appWindowPos = ImVec2((width - window_size.x) * 0.5f, (height - window_size.y) * 0.5f);
        ImGui::SetNextWindowPos(appWindowPos);

        ImGui::Begin("Startup Mode Selection", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove);
        ImGui::Spacing();

        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.45f, 0.6f, 0.85f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.45f, 0.7f, 0.95f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.45f, 0.5f, 0.75f, 1.0f));

        bool still_no_init = true;
        if (ImGui::Button("Uniform Vegetation", ImVec2(-1, 0))) {
            SetUniformRasterData();
            ResetGridMap(&current_raster_data_);
            still_no_init = false;
            model_startup_ = true;
            show_controls_ = true;
            show_model_parameter_config_ = true;
        }
        ImGui::Separator();
        if (ImGui::Button("Load from File", ImVec2(-1, 0))) {
            model_startup_ = true;
            show_controls_ = true;
            show_model_parameter_config_ = true;
            open_file_dialog_ = true;
            load_map_from_disk_ = true;
        }
        ImGui::PopStyleColor(3);
        ImGui::End();
        return still_no_init;
    } else {
        return false;
    }
}

void FireModel::ShowParameterConfig() {
    ImGui::Begin("Simulation Parameters");
    if (parameters_.emit_convective_) {
        ImGui::SeparatorText("Virtual Particles");
        if (ImGui::TreeNodeEx("##Virtual Particles",
                              ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_SpanAvailWidth)) {
            ImGui::Text("Particle Lifetime");
            ImGui::SliderScalar("tau_mem", ImGuiDataType_Double, &parameters_.virtualparticle_tau_mem_,
                                &parameters_.tau_min, &parameters_.tau_max, "%.3f", 1.0f);
            ImGui::Text("Hotness");
            ImGui::SliderScalar("Y_st", ImGuiDataType_Double, &parameters_.virtualparticle_y_st_,
                                &parameters_.min_Y_st_, &parameters_.max_Y_st_, "%.3f", 1.0f);
            ImGui::Text("Ignition Threshold");
            ImGui::SliderScalar("Y_lim", ImGuiDataType_Double, &parameters_.virtualparticle_y_lim_,
                                &parameters_.min_Y_lim_, &parameters_.max_Y_lim_, "%.3f", 1.0f);
            ImGui::Text("Height of Emission");
            ImGui::SliderScalar("Lt", ImGuiDataType_Double, &parameters_.Lt_, &parameters_.min_Lt_,
                                &parameters_.max_Lt_, "%.3f", 1.0f);
            ImGui::Text("Scaling Factor");
            ImGui::SliderScalar("Fl", ImGuiDataType_Double, &parameters_.virtualparticle_fl_, &parameters_.min_Fl_,
                                &parameters_.max_Fl_, "%.3f", 1.0f);
            ImGui::Text("Constant");
            ImGui::SliderScalar("C0", ImGuiDataType_Double, &parameters_.virtualparticle_c0_, &parameters_.min_C0_,
                                &parameters_.max_C0_, "%.3f", 1.0f);
            ImGui::TreePop();
            ImGui::Spacing();
        }
    }

    if (parameters_.emit_radiation_) {
        ImGui::SeparatorText("Radiation Particles");
        if(ImGui::TreeNodeEx("##Radiation Particles", ImGuiTreeNodeFlags_SpanAvailWidth)){
            ImGui::Text("Radiation Hotness");
            ImGui::SliderScalar("Y_st_r", ImGuiDataType_Double, &parameters_.radiationparticle_y_st_, &parameters_.min_Y_st_, &parameters_.max_Y_st_, "%.3f", 1.0f);
            ImGui::Text("Radiation Ignition Threshold");
            ImGui::SliderScalar("Y_lim_r", ImGuiDataType_Double, &parameters_.radiationparticle_y_lim_, &parameters_.min_Y_lim_, &parameters_.max_Y_lim_, "%.3f", 1.0f);
//            ImGui::Text("Radiation Length");
//            ImGui::SliderScalar("Lr", ImGuiDataType_Double, &parameters_.radiationparticle_Lr_, &parameters_.min_Lr_, &parameters_.max_Lr_, "%.3f", 1.0f);
            ImGui::TreePop();
            ImGui::Spacing();
        }
    }

    if (parameters_.map_is_uniform_) {
        ImGui::SeparatorText("Cell (Terrain)");
        if(ImGui::TreeNodeEx("##CellTerrain", ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_SpanAvailWidth)) {
            ImGui::Spacing();
            ImGui::Text("Cell Size");
            //Colored Text in grey on the same line
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "(?)");
            if(ImGui::IsItemHovered())
                ImGui::SetTooltip("always press [Reset GridMap] manually after changing these values");
            ImGui::SliderScalar("##Cell Size", ImGuiDataType_Double, &parameters_.cell_size_, &parameters_.min_cell_size_, &parameters_.max_cell_size_, "%.3f", 1.0f);
            ImGui::Text("Cell Ignition Threshold");
            ImGui::SliderScalar("##Cell Ignition Threshold", ImGuiDataType_Double, &parameters_.cell_ignition_threshold_, &parameters_.min_ignition_threshold_, &parameters_.max_ignition_threshold_, "%.3f", 1.0f);
            ImGui::Text("Cell Burning Duration");
            ImGui::SliderScalar("##Cell Burning Duration", ImGuiDataType_Double, &parameters_.cell_burning_duration_, &parameters_.min_burning_duration_, &parameters_.max_burning_duration_, "%.3f", 1.0f);
            ImGui::TreePop();
            ImGui::Spacing();
        }
    }

    ImGui::SeparatorText("Wind");
    if(ImGui::TreeNodeEx("##Wind", ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_SpanAvailWidth)) {
        double min_angle_degree = 0;
        double max_angle_degree = (2* M_PI);
        bool update_wind = false;
        ImGui::Text("Wind Speed");
        if(ImGui::SliderScalar("##Wind Speed", ImGuiDataType_Double, &parameters_.wind_uw_, &parameters_.min_Uw_, &parameters_.max_Uw_, "%.3f", 1.0f)) update_wind = true;
        ImGui::Text("A");
        if(ImGui::SliderScalar("##A", ImGuiDataType_Double, &parameters_.wind_a_, &parameters_.min_A_, &parameters_.max_A_, "%.3f", 1.0f)) update_wind = true;
        ImGui::Text("Wind Angle");
        if(ImGui::SliderScalar("##Wind Angle", ImGuiDataType_Double, &parameters_.wind_angle_, &min_angle_degree, &max_angle_degree, "%.1f", 1.0f)) update_wind = true;
        if (update_wind)
            wind_->UpdateWind();
        ImGui::TreePop();
        ImGui::Spacing();
    }
    ImGui::End();
}

void FireModel::ShowPopups() {
    // Start the Dear ImGui frame
    for (auto it = popups_.begin(); it != popups_.end();) {
        char popupId[20];
        snprintf(popupId, sizeof(popupId), "Cell %d %d", it->first, it->second);

        if (!popup_has_been_opened_[*it]){
            ImGui::SetNextWindowPos(ImGui::GetMousePos());
            popup_has_been_opened_[*it] = true;
        }

        if (ImGui::Begin(popupId, nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoTitleBar)) {
            ImGui::TextColored(ImVec4(0.5f, 0.5f, 1.0f, 1.0f), "Cell %d %d", it->first, it->second);
            ImGui::SameLine();
            float windowWidth = ImGui::GetWindowContentRegionMax().x - ImGui::GetWindowContentRegionMin().x;
            float buttonWidth = ImGui::CalcTextSize("X").x + ImGui::GetStyle().FramePadding.x;
            ImGui::SetCursorPosX(windowWidth - buttonWidth);
            if (ImGui::Button("X")) {
                popup_has_been_opened_.erase(*it);  // Removes the popup from the map.
                it = popups_.erase(it);  // Removes the popup from the set and gets the next iterator.
                ImGui::End();
                continue;  // Skip the rest of the loop for this iteration.
            }
            gridmap_->ShowCellInfo(it->first, it->second);
        }
        ImGui::End();
        ++it;  // Go to the next popup in the set.
    }
    if(init_gridmap_) {
        init_gridmap_ = false;
        ResetGridMap(&current_raster_data_);
    }
}

FireModel::~FireModel() {

}

