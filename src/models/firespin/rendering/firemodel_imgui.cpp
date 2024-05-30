//
// Created by nex on 09.05.24.
//

#include "firemodel_imgui.h"

ImguiHandler::ImguiHandler(bool python_code, FireModelParameters &parameters) : parameters_(parameters) {
    python_code_ = python_code;
}

void ImguiHandler::ShowControls(std::function<void(bool &, bool &, int &)> controls, bool &update_simulation, bool &render_simulation, int &delay) {
    if (show_controls_)
        controls(update_simulation, render_simulation, delay);
}

void ImguiHandler::ImGuiSimulationSpeed(){
    ImGui::Text("Simulation Speed");
    ImGui::SliderScalar("dt", ImGuiDataType_Double, &parameters_.dt_, &parameters_.min_dt_, &parameters_.max_dt_, "%.8f", 1.0f);
    ImGui::Spacing();
}

void ImguiHandler::ImGuiModelMenu(std::shared_ptr<FireModelRenderer> model_renderer, std::vector<std::vector<int>> &current_raster_data) {
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
                    onSetUniformRasterData();
                    onResetGridMap(&current_raster_data);
                }
                if (ImGui::MenuItem("Load Map with Classes")) {
                    onFillRasterWithEnum();
                    onResetGridMap(&current_raster_data);
                }
                if (ImGui::MenuItem("Save Map")) {
                    // Open File Dialog
                    open_file_dialog_ = true;
                    // Save Map
                    save_map_to_disk_ = true;
                }
                if (ImGui::MenuItem("Reset GridMap"))
                    onResetGridMap(&current_raster_data);
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
                    model_renderer->SetFullRedraw();
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

void ImguiHandler::Config(std::shared_ptr<GridMap> gridmap, std::shared_ptr<FireModelRenderer> model_renderer,
                          std::vector<std::vector<int>> &current_raster_data, double running_time,
                          std::shared_ptr<Wind> wind) {
    if (show_demo_window_)
        ImGui::ShowDemoWindow(&show_demo_window_);

    if (!ImGuiOnStartup(model_renderer, current_raster_data)) {

        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(5, 7));

        if (show_model_parameter_config_)
            ShowParameterConfig(wind);

        if (show_model_analysis_) {
            ImGui::Begin("Simulation Analysis");
            ImGui::Spacing();
            std::string analysis_text;
            analysis_text += "Number of particles: " + std::to_string(gridmap->GetNumParticles()) + "\n";
            analysis_text += "Number of cells: " + std::to_string(gridmap->GetNumCells()) + "\n";
            analysis_text += "Running Time: " + formatTime(running_time) + "\n";
            analysis_text +=
                    "Height: " + std::to_string(gridmap->GetRows() * parameters_.GetCellSize() / 1000) + "km | ";
            analysis_text += "Width: " + std::to_string(gridmap->GetCols() * parameters_.GetCellSize() / 1000) + "km";
            ImGui::TextWrapped("%s", analysis_text.c_str());
            ImGui::End();
        }



        ImGui::PopStyleVar();
    }
}

void ImguiHandler::PyConfig(std::vector<float> rewards, int rewards_pos,std::vector<float> all_rewards,
                            bool &agent_is_running, std::string &user_input, std::string &model_output,
                            std::shared_ptr<std::vector<std::shared_ptr<DroneAgent>>> drones,
                            std::shared_ptr<FireModelRenderer> model_renderer) {
    if (show_rl_controls_ && python_code_) {
        static char input_text[512] = "";
        ImGui::Begin("RL Controls", NULL, ImGuiWindowFlags_AlwaysVerticalScrollbar);
        bool button_color = false;
        if (agent_is_running) {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.35f, 0.6f, 0.85f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.45f, 0.7f, 0.95f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.25f, 0.5f, 0.75f, 1.0f));
            button_color = true;
        }
        if (ImGui::Button(agent_is_running ? "Stop Training" : "Start Training")) {
            agent_is_running = !agent_is_running;
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Click to %s Reinforcement Learning.", agent_is_running ? "stop" : "start");
        if (button_color) {
            ImGui::PopStyleColor(3);
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset Drones")) {
            onResetDrones();
        }

        ImGui::Text("Ask ROSHAN-AI a question:");
        ImGui::InputText("##input", input_text, IM_ARRAYSIZE(input_text));
        if (ImGui::Button("Send")) {
            // Do something with the text
            user_input = input_text;
            // Clear the input text
            memset(input_text, 0, sizeof(input_text));
        }
        ImGui::TextWrapped("%s", model_output.c_str());
        ImGui::End();
    }

    if (show_drone_analysis_ && python_code_) {
        ImGuiWindowFlags window_flags =
                ImGuiWindowFlags_HorizontalScrollbar | ImGuiWindowFlags_AlwaysVerticalScrollbar;
        ImGui::Begin("Drone Analysis", &show_drone_analysis_, window_flags);
        ImGui::Spacing();

        static int current_drone_index = 0;

        if (drones->size() > 0) {
            // Create an array of drone names
            std::vector<std::string> drone_names;
            for (const auto &drone: *drones) {
                drone_names.push_back(std::to_string(drone->GetId()));
            }

            // Create a combo box for selecting a drone
            ImGui::Combo("Select Drone", &current_drone_index, [](void *data, int idx, const char **out_text) {
                auto &names = *static_cast<std::vector<std::string> *>(data);
                if (idx < 0 || idx >= names.size()) return false;
                *out_text = names[idx].c_str();
                return true;
            }, &drone_names, drone_names.size());

            // Get the currently selected drone
            auto &selected_drone = (*drones)[current_drone_index];

            // Use columns to separate text and images
            ImGui::Columns(2, nullptr, true);

            // Display drone information
            ImGui::Text("Drone %d", selected_drone->GetId());
            std::pair<double, double> pos = selected_drone->GetLastState().GetPositionNorm();
            ImGui::Text("Grid Position: (%f, %f)", pos.first, pos.second);
            ImGui::Text("Out of Area Counter: %d", selected_drone->GetOutOfAreaCounter());
            ImGui::Text("Real Position: (%.2f, %.2f)", selected_drone->GetRealPosition().first,
                        selected_drone->GetRealPosition().second);
            ImGui::Spacing();

            // Display network input information
            ImGui::Text("Network Input:");
            ImGui::BulletText("Relative Position: %.2f, %.2f",
                              selected_drone->GetLastState().GetPositionNorm().first,
                              selected_drone->GetLastState().GetPositionNorm().second);
            ImGui::BulletText("Velocity (x,y) m/s: %.2f, %.2f",
                              selected_drone->GetLastState().GetVelocityNorm().first,
                              selected_drone->GetLastState().GetVelocityNorm().second);

            // Move to the next column
            ImGui::NextColumn();

            // Draw terrain map
            ImVec2 title_origin = ImGui::GetCursorScreenPos();
            ImGui::Text("Terrain");
            ImVec2 map_origin = ImGui::GetCursorScreenPos();
            DrawGrid(selected_drone->GetLastState().get_terrain(), model_renderer, 5.0f);

            // Fire Status Title
            ImVec2 firedet_title_origin = ImVec2(
                    title_origin.x + selected_drone->GetLastState().get_terrain()[0].size() * 5.0f + 10,
                    title_origin.y);
            ImGui::SetCursorScreenPos(firedet_title_origin);
            ImGui::Text("Fire Status");
            // Draw fire status map
            ImVec2 firedet_map_origin = ImVec2(
                    map_origin.x + selected_drone->GetLastState().get_terrain()[0].size() * 5.0f + 10,
                    map_origin.y);
            ImGui::SetCursorScreenPos(firedet_map_origin);
            DrawGrid(selected_drone->GetLastState().get_fire_status(), model_renderer, 5.0f, true);

            // Perception Map Title
            ImVec2 perception_title_origin = ImVec2(
                    firedet_title_origin.x + selected_drone->GetLastState().get_terrain().size() * 5.0f + 10,
                    firedet_title_origin.y);
            ImGui::SetCursorScreenPos(perception_title_origin);
            ImGui::Text("Perception Map");

            // Draw perception map
            ImVec2 perception_map_origin = ImVec2(
                    firedet_map_origin.x + selected_drone->GetLastState().get_terrain().size() * 5.0f + 10,
                    firedet_map_origin.y);
            ImGui::SetCursorScreenPos(perception_map_origin);
            DrawGrid(selected_drone->GetLastState().get_map(), model_renderer, 2.0f, true);

            // Move back to one column to column 1
            ImGui::Columns(1);
            ImGui::Spacing();
            ImGui::Text("Episodic Rewards");
            // Plot rewards using ImGui::PlotLines
            if (!rewards.empty()) {
                this->DrawBuffer(rewards, rewards_pos);
            } else {
                ImGui::Text("No rewards data available.");
            }
            ImGui::Text("Accumulated Episodic Rewards");
            if (!all_rewards.empty()) {
                this->DrawBuffer(all_rewards, all_rewards.size());
            } else {
                ImGui::Text("No total rewards data available.");
            }
        } else {
            ImGui::Text("No drones available.");
        }

        ImGui::End();
    }
}

void ImguiHandler::FileHandling(std::shared_ptr<DatasetHandler> dataset_handler, std::vector<std::vector<int>> &current_raster_data){

    // Show the file dialog for loading and saving maps
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
                    dataset_handler->LoadMap(filePathName);
                    std::vector<std::vector<int>> rasterData;
                    dataset_handler->LoadMapDataset(rasterData);
                    current_raster_data.clear();
                    current_raster_data = rasterData;
                    parameters_.map_is_uniform_ = false;
                    load_map_from_disk_ = false;
                    init_gridmap_ = true;
                }
                else if (save_map_to_disk_) {
                    dataset_handler->SaveRaster(filePathName);
                    save_map_to_disk_ = false;
                }
            }
            // close
            ImGuiFileDialog::Instance()->Close();
            open_file_dialog_ = false;
        }
    }
}

bool ImguiHandler::ImGuiOnStartup(std::shared_ptr<FireModelRenderer> model_renderer, std::vector<std::vector<int>> &current_raster_data) {
    if (!model_startup_) {
        int width, height;
        SDL_GetRendererOutputSize(model_renderer->GetRenderer().get(), &width, &height);
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
            onSetUniformRasterData();
            onResetGridMap(&current_raster_data);
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

void ImguiHandler::ShowParameterConfig(std::shared_ptr<Wind> wind) {
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
            wind->UpdateWind();
        ImGui::TreePop();
        ImGui::Spacing();
    }
    ImGui::End();
}

void ImguiHandler::ShowPopups(std::shared_ptr<GridMap> gridmap, std::vector<std::vector<int>> &current_raster_data) {
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
            gridmap->ShowCellInfo(it->first, it->second);
        }
        ImGui::End();
        ++it;  // Go to the next popup in the set.
    }
    if(init_gridmap_) {
        init_gridmap_ = false;
        onResetGridMap(&current_raster_data);
    }
}

void ImguiHandler::HandleEvents(SDL_Event event, ImGuiIO *io, std::shared_ptr<GridMap> gridmap, std::shared_ptr<FireModelRenderer> model_renderer,
                                std::shared_ptr<DatasetHandler> dataset_handler, std::vector<std::vector<int>> &current_raster_data, bool agent_is_running) {
    // SDL Events
    if (event.type == SDL_MOUSEBUTTONDOWN && !io->WantCaptureMouse && event.button.button == SDL_BUTTON_LEFT) {
        int x, y;
        SDL_GetMouseState(&x, &y);
        std::pair<int, int> gridPos = model_renderer->ScreenToGridPosition(x, y);
        x = gridPos.first;
        y = gridPos.second;
        if (x >= 0 && x < gridmap->GetCols() && y >= 0 && y < gridmap->GetRows()) {
            if(gridmap->At(x, y).CanIgnite())
                gridmap->IgniteCell(x, y);
            else if(gridmap->GetCellState(x, y) == CellState::GENERIC_BURNING)
                gridmap->ExtinguishCell(x, y);
        }
    } else if (event.type == SDL_MOUSEWHEEL && !io->WantCaptureMouse) {
        if (event.wheel.y > 0) // scroll up
        {
            model_renderer->ApplyZoom(1.1);
        }
        else if (event.wheel.y < 0) // scroll down
        {
            model_renderer->ApplyZoom(0.9);
        }
    } else if (event.type == SDL_MOUSEMOTION && !io->WantCaptureMouse) {
        int x, y;
        Uint32 mouseState = SDL_GetMouseState(&x, &y);
        if (mouseState & SDL_BUTTON(SDL_BUTTON_RIGHT)) // Middle mouse button is pressed
        {
            model_renderer->ChangeCameraPosition(-event.motion.xrel, -event.motion.yrel);
        }
    } else if (event.type == SDL_MOUSEBUTTONDOWN && event.button.button == SDL_BUTTON_MIDDLE && !io->WantCaptureMouse) {
        int x, y;
        SDL_GetMouseState(&x, &y);
        std::pair<int, int> cell_pos = model_renderer->ScreenToGridPosition(x, y);
        if (cell_pos.first >= 0 && cell_pos.first < gridmap->GetCols() && cell_pos.second >= 0 && cell_pos.second < gridmap->GetRows()) {
            popups_.insert(cell_pos);
            popup_has_been_opened_.insert({cell_pos, false});
        }
    } else if (event.type == SDL_WINDOWEVENT) {
        if (event.window.event == SDL_WINDOWEVENT_SIZE_CHANGED) {
            model_renderer->ResizeEvent();
        }
    } else if (event.type == SDL_KEYDOWN && parameters_.GetNumberOfDrones() == 1 && !agent_is_running && !io->WantTextInput) {
        if (event.key.keysym.sym == SDLK_w)
            onMoveDrone(0, 0, parameters_.GetDroneSpeed(0.1), 0);
        // MoveDroneByAngle(0, 0.25, 0, 0);
        if (event.key.keysym.sym == SDLK_s)
            onMoveDrone(0, 0, -parameters_.GetDroneSpeed(0.1), 0);
        // MoveDroneByAngle(0, -0.25, 0, 0);
        if (event.key.keysym.sym == SDLK_a)
            onMoveDrone(0, -parameters_.GetDroneSpeed(0.1), 0, 0);
        // MoveDroneByAngle(0, 0, -0.25, 0);
        if (event.key.keysym.sym == SDLK_d)
            onMoveDrone(0, parameters_.GetDroneSpeed(0.1), 0, 0);
        // MoveDroneByAngle(0, 0, 0.25, 0);
        if (event.key.keysym.sym == SDLK_SPACE)
            onMoveDrone(0, 0, 0, 1);
        // MoveDroneByAngle(0, 0, 0, 1);
    }
    // Browser Events
    // TODO: Eventloop only gets executed when Application is in Focus. Fix this.
    if (dataset_handler != nullptr) {
        if (dataset_handler->NewDataPointExists() && browser_selection_flag_) {
            std::vector<std::vector<int>> rasterData;
            dataset_handler->LoadRasterDataFromJSON(rasterData);
            current_raster_data.clear();
            current_raster_data = rasterData;
            browser_selection_flag_ = false;
            parameters_.map_is_uniform_ = false;
            onResetGridMap(&current_raster_data);
        }
    }
}

void ImguiHandler::DrawBuffer(std::vector<float> buffer, int buffer_pos) {
    // Calculate the minimum and maximum values from the data
    float min_value = *std::min_element(buffer.begin(), buffer.end());
    float max_value = *std::max_element(buffer.begin(), buffer.end());
    // Calculate the position and size of the graph
    //ImVec2 graph_pos = ImGui::GetCursorScreenPos();
    ImGui::PlotLines("", buffer.data(), static_cast<int>(buffer.size()), 0, nullptr, min_value, max_value, ImVec2(0, 150));
    // Get the draw list
    ImDrawList* draw_list = ImGui::GetWindowDrawList();

    // Get the position and size of the plot
    ImVec2 graph_pos = ImGui::GetItemRectMin(); // Top-left corner of the plot
    ImVec2 graph_size = ImGui::GetItemRectSize(); // Size of the plot

    // Ensure pos_index is within bounds
    if (buffer_pos >= 0 && buffer_pos < buffer.size()) {
        // Calculate the x-position of the data point in screen coordinates
        float x = graph_pos.x + ((float)buffer_pos / (buffer.size() - 1)) * graph_size.x;

        // Draw a vertical line at the data point position
        draw_list->AddLine(ImVec2(x, graph_pos.y), ImVec2(x, graph_pos.y + graph_size.y), IM_COL32(255, 0, 0, 255), 2.0f);

        // Optionally, add a text label at the data point position
        char label[32];
        snprintf(label, sizeof(label), "Value: %.2f", buffer[buffer_pos]);
        draw_list->AddText(ImVec2(x + 10, graph_pos.y), IM_COL32(255, 255, 255, 255), label);
    }
}

void ImguiHandler::DrawGrid(const std::vector<std::vector<int>>& grid, std::shared_ptr<FireModelRenderer> renderer, float cell_size, bool is_fire_status) {
    ImVec2 cursor_pos = ImGui::GetCursorScreenPos();
    for (int y = 0; y < grid.size(); ++y) {
        for (int x = 0; x < grid[y].size(); ++x) {
            ImVec4 color = is_fire_status
                           ? (grid[y][x] == 1 ? ImVec4(1.0f, 0.0f, 0.0f, 1.0f) : ImVec4(0.0f, 1.0f, 0.0f, 1.0f))
                           : renderer->GetMappedColor(grid[y][x]);
            ImVec2 p_min = ImVec2(cursor_pos.x + x * cell_size, cursor_pos.y + y * cell_size);
            ImVec2 p_max = ImVec2(cursor_pos.x + (x + 1) * cell_size, cursor_pos.y + (y + 1) * cell_size);
            ImGui::GetWindowDrawList()->AddRectFilled(p_min, p_max, IM_COL32(color.x * 255, color.y * 255, color.z * 255, color.w * 255));
        }
    }
}

void ImguiHandler::OpenBrowser(std::string url) {
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
