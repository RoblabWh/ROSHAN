//
// Created by nex on 09.05.24.
//

#include "firemodel_imgui.h"

ImguiHandler::ImguiHandler(Mode mode, FireModelParameters &parameters) : parameters_(parameters), mode_(mode) {

}

void ImguiHandler::ImGuiSimulationControls(std::shared_ptr<GridMap> gridmap, std::vector<std::vector<int>> &current_raster_data,
                                           std::shared_ptr<FireModelRenderer> model_renderer, bool &update_simulation,
                                           bool &render_simulation, int &delay, float framerate, double running_time) {
    if (show_controls_) {
        ImGuiWindowFlags window_flags =
                ImGuiWindowFlags_HorizontalScrollbar | ImGuiWindowFlags_AlwaysVerticalScrollbar;
        ImGui::Begin("Simulation Controls", &show_controls_, window_flags);
        bool button_color = false;
        if (update_simulation) {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.35f, 0.6f, 0.85f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.45f, 0.7f, 0.95f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.25f, 0.5f, 0.75f, 1.0f));
            button_color = true;
        }
        if (ImGui::Button(update_simulation ? "Stop Simulation" : "Start Simulation")) {
            update_simulation = !update_simulation;
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Click to %s the simulation.", update_simulation ? "stop" : "start");
        if (button_color) {
            ImGui::PopStyleColor(3);
        }
        ImGui::SameLine();

        button_color = false;
        if (render_simulation) {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.35f, 0.6f, 0.85f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.45f, 0.7f, 0.95f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.25f, 0.5f, 0.75f, 1.0f));
            button_color = true;
        }
        if (ImGui::Button(render_simulation ? "Stop Rendering" : "Start Rendering")) {
            render_simulation = !render_simulation;
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Click to %s rendering the simulation.", render_simulation ? "stop" : "start");
        if (button_color) {
            ImGui::PopStyleColor(3);
        }
        ImGui::SameLine();
        if(ImGui::Button("Reset GridMap"))
            onResetGridMap(&current_raster_data);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Resets the GridMap to the initial state of the currently loaded map.");
        if (ImGui::BeginTabBar("SimStatus")){
            if(ImGui::BeginTabItem("Simulation Info")){
                ImGui::Text("Simulation Analysis");
                std::string analysis_text;
                analysis_text += "Number of particles: " + std::to_string(gridmap->GetNumParticles()) + "\n";
                analysis_text += "Number of cells: " + std::to_string(gridmap->GetNumCells()) + "\n";
                analysis_text += "Running Time: " + formatTime(running_time) + "\n";
                analysis_text +=
                        "Height: " + std::to_string(gridmap->GetRows() * parameters_.GetCellSize() / 1000) + "km | ";
                analysis_text += "Width: " + std::to_string(gridmap->GetCols() * parameters_.GetCellSize() / 1000) + "km";
                ImGui::TextWrapped("%s", analysis_text.c_str());
                ImGui::Spacing();
                ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / framerate, framerate);
                ImGui::EndTabItem();
            }
            if(ImGui::BeginTabItem("Simulation Speed")){
                ImGui::Spacing();
                ImGui::Text("Simulation Delay");
                ImGui::SliderInt("Delay (ms)", &delay, 0, 500);
                ImGui::Spacing();
                ImGui::Text("Simulation Speed");
                ImGui::SliderScalar("dt", ImGuiDataType_Double, &parameters_.dt_, &parameters_.min_dt_, &parameters_.max_dt_, "%.8f", 1.0f);
                ImGui::EndTabItem();
            }
            if(ImGui::BeginTabItem("Render Options")) {
                if(ImGui::Checkbox("Render Grid", &parameters_.render_grid_)){
                    model_renderer->SetFullRedraw();
                }
                if(ImGui::Checkbox("Render Noise", &parameters_.has_noise_)){
                    model_renderer->SetInitCellNoise();
                    model_renderer->SetFullRedraw();
                }
                if(ImGui::Checkbox("Lingering", &parameters_.lingering_)){
                    model_renderer->SetFullRedraw();
                }
                ImGui::EndTabItem();
            }
        }
        ImGui::EndTabBar();
        ImGui::End();
    }
}

void ImguiHandler::ImGuiModelMenu(std::vector<std::vector<int>> &current_raster_data) {
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
                    open_file_dialog_ = true;
                    save_map_to_disk_ = true;
                }
                if (ImGui::MenuItem("Reset GridMap"))
                    onResetGridMap(&current_raster_data);
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("Model Particles")) {
                ImGui::MenuItem("Emit Convective Particles", NULL, &parameters_.emit_convective_);
                ImGui::MenuItem("Emit Radiation Particles", NULL, &parameters_.emit_radiation_);
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("View")) {
                ImGui::MenuItem("Show Controls", NULL, &show_controls_);
                if (mode_ == Mode::GUI_RL)
                    ImGui::MenuItem("Show RL Controls", NULL, &show_rl_status_);
                ImGui::MenuItem("Show Parameter Config", NULL, &show_model_parameter_config_);
                if(parameters_.has_noise_)
                    ImGui::MenuItem("Show Noise Config", NULL, &show_noise_config_);
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("Help")) {
                ImGui::MenuItem("ImGui Help", NULL, &show_demo_window_);
                ImGui::EndMenu();
            }
            ImGui::EndMainMenuBar();
        }
    }
}

void ImguiHandler::Config(std::shared_ptr<FireModelRenderer> model_renderer,
                          std::vector<std::vector<int>> &current_raster_data,
                          std::shared_ptr<Wind> wind) {
    if (show_demo_window_)
        ImGui::ShowDemoWindow(&show_demo_window_);

    if (!ImGuiOnStartup(model_renderer, current_raster_data)) {

        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(5, 7));

        if (show_model_parameter_config_)
            ShowParameterConfig(wind);

        ImGui::PopStyleVar();
    }
}

void ImguiHandler::PyConfig(std::vector<float> rewards, int rewards_pos,std::vector<float> all_rewards,
                            bool &agent_is_running, std::string &user_input, std::string &model_output,
                            std::shared_ptr<std::vector<std::shared_ptr<DroneAgent>>> drones,
                            const std::shared_ptr<FireModelRenderer>& model_renderer) {
    if (show_rl_status_ && mode_ == Mode::GUI_RL) {
        ImGuiWindowFlags window_flags =
                ImGuiWindowFlags_HorizontalScrollbar | ImGuiWindowFlags_AlwaysVerticalScrollbar;
        ImGui::Begin("Reinforcement Learning Status", &show_rl_status_, window_flags);
        ImGui::Spacing();

        // RL Controls, always showing
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

        std::string console;
        bool show_file_dialog = false;
        std::string selected_key;
        py::dict rl_status = onGetRLStatus();

        for (auto item : rl_status) {
            auto key = item.first.cast<std::string>();
            auto value = py::reinterpret_borrow<py::object>(item.second);

            if (py::isinstance<py::bool_>(value)) {
                bool value_bool = value.cast<bool>();
                ImGui::Text("%s: %s", key.c_str(), value_bool ? "true" : "false");
            } else if(py::isinstance<py::str>(value)) {
                if (key == "console"){
                    console = value.cast<std::string>();
                } else if (key == "model_path" || key == "model_name"){
                    auto value_str = value.cast<std::string>();
                    if (ImGui::Selectable(key.c_str())) {
                        open_file_dialog_ = true;
                        model_path_selection_ = true;
                        path_key_ = key;
                    }
                    ImGui::SameLine();
                    ImGui::Text("%s", value_str.c_str());
                } else {
                    auto value_str = value.cast<std::string>();
                    ImGui::Text("%s: %s", key.c_str(), value_str.c_str());
                }
            } else if (py::isinstance<py::int_>(value)) {
                int value_int = value.cast<int>();
                ImGui::Text("%s: %d", key.c_str(), value_int);
            }
        }
        if (ImGui::BeginTabBar("RLStatus")){
            if (ImGui::BeginTabItem("Info")) {
                ImGui::BeginChild("scrolling", ImVec2(0, ImGui::GetTextLineHeightWithSpacing() * 10), true, ImGuiWindowFlags_HorizontalScrollbar);
                ImGui::TextUnformatted(console.c_str());
                ImGui::EndChild();
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("ROSHAN-AI")) {
                static char input_text[512] = "";
                ImGui::Text("Ask ROSHAN-AI a question:");
                ImGui::InputText("##input", input_text, IM_ARRAYSIZE(input_text));
                if (ImGui::Button("Send")) {
                    // Do something with the text
                    user_input = input_text;
                    // Clear the input text
                    memset(input_text, 0, sizeof(input_text));
                }
                ImGui::BeginChild("scrolling", ImVec2(0, ImGui::GetTextLineHeightWithSpacing() * 15), true, ImGuiWindowFlags_HorizontalScrollbar);
                ImGui::TextWrapped("%s", model_output.c_str());
                ImGui::EndChild();
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Drone Analysis")){
                ImGui::Checkbox("Show Input Images", &show_input_images_);
                ImGui::BeginChild("scrolling", ImVec2(0, ImGui::GetTextLineHeightWithSpacing() * 40), true, ImGuiWindowFlags_HorizontalScrollbar);

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
                    ImGui::Text("Out of Area Counter: %d", selected_drone->GetOutOfAreaCounter());
                    ImGui::Text("Real Position: (%.2f, %.2f)", selected_drone->GetRealPosition().first,
                                selected_drone->GetRealPosition().second);
                    ImGui::Spacing();

                    // Display network input information
                    ImGui::Text("Network Input:");
                    ImGui::BulletText("Relative Position: (%.2f, %.2f)",
                                      selected_drone->GetLastState().GetPositionNorm().first,
                                      selected_drone->GetLastState().GetPositionNorm().second);
                    ImGui::BulletText("Velocity (x,y) m/s: %.2f, %.2f",
                                      selected_drone->GetLastState().GetVelocityNorm().first,
                                      selected_drone->GetLastState().GetVelocityNorm().second);
                    // Move to the next column
                    ImGui::NextColumn();

                    if (show_input_images_){
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
                    }


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
                ImGui::EndChild();
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Env Controls")){
                ImGui::Text("Fire Percentage");
                ImGui::SliderInt("%", &parameters_.fire_percentage_, 0, 100);
                if (ImGui::Button("Start some fires")) {
                    this->startFires(parameters_.fire_percentage_);
                }
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Click to start some fires");
                ImGui::Spacing();
                ImGui::EndTabItem();
            }
        }
        ImGui::EndTabBar();
        ImGui::End();
    }
}

void ImguiHandler::FileHandling(std::shared_ptr<DatasetHandler> dataset_handler, std::vector<std::vector<int>> &current_raster_data){

    // Show the file dialog for loading and saving maps
    if (open_file_dialog_) {
        // open Dialog Simple
        IGFD::FileDialogConfig config;
        std::optional<std::string> vFilters;
        std::string vTitle;
        std::string vKey;
        if (model_path_selection_){
            config.path = "../models";
            if (path_key_ == "model_path"){
                vTitle = "Choose Model Path";
                vFilters.reset();
                vKey = "ChooseFolderDlgKey";
            }
            else if (path_key_ == "model_name"){
                vTitle = "Choose Model Name";
                vKey = "ChooseFileDlgKey";
                vFilters = ".pth";
            }
        } else {
            vTitle = "Choose File or Filename";
            config.path = "../maps";
            vFilters = ".tif";
            vKey = "ChooseFileDlgKey";
        }

        if (vFilters) {
            ImGuiFileDialog::Instance()->OpenDialog(vKey, vTitle, vFilters->c_str(), config);
        } else {
            ImGuiFileDialog::Instance()->OpenDialog(vKey, vTitle, nullptr, config);
        }

        // display
        if (ImGuiFileDialog::Instance()->Display(vKey)) {
            // action if OK
            if (ImGuiFileDialog::Instance()->IsOk()) {
                std::string filePathName = ImGuiFileDialog::Instance()->GetFilePathName();
                std::string filePath = ImGuiFileDialog::Instance()->GetCurrentPath();
                std::string fileName = ImGuiFileDialog::Instance()->GetCurrentFileName();
                if (load_map_from_disk_){
                    dataset_handler->LoadMap(filePathName);
                    std::vector<std::vector<int>> rasterData;
                    dataset_handler->LoadMapDataset(rasterData);
                    current_raster_data.clear();
                    current_raster_data = rasterData;
                    parameters_.map_is_uniform_ = false;
                    load_map_from_disk_ = false;
                    init_gridmap_ = true;
                    if (!model_startup_){
                        show_controls_ = true;
                        model_startup_ = true;
                    }
                }
                else if (save_map_to_disk_) {
                    dataset_handler->SaveRaster(filePathName);
                    save_map_to_disk_ = false;
                }
                else if (model_path_selection_) {
                    py::dict rl_status = onGetRLStatus();
                    if(path_key_ == "model_path")
                        rl_status[py::str(path_key_)] = py::str(filePath);
                    else if(path_key_ == "model_name")
                        rl_status[py::str(path_key_)] = py::str(fileName);
                    onSetRLStatus(rl_status);
                    model_path_selection_ = false;
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
            // TODO I don't really know why this works, but it does. If I don't call this function, everything works fine
            //  until I reset the GridMap again, which works fine as well. But if I then try to zoom in it crashes.
            onResetGridMap(&current_raster_data);
            still_no_init = false;
            model_startup_ = true;
            show_controls_ = true;
            show_model_parameter_config_ = true;
        }
        ImGui::Separator();
        if (ImGui::Button("Load from File", ImVec2(-1, 0))) {
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
    ImGuiWindowFlags window_flags =
            ImGuiWindowFlags_HorizontalScrollbar | ImGuiWindowFlags_AlwaysVerticalScrollbar;
    ImGui::Begin("Simulation Parameters", &show_model_parameter_config_, window_flags);
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
            if(show_noise_config_) {
                static int noise_level = 1;
                static int noise_size = 1;
                ImGui::SliderInt("Noise Level", &noise_level, 1, 200);
                ImGui::SliderInt("Noise Size", &noise_size, 1, 200);
                if (ImGui::Button("Reset Map")) {
                    onResetGridMap(&current_raster_data);
                }
                if (ImGui::Button("Add Noise")) {
                    CellState state = gridmap->GetCellState(it->first, it->second);
                    onSetNoise(state, noise_level, noise_size);
                }
            }
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
