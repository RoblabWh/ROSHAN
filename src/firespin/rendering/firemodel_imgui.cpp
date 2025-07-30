//
// Created by nex on 09.05.24.
//

#include "firemodel_imgui.h"

ImguiHandler::ImguiHandler(Mode mode, FireModelParameters &parameters) : parameters_(parameters), mode_(mode) {

}

void ImguiHandler::ImGuiSimulationControls(const std::shared_ptr<GridMap>& gridmap, std::vector<std::vector<int>> &current_raster_data,
                                           const std::shared_ptr<FireModelRenderer>& model_renderer, bool &update_simulation,
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
                analysis_text += "Running Time: " + formatTime(int(running_time)) + "\n";
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
                if(ImGui::Checkbox("Episode Termination Indicator", &parameters_.episode_termination_indicator_)){
                    model_renderer->SetFlashScreen(parameters_.episode_termination_indicator_);
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
                if (parameters_.GetCorineLoaded()) {
                    if (ImGui::MenuItem("Open Browser")) {
                        std::string url = "http://localhost:3000/map.html";
                        OpenBrowser(url);
                    }
                    if (ImGui::MenuItem("Load Map from Browser Selection"))
                        browser_selection_flag_ = true;
                }
                if (ImGui::MenuItem("Load Map from Disk")) {
                    // Open File Dialog
                    open_file_dialog_ = true;
                    // Load Map
                    load_map_from_disk_ = true;
                }
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
                ImGui::MenuItem("Emit Convective Particles", nullptr, &parameters_.emit_convective_);
                ImGui::MenuItem("Emit Radiation Particles", nullptr, &parameters_.emit_radiation_);
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("View")) {
                ImGui::MenuItem("Show Controls", nullptr, &show_controls_);
                if (mode_ == Mode::GUI_RL)
                    ImGui::MenuItem("Show RL Controls", nullptr, &show_rl_status_);
                ImGui::MenuItem("Show Parameter Config", nullptr, &show_model_parameter_config_);
                if(parameters_.has_noise_)
                    ImGui::MenuItem("Show Noise Config", nullptr, &show_noise_config_);
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("Help")) {
                ImGui::MenuItem("ImGui Help", nullptr, &show_demo_window_);
                ImGui::EndMenu();
            }
            ImGui::EndMainMenuBar();
        }
    }
}

void ImguiHandler::Config(const std::shared_ptr<FireModelRenderer>& model_renderer,
                          std::vector<std::vector<int>> &current_raster_data,
                          const std::shared_ptr<Wind>& wind) {
    if (show_demo_window_)
        ImGui::ShowDemoWindow(&show_demo_window_);

    if (!ImGuiOnStartup(model_renderer, current_raster_data)) {

        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(5, 7));

        if (show_model_parameter_config_)
            ShowParameterConfig(wind);

        ImGui::PopStyleVar();
    }
}

void ImguiHandler::RLStatusParser(const py::dict& rl_status) {
    auto rl_mode = rl_status["rl_mode"].cast<std::string>();
    auto model_path = rl_status["model_path"].cast<std::string>();
    auto model_name = rl_status["model_name"].cast<std::string>();
    auto obs_collected = rl_status["obs_collected"].cast<int>();
    auto min_update = rl_status["min_update"].cast<int>();
    auto auto_train = rl_status["auto_train"].cast<bool>();
    auto train_step = rl_status["train_step"].cast<int>();
    auto policy_updates = rl_status["policy_updates"].cast<int>();
    auto objective = rl_status["objective"].cast<double>();
    auto best_objective = rl_status["best_objective"].cast<double>();
    auto current_episode = rl_status["current_episode"].cast<int>();
    ImVec4 color = ImVec4(0.33f, 0.67f, 0.86f, 1.0f);
    ImGui::Separator();
    ImGui::SetWindowFontScale(1.5f);
    ImGui::PushStyleColor(ImGuiCol_Text, color);
    if (ImGui::Selectable("Mode of operation:")) {
        ImGui::OpenPopup("Warning RL Mode");
    }
    ImGui::SetWindowFontScale(1.0f);
    ImGui::PopStyleColor();

    if (ImGui::BeginPopupModal("Warning RL Mode", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::Text("You are about to switch to %s mode.", rl_mode == "train" ? "eval" : "train");
        ImGui::Separator();

        if (ImGui::Button("OK", ImVec2(120, 0))) {
            auto new_value_str = rl_mode == "train" ? "eval" : "train";
            rl_status["rl_mode"] = new_value_str;
            rl_mode = new_value_str;
            auto console = rl_status["console"].cast<std::string>();
            console += "Switched to " + static_cast<std::string>(new_value_str) + " mode.\n";
            rl_status[py::str("console")] = console;
            onSetRLStatus(rl_status);
            ImGui::CloseCurrentPopup();
        }

        ImGui::SameLine();

        if (ImGui::Button("Cancel", ImVec2(120, 0))) {
            ImGui::CloseCurrentPopup();
        }

        ImGui::EndPopup();
    }
    ImGui::SameLine();
    ImGui::SetWindowFontScale(1.5f);
    ImGui::PushStyleColor(ImGuiCol_Text, color);
    ImGui::Text("%s", rl_mode.c_str());
    ImGui::SetWindowFontScale(1.0f);
    ImGui::PopStyleColor();

    ImGui::Separator();
    ImGui::SetWindowFontScale(1.5f);
    ImGui::PushStyleColor(ImGuiCol_Text, color);
    ImGui::Text("Model will be saved under: ");
    ImGui::SetWindowFontScale(1.0f);
    ImGui::PopStyleColor();
    if (ImGui::Selectable("Model Path")) {
        open_file_dialog_ = true;
        model_path_selection_ = true;
        path_key_ = "model_path";
    }
    ImGui::SameLine();
    ImGui::Text(": %s", model_path.c_str());

    if (ImGui::Selectable("Model Name")) {
        open_file_dialog_ = true;
        model_path_selection_ = true;
        path_key_ = "model_name";
    }
    ImGui::SameLine();
    ImGui::Text(": %s", model_name.c_str());

    ImGui::Separator();
    ImGui::SetWindowFontScale(1.5f);
    ImGui::PushStyleColor(ImGuiCol_Text, color);
    ImGui::Text("Hyperparameter");
    ImGui::SetWindowFontScale(1.0f);
    ImGui::PopStyleColor();
    ImGui::Text("Train Step(Policy Updates): %d(%d)", train_step, policy_updates);
    ImGui::Text("Horizon: %d/%d", obs_collected, min_update);
//    ImGui::Text("Auto Train: %s", auto_train ? "true" : "false");
    ImGui::Separator();
    ImGui::SetWindowFontScale(1.5f);
    ImGui::PushStyleColor(ImGuiCol_Text, color);
    ImGui::Text("Overview");
    ImGui::SetWindowFontScale(1.0f);
    ImGui::PopStyleColor();
    ImGui::Text("Current Episode: %d", current_episode);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("The current episode the agent is in. After 100 episodes the objective will be calculated.");
    if (current_episode > 100){
        ImGui::Text("Current Objective: %.2f", objective);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("The Objective is the goal of the agent is trying to reach. It is calculated from the last 100 episodes.\n"
                              "The best Objective might actually go lower sometimes, this behaviour is due to the updates in the policy.");
        ImGui::Text("Best Objective: %.2f", best_objective);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("The best Objective might go lower sometimes, this behaviour is intended and occurs because it is recalculated at policy update.");
    }
    ImGui::Text("Environment Steps before Failure: %d/%d",parameters_.GetCurrentEnvSteps(), parameters_.GetTotalEnvSteps());
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("The number of steps the environment will take before the episode is considered a failure.\n"
                          "This number is calculated by the size of the map and the simulation time:\n"
                          "sqrt(grid_nx_ * grid_nx_ + grid_ny_ * grid_ny_) * (20 / (max_velocity * dt_)");
    ImGui::Separator();
    if(auto_train){
        ImGui::Separator();
        ImGui::SetWindowFontScale(1.5f);
        ImGui::PushStyleColor(ImGuiCol_Text, color);
        ImGui::Text("Auto Training: ON");
        ImGui::SetWindowFontScale(1.0f);
        ImGui::PopStyleColor();
        auto train_episodes = rl_status["train_episodes"].cast<int>();
        auto max_eval = rl_status["max_eval"].cast<int>();
        auto max_train = rl_status["max_train"].cast<int>();
        auto train_episode = rl_status["train_episode"].cast<int>();
        ImGui::Text("Train Episodes: %d/%d", train_episode + 1, train_episodes);
        ImGui::Text("Training Steps before Evaluation: %d", max_train);
        ImGui::Text("Evaluation Episodes: %d", max_eval);
    }
}

void ImguiHandler::PyConfig(std::string &user_input,
                            std::string &model_output,
                            const std::shared_ptr<GridMap>& gridmap,
                            const std::shared_ptr<std::vector<std::shared_ptr<FlyAgent>>>& drones,
                            const std::shared_ptr<FireModelRenderer>& model_renderer) {
    if (show_rl_status_ && mode_ == Mode::GUI_RL && model_startup_) {
        py::dict rl_status = onGetRLStatus();

        ImGuiWindowFlags window_flags =
                ImGuiWindowFlags_HorizontalScrollbar | ImGuiWindowFlags_AlwaysVerticalScrollbar;
        ImGui::Begin("Reinforcement Learning Status", &show_rl_status_, window_flags);
        ImGui::Spacing();

        // RL Controls, always showing
        bool button_color = false;
        if (parameters_.agent_is_running_) {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.35f, 0.6f, 0.85f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.45f, 0.7f, 0.95f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.25f, 0.5f, 0.75f, 1.0f));
            button_color = true;
        }
        auto rl_mode = rl_status["rl_mode"].cast<std::string>();
        auto running_str_start = rl_mode == "train" ? "Start Training" : "Start Evaluation";
        auto running_str_stop = rl_mode == "train" ? "Stop Training" : "Stop Evaluation";
        if (ImGui::Button(parameters_.agent_is_running_ ? running_str_stop : running_str_start)) {
            parameters_.agent_is_running_ = !parameters_.agent_is_running_;
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Click to %s Reinforcement Learning.", parameters_.agent_is_running_ ? "stop" : "start");
        if (button_color) {
            ImGui::PopStyleColor(3);
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset Drones")) {
            onResetDrones();
        }

        RLStatusParser(rl_status);
        auto console = rl_status["console"].cast<std::string>();

        if (ImGui::BeginTabBar("RLStatus")){
            if (ImGui::BeginTabItem("Console")) {
                ImGui::BeginChild("scrolling", ImVec2(0, ImGui::GetTextLineHeightWithSpacing() * 20), true, ImGuiWindowFlags_HorizontalScrollbar);
                ImGui::TextUnformatted(console.c_str());
                ImGui::EndChild();
                ImGui::EndTabItem();
                if (ImGui::Button("Clear Console")){
                    rl_status[py::str("console")] = "";
                    onSetRLStatus(rl_status);
                }
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
            if (ImGui::BeginTabItem("Drone Information")){
                ImVec4 color = ImVec4(0.33f, 0.67f, 0.86f, 1.0f);
                static bool show_exploration_map = false;
                static bool show_input_images = false;
                ImGui::Checkbox("Show View Maps", &show_exploration_map);
                ImGui::Checkbox("Show Drone View", &show_input_images);

                ImGui::BeginChild("scrolling", ImVec2(0, ImGui::GetTextLineHeightWithSpacing() * 40),
                                  true, ImGuiWindowFlags_HorizontalScrollbar);

                static int current_drone_index = 0;

                if (!drones->empty()) {
                    // Create an Array of Drone Names
                    std::vector<std::string> drone_names;
                    for (const auto &drone: *drones) {
                        drone->SetActive(false);
                        drone_names.push_back(drone->GetAgentType() + "_" + std::to_string(drone->GetId()));
                    }

                    // Create a combo box for selecting a drone
                    ImGui::Combo("Select Drone", &current_drone_index, [](void *data, int idx, const char **out_text) {
                        auto &names = *static_cast<std::vector<std::string> *>(data);
                        if (idx < 0 || idx >= names.size()) return false;
                        *out_text = names[idx].c_str();
                        return true;
                    }, &drone_names, static_cast<int>(drone_names.size()));

                    // Get the currently selected drone
                    auto &selected_drone = (*drones)[current_drone_index];
                    selected_drone->SetActive(true);

                    if (show_exploration_map){
                        if (ImGui::Begin("Exploration & Fire Map")) {
                            static bool show_explored_map = true;
                            static bool show_fire_map = false;
                            static bool show_step_explore = false;
                            static bool show_total_drone_view = false;
                            static bool interpolated = true;
                            ImGui::SliderInt("##size_slider", &parameters_.exploration_map_show_size_, 5, 200);
                            ImGui::Checkbox("Interpolated", &interpolated);
                            ImGui::Checkbox("TotalDroneView", &show_total_drone_view);
                            ImGui::Checkbox("ExploredMapTotal", &show_explored_map);
                            ImGui::Checkbox("StepExploreMap", &show_step_explore);
                            ImGui::Checkbox("FireMap", &show_fire_map);
                            ImGui::Spacing();
                            ImGui::Separator();
                            ImGui::Spacing();
                            if (ImGui::BeginTable("ViewTable", 1, ImGuiTableFlags_NoBordersInBody)){
                                if (show_total_drone_view){
                                    ImGui::TableNextRow(ImGuiTableRowFlags_None, 5.0f * static_cast<float>(parameters_.exploration_map_show_size_));
                                    DrawGrid(*gridmap->GetInterpolatedDroneView(selected_drone->GetGridPosition(),
                                                                               selected_drone->GetViewRange(),
                                                                               parameters_.exploration_map_show_size_,
                                                                               interpolated), "total_view");
                                    ImGui::TableNextRow();
                                    ImGui::Spacing();
                                    ImGui::Separator();
                                    ImGui::Spacing();
                                }
                                if (show_explored_map) {
                                    ImGui::TableNextRow(ImGuiTableRowFlags_None, 5.0f * static_cast<float>(parameters_.exploration_map_show_size_));
                                    DrawGrid(*gridmap->GetExploredMap(parameters_.exploration_map_show_size_,
                                                                     interpolated), "exploration_interpolated");
                                    ImGui::TableNextRow();
                                    ImGui::Spacing();
                                    ImGui::Separator();
                                    ImGui::Spacing();
                                }
                                if (show_step_explore) {
                                    ImGui::TableNextRow(ImGuiTableRowFlags_None, 5.0f * static_cast<float>(parameters_.exploration_map_show_size_));
                                    DrawGrid(*gridmap->GetStepExploredMap(parameters_.exploration_map_show_size_,
                                                                      interpolated), "exploration_interpolated_2");
                                    ImGui::TableNextRow();
                                    ImGui::Spacing();
                                    ImGui::Separator();
                                    ImGui::Spacing();
                                }
                                if (show_fire_map) {
                                    ImGui::TableNextRow(ImGuiTableRowFlags_None, 5.0f * static_cast<float>(parameters_.exploration_map_show_size_));
                                    DrawGrid(*gridmap->GetFireMap(parameters_.exploration_map_show_size_, interpolated),
                                             "fire_interpolated");
                                    ImGui::TableNextRow();
                                    ImGui::Spacing();
                                    ImGui::Separator();
                                    ImGui::Spacing();
                                }
                                ImGui::EndTable();
                            }
                            ImGui::End();
                        }
                    }

                    // Display Drone Information
                    ImGui::TextColored(color, "Drone State Information");
                    if (ImGui::BeginTable("DroneInfoTable", 2, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg))
                    {
                        ImGui::TableSetupColumn("Label", ImGuiTableColumnFlags_WidthFixed, 200.0f);
                        ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);
                        ImGui::TableHeadersRow();

                        ImGui::TableNextRow();
                        ImGui::TableNextColumn(); ImGui::Text("Agent Type");
                        ImGui::TableNextColumn(); ImGui::Text("%s", selected_drone->GetAgentType().c_str());

                        ImGui::TableNextRow();
                        ImGui::TableNextColumn(); ImGui::Text("GetOutOfAreaCounter");
                        ImGui::TableNextColumn(); ImGui::Text("%d", selected_drone->GetOutOfAreaCounter());

                        ImGui::TableNextRow();
                        ImGui::TableNextColumn(); ImGui::Text("GetGoalPosition");
                        ImGui::TableNextColumn(); ImGui::Text("(%.6f, %.6f)",
                                                              selected_drone->GetGoalPosition().first,
                                                              selected_drone->GetGoalPosition().second);

                        ImGui::TableNextRow();
                        ImGui::TableNextColumn(); ImGui::Text("GetRealPosition");
                        ImGui::TableNextColumn(); ImGui::Text("(%.6f, %.6f)",
                                                              selected_drone->GetRealPosition().first,
                                                              selected_drone->GetRealPosition().second);

                        ImGui::TableNextRow();
                        ImGui::TableNextColumn(); ImGui::Text("GetGridPosition");
                        ImGui::TableNextColumn(); ImGui::Text("(%d, %d)",
                                                              selected_drone->GetGridPosition().first,
                                                              selected_drone->GetGridPosition().second);

                        ImGui::TableNextRow();
                        ImGui::TableNextColumn(); ImGui::Text("GetGridPositionDouble");
                        ImGui::TableNextColumn(); ImGui::Text("(%.6f, %.6f)",
                                                              selected_drone->GetLastState().GetGridPositionDouble().first,
                                                              selected_drone->GetLastState().GetGridPositionDouble().second);

                        ImGui::TableNextRow();
                        ImGui::TableNextColumn(); ImGui::Text("GetGridPositionDoubleNorm");
                        ImGui::TableNextColumn(); ImGui::Text("(%.6f, %.6f)",
                                                              selected_drone->GetLastState().GetGridPositionDoubleNorm().first,
                                                              selected_drone->GetLastState().GetGridPositionDoubleNorm().second);

                        ImGui::TableNextRow();
                        ImGui::TableNextColumn(); ImGui::Text("GetPositionInExplorationMap");
                        ImGui::TableNextColumn(); ImGui::Text("(%.6f, %.6f)",
                                                              selected_drone->GetLastState().GetPositionInExplorationMap().first,
                                                              selected_drone->GetLastState().GetPositionInExplorationMap().second);

                        ImGui::TableNextRow();
                        ImGui::TableNextColumn(); ImGui::Text("GetPositionNormAroundCenter");
                        ImGui::TableNextColumn(); ImGui::Text("(%.6f, %.6f)",
                                                              selected_drone->GetLastState().GetPositionNormAroundCenter().first,
                                                              selected_drone->GetLastState().GetPositionNormAroundCenter().second);

                        ImGui::TableNextRow();
                        ImGui::TableNextColumn(); ImGui::Text("GetDistanceToNearestBoundaryNorm");
                        ImGui::TableNextColumn(); ImGui::Text("%.6f", selected_drone->GetLastState().GetDistanceToNearestBoundaryNorm());

                        ImGui::TableNextRow();
                        ImGui::TableNextColumn(); ImGui::Text("GetDroneInGrid");
                        ImGui::TableNextColumn(); ImGui::Text("%s", selected_drone->GetDroneInGrid() ? "true" : "false");

                        ImGui::TableNextRow();
                        ImGui::TableNextColumn(); ImGui::Text("GetNewlyExploredCells");
                        ImGui::TableNextColumn(); ImGui::Text("%d", selected_drone->GetNewlyExploredCells());

                        ImGui::TableNextRow();
                        ImGui::TableNextColumn(); ImGui::Text("GetLastTimeStepRevisitedCellsOfAllAgents");
                        ImGui::TableNextColumn(); ImGui::Text("%d", selected_drone->GetRevisitedCells());

                        ImGui::EndTable();
                    }

                    ImGui::Spacing();
                    ImGui::Separator();
                    ImGui::Spacing();
                    ImGui::TextColored(color, "Drone Network Input");
                    if (ImGui::BeginTable("NetworkInputTable", 2, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
                        ImGui::TableSetupColumn("Label", ImGuiTableColumnFlags_WidthFixed, 200.0f);
                        ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);
                        ImGui::TableHeadersRow();

                        ImGui::TableNextRow();
                        ImGui::TableNextColumn(); ImGui::Text("GetDeltaGoal");
                        ImGui::TableNextColumn(); ImGui::Text("(%.6f, %.6f)",
                                                              selected_drone->GetLastState().GetDeltaGoal().first,
                                                              selected_drone->GetLastState().GetDeltaGoal().second);

                        ImGui::TableNextRow();
                        ImGui::TableNextColumn(); ImGui::Text("GetVelocityNorm");
                        ImGui::TableNextColumn(); ImGui::Text("%.6f, %.6f",
                                                              selected_drone->GetLastState().GetVelocityNorm().first,
                                                              selected_drone->GetLastState().GetVelocityNorm().second);

                        ImGui::EndTable();
                    }
                    ImGui::Spacing();
                    ImGui::Separator();
                    ImGui::Spacing();
                    if (show_input_images){
                        if (ImGui::BeginTable("MapsTable", 2, ImGuiTableFlags_Borders))
                        {
                            ImGui::TableNextColumn();
                            ImGui::Text("GetTerrainView");
                            DrawGrid(selected_drone->GetLastState().GetTerrainView(), "terrain");

                            ImGui::TableNextColumn();
                            ImGui::Text("GetFireView");
                            DrawGrid(selected_drone->GetLastState().GetFireView(), "fire");

                            ImGui::EndTable();
                        }
                        auto view_range = selected_drone->GetViewRange();
                        ImGui::Dummy(ImVec2(0.0f, static_cast<float>(view_range) * 5.0f)); // Adds vertical spacing
                        ImGui::Spacing();
                        ImGui::Separator();
                        ImGui::Spacing();
                    }
                    ImGui::Spacing();
                    ImGui::Separator();
                    ImGui::Spacing();
                    ImGui::TextColored(color, "Episodic Rewards");
                    // Plot rewards using ImGui::PlotLines
                    auto rewards = selected_drone->GetEpisodeRewards().getBuffer();
                    auto rewards_pos = static_cast<int>(selected_drone->GetEpisodeRewards().getHead());
                    if (!rewards.empty()) {
                        ImguiHandler::DrawBuffer(rewards, rewards_pos);
                    } else {
                        ImGui::Text("No rewards data available.");
                    }
                    ImGui::Spacing();
                    ImGui::Separator();
                    ImGui::Spacing();
                    ImGui::TextColored(color, "Reward Components");
                    auto reward_components = selected_drone->GetRewardComponents();
                    if (ImGui::BeginTable("RewardComponentTable", 2, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
                        ImGui::TableSetupColumn("Label", ImGuiTableColumnFlags_WidthFixed, 200.0f);
                        ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);
                        ImGui::TableHeadersRow();

                        for (const auto& [key, value] : reward_components) {
                            ImGui::TableNextRow();
                            ImGui::TableNextColumn(); ImGui::Text("%s", key.c_str());
                            ImGui::TableNextColumn(); ImGui::Text("%.6f", value);
                        }

                        ImGui::EndTable();
                    }
                } else {
                    ImGui::Text("No drones available.");
                }
                ImGui::EndChild();
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Env Controls")){
                ImVec4 color = ImVec4(0.33f, 0.67f, 0.86f, 1.0f);
                ImGui::SetWindowFontScale(1.5f);
                ImGui::PushStyleColor(ImGuiCol_Text, color);
                ImGui::TextColored(color, "Fire Controls");
                ImGui::SetWindowFontScale(1.0f);
                ImGui::PopStyleColor();
                if (ImGui::BeginTable("##FireControlsTable", 2, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)){
                    ImGui::TableSetupColumn("Label", ImGuiTableColumnFlags_WidthFixed, 200.0f);
                    ImGui::TableSetupColumn("Control", ImGuiTableColumnFlags_WidthStretch);
                    ImGui::TableHeadersRow();

                    ImGui::TableNextRow();
                    ImGui::TableNextColumn(); ImGui::Text("Fire Percentage");
                    ImGui::TableNextColumn(); ImGui::SetNextItemWidth(-1); ImGui::SliderFloat("##fire_percentage_map", &parameters_.fire_percentage_, 0, 100);

                    ImGui::TableNextRow();
                    ImGui::TableNextColumn(); ImGui::Text("Fire Spread Probability");
                    ImGui::TableNextColumn(); ImGui::SetNextItemWidth(-1); ImGui::SliderFloat("##fire_spread_prob", &parameters_.fire_spread_prob_, 0, 1);

                    ImGui::TableNextRow();
                    ImGui::TableNextColumn(); ImGui::Text("Fire Noise");
                    ImGui::TableNextColumn(); ImGui::SetNextItemWidth(-1); ImGui::SliderFloat("##fire_noise", &parameters_.fire_noise_, -1, 1);

                    ImGui::TableNextRow();
                    ImGui::TableNextColumn(); ImGui::Text("Ignite only Single Cells");
                    ImGui::TableNextColumn(); ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (ImGui::GetContentRegionAvail().x / 2) - 10);
                    ImGui::Checkbox("##SingleCellIgnite", &parameters_.ignite_single_cells_);

                    ImGui::TableNextRow();
                    ImGui::TableNextColumn(); ImGui::Text("Start new Fires");
                    ImGui::TableNextColumn();
                    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (ImGui::GetContentRegionAvail().x / 2) - 50);
                    if (ImGui::Button("##StartFires", ImVec2(100, 17))) { this->startFires(parameters_.fire_percentage_);}
                    if (ImGui::IsItemHovered()){ ImGui::SetTooltip("Click to start some fires");}
                    ImGui::EndTable();
                }
                ImGui::SetWindowFontScale(1.5f);
                ImGui::PushStyleColor(ImGuiCol_Text, color);
                ImGui::TextColored(color, "Start and Goal Controls");
                ImGui::SetWindowFontScale(1.0f);
                ImGui::PopStyleColor();
                if (ImGui::BeginTable("##GoalControl", 2, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
                    ImGui::TableSetupColumn("Label", ImGuiTableColumnFlags_WidthFixed, 200.0f);
                    ImGui::TableSetupColumn("Control", ImGuiTableColumnFlags_WidthStretch);
                    ImGui::TableHeadersRow();

                    ImGui::TableNextRow();
                    ImGui::TableNextColumn(); ImGui::Text("Fire Goal Percentage");
                    ImGui::TableNextColumn(); ImGui::SetNextItemWidth(-1); if(ImGui::SliderFloat("##fire_goal_perc", &parameters_.fire_goal_percentage_, 0, 1))
                    {
                        if(parameters_.fire_goal_percentage_ != 1) {
                            parameters_.groundstation_start_percentage_ = 0;
                        }
                    }
                    if (ImGui::IsItemHovered())
                        ImGui::SetTooltip("The percentage of goals that are set to a fire location, rather than a \n"
                                          "ground station. Set this to 0 to disable fire goals or to 1 to disable all \n"
                                          "groundstation goals.");

                    ImGui::TableNextRow();
                    ImGui::TableNextColumn(); ImGui::Text("Groundstation Start \nPercentage");
                    ImGui::TableNextColumn(); ImGui::SetNextItemWidth(-1); if(ImGui::SliderFloat("##groundstation_start_perc", &parameters_.groundstation_start_percentage_, 0, 1)){
                        if(parameters_.groundstation_start_percentage_ != 0) {
                            parameters_.fire_goal_percentage_ = 1;
                        }
                    }

                    ImGui::TableNextRow();
                    ImGui::TableNextColumn(); ImGui::Text("Non Groundstation Corner \nStart Percentage");
                    ImGui::TableNextColumn(); ImGui::SetNextItemWidth(-1); ImGui::SliderFloat("##corner_start_perc", &parameters_.corner_start_percentage_, 0, 1);

                    ImGui::EndTable();

                }
                ImGui::SetWindowFontScale(1.5f);
                ImGui::PushStyleColor(ImGuiCol_Text, color);
                ImGui::TextColored(color, "fly_agent Behaviour");
                ImGui::SetWindowFontScale(1.0f);
                ImGui::PopStyleColor();
                if (ImGui::BeginTable("##FlyAgentControl", 2, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
                    ImGui::TableSetupColumn("Label", ImGuiTableColumnFlags_WidthFixed, 200.0f);
                    ImGui::TableSetupColumn("Control", ImGuiTableColumnFlags_WidthStretch);
                    ImGui::TableHeadersRow();

                    ImGui::TableNextRow();
                    ImGui::TableNextColumn(); ImGui::Text("Extinguish all Fires");
                    ImGui::TableNextColumn(); ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (ImGui::GetContentRegionAvail().x / 2) - 10);
                    ImGui::Checkbox("##ExtinguishallFires", &parameters_.extinguish_all_fires_);

                    if (ImGui::IsItemHovered())
                        ImGui::SetTooltip("If checked the Agent doesn't stop at the first goal during evaluation \n"
                                          "and only stops if the Map is burned down too much or he extinguished all fires.");

                    ImGui::TableNextRow();
                    ImGui::TableNextColumn(); ImGui::Text("Recharge Time");
                    ImGui::TableNextColumn(); ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (ImGui::GetContentRegionAvail().x / 2) - 10);
                    ImGui::Checkbox("##RechargeTime", &parameters_.recharge_time_active_);

                    if (ImGui::IsItemHovered())
                        ImGui::SetTooltip("If checked the Extinguishing Agent must recharge \n"
                                          "it's fire retardant at the Groundstation.");


                    ImGui::EndTable();

                }
                ImGui::SameLine();
                ImGui::Spacing();
                ImGui::EndTabItem();
            }
        }
        ImGui::EndTabBar();
        ImGui::End();
    }
}

void ImguiHandler::FileHandling(const std::shared_ptr<DatasetHandler>& dataset_handler, std::vector<std::vector<int>> &current_raster_data){

    // Show the file dialog for loading and saving maps
    if (open_file_dialog_) {
        // open Dialog Simple
        IGFD::FileDialogConfig config;
        std::optional<std::string> vFilters;
        std::string vTitle;
        std::string vKey;
        if (model_path_selection_){
            auto root_path = get_project_path("root_path", {});
            config.path = root_path.string();
            if (path_key_ == "model_path"){
                vTitle = "Change Model Path Folder";
                vFilters.reset();
                vKey = "ChooseFolderDlgKey";
            }
            else if (path_key_ == "model_name"){
                vTitle = "Choose Model Name";
                vKey = "ChooseFileDlgKey";
                vFilters = ".pt";
            }
        }
        else if (model_load_selection_) {
            vTitle = "Choose Model to Load";
            auto root_path = get_project_path("root_path", {});
            config.path = root_path.string();
            vFilters = ".pt";
            vKey = "ChooseFileDlgKey";
        }
        else {
            vTitle = "Choose File or Filename";
            auto maps_path = get_project_path("maps_directory", {});
            config.path = maps_path.string();
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
                    if(path_key_ == "model_path"){
                        rl_status[py::str(path_key_)] = py::str(filePath);
                    }
                    else if(path_key_ == "model_name"){
                        rl_status[py::str("model_path")] = py::str(filePath);
                        rl_status[py::str(path_key_)] = py::str(fileName);
                    }
                    onSetRLStatus(rl_status);
                    model_path_selection_ = false;
                }
                else if (model_load_selection_){
                    // Load Model
                    py::dict rl_status = onGetRLStatus();
                    rl_status[py::str("model_path")] = py::str(filePath);
                    rl_status[py::str("model_name")] = py::str(fileName);
                    onSetRLStatus(rl_status);
                    model_load_selection_ = false;
                }
            }
            // close
            ImGuiFileDialog::Instance()->Close();
            open_file_dialog_ = false;
            model_load_selection_ = false;
            model_path_selection_ = false;
            load_map_from_disk_ = false;
            save_map_to_disk_ = false;
        }
    }
}

void ImguiHandler::CheckForModelPathSelection(const std::shared_ptr<FireModelRenderer>& model_renderer) {

    if (!parameters_.check_for_model_folder_empty_) {
        int width, height;
        SDL_GetRendererOutputSize(model_renderer->GetRenderer(), &width, &height);
        // Check if there are files in the model folder from previous runs
        auto rl_status = onGetRLStatus();
        auto model_path = rl_status["model_path"].cast<std::string>();
        auto model_name = rl_status["model_name"].cast<std::string>();
        auto resume = rl_status["resume"].cast<bool>();
        // Find files and folders in model_path
        std::filesystem::path model_dir(model_path);
        if (std::filesystem::exists(model_dir) && std::filesystem::is_directory(model_dir) && !resume) {
            if (!std::filesystem::is_empty(model_dir)) {
                ImGui::Begin("Model Folder Check", nullptr,
                             ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove |
                             ImGuiWindowFlags_AlwaysAutoResize);
                ImVec2 window_size = ImGui::GetWindowSize(); // Actual size after auto-resizing
                ImGui::SetWindowPos(ImVec2((static_cast<float>(width) - window_size.x) * 0.5f,
                                           (static_cast<float>(height) - window_size.y) * 0.5f));
                ImGui::Spacing();
                ImGui::Text("Model Folder not empty.");
                ImGui::Text("Delete all files in the folder:");
                ImGui::Text("%s", model_path.c_str());
                if (ImGui::Button("Delete Files and Continue",ImVec2(-1, 0))) {
                    parameters_.check_for_model_folder_empty_ = true;
                }
                if (ImGui::Button("Close Program", ImVec2(-1, 0))) {
                    // Close the dialog and close the program correctly
                    parameters_.check_for_model_folder_empty_ = true;
                    parameters_.initial_mode_selection_done_ = true;
                    parameters_.exit_carefully_ = true;
                }
                ImGui::End();
            } else {
                parameters_.check_for_model_folder_empty_ = true;
            }
        } else {
            parameters_.check_for_model_folder_empty_ = true;
        }
    }
}


bool ImguiHandler::ImGuiOnStartup(const std::shared_ptr<FireModelRenderer>& model_renderer, std::vector<std::vector<int>> &current_raster_data) {

    if (parameters_.skip_gui_init_) {
        this->CheckForModelPathSelection(model_renderer);
    }

    if (!model_startup_ && !parameters_.skip_gui_init_) {
        int width, height;
        static bool model_path_selection = false;
        SDL_GetRendererOutputSize(model_renderer->GetRenderer(), &width, &height);
        if (!model_mode_selection_ && mode_ == Mode::GUI_RL) {
            ImGui::Begin("Select Mode", nullptr,
                         ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove |
                         ImGuiWindowFlags_AlwaysAutoResize);
            ImVec2 window_size = ImGui::GetWindowSize(); // Actual size after auto-resizing
            ImGui::SetWindowPos(ImVec2((static_cast<float>(width) - window_size.x) * 0.5f,
                                       (static_cast<float>(height) - window_size.y) * 0.5f));
            ImGui::Spacing();
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.45f, 0.6f, 0.85f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.45f, 0.7f, 0.95f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.45f, 0.5f, 0.75f, 1.0f));
            auto rl_status = onGetRLStatus();

            if (ImGui::Button("Train Model", ImVec2(-1, 0))) {
//                py::dict rl_status = onGetRLStatus();
                auto console = rl_status["console"].cast<std::string>();
                console += "Initialized ROSHAN in Train mode.\n";
                rl_status[py::str("rl_mode")] = py::str("train");
                train_mode_selected_ = true;
                model_mode_selection_ = true;
                model_path_selection = true;
            }
            if (ImGui::Button("Load Model", ImVec2(-1, 0))) {
//                py::dict rl_status = onGetRLStatus();
                auto console = rl_status["console"].cast<std::string>();
                console += "Initialized ROSHAN in Eval mode.\n";
                rl_status[py::str("console")] = console;
                rl_status[py::str("rl_mode")] = py::str("eval");
                model_mode_selection_ = true;
                model_load_selection_ = true;
                open_file_dialog_ = true;
                parameters_.check_for_model_folder_empty_ = true;
            }
            ImGui::Spacing();
            auto resume = rl_status["resume"].cast<bool>();
            ImGui::Checkbox("Resume Training from Checkpoint?", &resume);
            ImGui::Spacing();
            rl_status[py::str("resume")] = resume;
            onSetRLStatus(rl_status);
            ImGui::PopStyleColor(3);
            ImGui::End();
            return true;
        }
        else if (model_path_selection) {
            this->CheckForModelPathSelection(model_renderer);
            if (parameters_.check_for_model_folder_empty_) {
                model_path_selection = false;
            }
            return true;
        }
        else if (train_mode_selected_) {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.45f, 0.6f, 0.85f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.45f, 0.7f, 0.95f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.45f, 0.5f, 0.75f, 1.0f));
            ImGui::Begin("Training Setup", nullptr,
                         ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove |
                         ImGuiWindowFlags_AlwaysAutoResize);
            ImVec2 window_size = ImGui::GetWindowSize(); // Actual size after auto-resizing
            ImGui::SetWindowPos(ImVec2((static_cast<float>(width) - window_size.x) * 0.5f,
                                       (static_cast<float>(height) - window_size.y) * 0.5f));
            ImGui::Spacing();
            ImGui::Text("Algorithm specific parameters must be changed in the Config files.");
            ImGui::Spacing();
            // Combo Box for selecting the Agent Types
            auto rl_status = onGetRLStatus();
            auto hierarchy_type = rl_status["hierarchy_type"].cast<std::string>();
            const char *hierarchy_types[] = {"fly_agent", "explore_agent", "planner_agent"};
            static int current_hierarchy_type =
                    hierarchy_type == "fly_agent" ? 0 : hierarchy_type == "explore_agent" ? 1 : 2;
            ImGui::Text("Select Agent Type");
            if (ImGui::BeginCombo("##Hierarchy Type", hierarchy_types[current_hierarchy_type])) {
                for (int n = 0; n < IM_ARRAYSIZE(hierarchy_types); n++) {
                    const bool is_selected = (current_hierarchy_type == n);
                    if (ImGui::Selectable(hierarchy_types[n], is_selected))
                        current_hierarchy_type = n;
                    if (is_selected)
                        ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }
            ImGui::Spacing();
            auto num_agents = rl_status["num_agents"].cast<int>();
            ImGui::InputInt("Number of Agents", &num_agents, 1, 10, ImGuiInputTextFlags_CharsDecimal);
            if (ImGui::IsItemHovered()) {
                // FlyAgent
                if (current_hierarchy_type == 0) {
                    ImGui::SetTooltip(
                            "The Number of Agents is the number of drones present in the environment. Each drone collects observations and shares the same network.");
                } // explore_agent
                else if (current_hierarchy_type == 1) {
                    ImGui::SetTooltip(
                            "The Number of Agents is the number of Explorers the explore_agents deploys. Each Explorer collects observations and shares the same network.");
                }
            }
            if (current_hierarchy_type == 0) {
                parameters_.SetNumberOfDrones(num_agents);
            } else if (current_hierarchy_type == 1) {
                parameters_.SetNumberOfExplorers(num_agents);
            } else {
                parameters_.SetNumberOfExtinguishers(num_agents);
            }
            rl_status[py::str("num_agents")] = num_agents;
            ImGui::Spacing();

            rl_status[py::str("hierarchy_type")] = hierarchy_types[current_hierarchy_type];


            // Combo Box for selecting the RL Algorithm
            auto rl_algorithm = rl_status["rl_algorithm"].cast<std::string>();
            const char *rl_algorithms[] = {"PPO", "TD3", "IQL"};
            static int current_rl_algorithm =
                    rl_algorithm == "PPO" ? 0 : rl_algorithm == "TD3" ? 1 : rl_algorithm == "IQL" ? 2 : 3;
            ImGui::Text("Select RL Algorithm");
            if (ImGui::BeginCombo("##RL Algorithm", rl_algorithms[current_rl_algorithm])) {
                for (int n = 0; n < IM_ARRAYSIZE(rl_algorithms); n++) {
                    const bool is_selected = (current_rl_algorithm == n);
                    if (ImGui::Selectable(rl_algorithms[n], is_selected))
                        current_rl_algorithm = n;
                    if (is_selected)
                        ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }
            rl_status[py::str("rl_algorithm")] = rl_algorithms[current_rl_algorithm];
            ImGui::Spacing();
            if (ImGui::Button("Proceed to Map Selection", ImVec2(-1, 0))) {
                train_mode_selected_ = false;
            }
            ImGui::PopStyleColor(3);
            ImGui::End();
            return true;
        }
        else {
            ImGui::Begin("Map Selection", nullptr,
                         ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove |
                         ImGuiWindowFlags_AlwaysAutoResize);
            ImGui::Spacing();
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.45f, 0.6f, 0.85f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.45f, 0.7f, 0.95f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.45f, 0.5f, 0.75f, 1.0f));
            ImVec2 window_size = ImGui::GetWindowSize(); // Actual size after auto-resizing
            ImGui::SetWindowPos(ImVec2((static_cast<float>(width) - window_size.x) * 0.5f,
                                       (static_cast<float>(height) - window_size.y) * 0.5f));
            bool still_no_init = true;
            if (ImGui::Button("Uniform Vegetation")) {
                onSetUniformRasterData();
                // TODO I don't really know why this works, but it does. If I don't call this function, everything works fine
                //  until I reset the GridMap again, which works fine as well. But if I then try to zoom in it crashes.
                onResetGridMap(&current_raster_data);
                still_no_init = false;
                model_startup_ = true;
                show_controls_ = true;
                show_model_parameter_config_ = false;
                parameters_.initial_mode_selection_done_ = true;
            }
            ImGui::Separator();
            if (ImGui::Button("Load from File", ImVec2(-1, 0))) {
                show_model_parameter_config_ = false;
                open_file_dialog_ = true;
                load_map_from_disk_ = true;
                parameters_.initial_mode_selection_done_ = true;
            }
            ImGui::PopStyleColor(3);
            ImGui::End();
            return still_no_init;
        }
    }
    return false;
}

void ImguiHandler::ShowParameterConfig(const std::shared_ptr<Wind>& wind) {
    ImGuiWindowFlags window_flags =
            ImGuiWindowFlags_HorizontalScrollbar | ImGuiWindowFlags_AlwaysVerticalScrollbar;
    ImGui::Begin("Simulation Parameters", &show_model_parameter_config_, window_flags);
    static double tau_min = 0.01;
    static double tau_max = 100.0;
    static double min_Y_st = 0.0;
    static double max_Y_st = 1.0;
    static double min_Y_lim = 0.1;
    static double max_Y_lim = 0.3;
    static double min_Fl = 0.0;
    static double max_Fl = 10.0;
    static double min_C0 = 1.5;
    static double max_C0 = 2.0;
    static double min_Lt = 10.0;
    static double max_Lt = 100.0;
    if (parameters_.emit_convective_) {
        ImGui::SeparatorText("Virtual Particles");
        if (ImGui::TreeNodeEx("##Virtual Particles",
                              ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_SpanAvailWidth)) {
            ImGui::Text("Particle Lifetime");
            ImGui::SliderScalar("tau_mem", ImGuiDataType_Double, &parameters_.virtualparticle_tau_mem_,
                                &tau_min, &tau_max, "%.3f", 1.0f);
            ImGui::Text("Hotness");
            ImGui::SliderScalar("Y_st", ImGuiDataType_Double, &parameters_.virtualparticle_y_st_,
                                &min_Y_st, &max_Y_st, "%.3f", 1.0f);
            ImGui::Text("Ignition Threshold");
            ImGui::SliderScalar("Y_lim", ImGuiDataType_Double, &parameters_.virtualparticle_y_lim_,
                                &min_Y_lim, &max_Y_lim, "%.3f", 1.0f);
            ImGui::Text("Height of Emission");
            ImGui::SliderScalar("Lt", ImGuiDataType_Double, &parameters_.Lt_, &min_Lt,
                                &max_Lt, "%.3f", 1.0f);
            ImGui::Text("Scaling Factor");
            ImGui::SliderScalar("Fl", ImGuiDataType_Double, &parameters_.virtualparticle_fl_, &min_Fl,
                                &max_Fl, "%.3f", 1.0f);
            ImGui::Text("Constant");
            ImGui::SliderScalar("C0", ImGuiDataType_Double, &parameters_.virtualparticle_c0_, &min_C0,
                                &max_C0, "%.3f", 1.0f);
            ImGui::TreePop();
            ImGui::Spacing();
        }
    }

    if (parameters_.emit_radiation_) {
        ImGui::SeparatorText("Radiation Particles");
        if(ImGui::TreeNodeEx("##Radiation Particles", ImGuiTreeNodeFlags_SpanAvailWidth)){
            ImGui::Text("Radiation Hotness");
            ImGui::SliderScalar("Y_st_r", ImGuiDataType_Double, &parameters_.radiationparticle_y_st_, &min_Y_st, &max_Y_st, "%.3f", 1.0f);
            ImGui::Text("Radiation Ignition Threshold");
            ImGui::SliderScalar("Y_lim_r", ImGuiDataType_Double, &parameters_.radiationparticle_y_lim_, &min_Y_lim, &max_Y_lim, "%.3f", 1.0f);
//            ImGui::Text("Radiation Length");
//            ImGui::SliderScalar("Lr", ImGuiDataType_Double, &parameters_.radiationparticle_Lr_, &parameters_.min_Lr_, &parameters_.max_Lr_, "%.3f", 1.0f);
            ImGui::TreePop();
            ImGui::Spacing();
        }
    }

    if (parameters_.map_is_uniform_) {
        ImGui::SeparatorText("Cell (Terrain)");
        static double min_burning_duration = 1.0;
        static double max_burning_duration = 200.0;
        static double min_ignition_threshold = 1.0;
        static double max_ignition_threshold = 500.0;
        static double min_cell_size = 1.0;
        static double max_cell_size = 100.0;
        if(ImGui::TreeNodeEx("##CellTerrain", ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_SpanAvailWidth)) {
            ImGui::Spacing();
            ImGui::Text("Cell Size");
            //Colored Text in grey on the same line
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "(?)");
            if(ImGui::IsItemHovered())
                ImGui::SetTooltip("always press [Reset GridMap] manually after changing these values");
            ImGui::SliderScalar("##Cell Size", ImGuiDataType_Double, &parameters_.cell_size_, &min_cell_size, &max_cell_size, "%.3f", 1.0f);
            ImGui::Text("Cell Ignition Threshold");
            ImGui::SliderScalar("##Cell Ignition Threshold", ImGuiDataType_Double, &parameters_.cell_ignition_threshold_, &min_ignition_threshold, &max_ignition_threshold, "%.3f", 1.0f);
            ImGui::Text("Cell Burning Duration");
            ImGui::SliderScalar("##Cell Burning Duration", ImGuiDataType_Double, &parameters_.cell_burning_duration_, &min_burning_duration, &max_burning_duration, "%.3f", 1.0f);
            ImGui::TreePop();
            ImGui::Spacing();
        }
    }

    ImGui::SeparatorText("Wind");
    static double min_Uw = 0.0;
    static double max_Uw = 35.0;
    static double min_A = 0.2;
    static double max_A = 0.5;
    if(ImGui::TreeNodeEx("##Wind", ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_SpanAvailWidth)) {
        double min_angle_degree = 0;
        double max_angle_degree = (2* M_PI);
        bool update_wind = false;
        ImGui::Text("Wind Speed");
        if(ImGui::SliderScalar("##Wind Speed", ImGuiDataType_Double, &parameters_.wind_uw_, &min_Uw, &max_Uw, "%.3f", 1.0f)) update_wind = true;
        ImGui::Text("A");
        if(ImGui::SliderScalar("##A", ImGuiDataType_Double, &parameters_.wind_a_, &min_A, &max_A, "%.3f", 1.0f)) update_wind = true;
        ImGui::Text("Wind Angle");
        if(ImGui::SliderScalar("##Wind Angle", ImGuiDataType_Double, &parameters_.wind_angle_, &min_angle_degree, &max_angle_degree, "%.1f", 1.0f)) update_wind = true;
        if (update_wind)
            wind->UpdateWind();
        ImGui::TreePop();
        ImGui::Spacing();
    }
    ImGui::End();
}

void ImguiHandler::ShowPopups(const std::shared_ptr<GridMap>& gridmap, std::vector<std::vector<int>> &current_raster_data) {
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

void ImguiHandler::HandleEvents(SDL_Event event, ImGuiIO *io, const std::shared_ptr<GridMap>& gridmap, const std::shared_ptr<FireModelRenderer>& model_renderer,
                                const std::shared_ptr<DatasetHandler>& dataset_handler, std::vector<std::vector<int>> &current_raster_data, bool agent_is_running) {
    // SDL Events
    if (event.type == SDL_MOUSEBUTTONDOWN && !io->WantCaptureMouse && event.button.button == SDL_BUTTON_LEFT) {
        int x, y;
        SDL_GetMouseState(&x, &y);
        std::pair<int, int> gridPos = model_renderer->ScreenToGridPosition(x, y);
        x = gridPos.first;
        y = gridPos.second;
        if (x >= 0 && x < gridmap->GetRows() && y >= 0 && y < gridmap->GetCols()) {
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
    }
    else if (event.type == SDL_MOUSEMOTION && !io->WantCaptureMouse) {
        int x, y;
        Uint32 mouseState = SDL_GetMouseState(&x, &y);
        if (mouseState & SDL_BUTTON(SDL_BUTTON_RIGHT)) // Middle mouse button is pressed
        {
            model_renderer->ChangeCameraPosition(-event.motion.xrel, -event.motion.yrel);
        }
    }
    else if (event.type == SDL_MOUSEBUTTONDOWN && event.button.button == SDL_BUTTON_MIDDLE && !io->WantCaptureMouse) {
        int x, y;
        SDL_GetMouseState(&x, &y);
        std::pair<int, int> cell_pos = model_renderer->ScreenToGridPosition(x, y);
        if (cell_pos.first >= 0 && cell_pos.first < gridmap->GetRows() && cell_pos.second >= 0 && cell_pos.second < gridmap->GetCols()) {
            popups_.insert(cell_pos);
            popup_has_been_opened_.insert({cell_pos, false});
        }
    }
    else if (event.type == SDL_WINDOWEVENT) {
        if (event.window.event == SDL_WINDOWEVENT_SIZE_CHANGED) {
            model_renderer->ResizeEvent();
        }
    }
    // TODO Manual Drone Control for exploration flyers
    else if (event.type == SDL_KEYDOWN && parameters_.GetNumberOfFlyAgents() == 1 && !agent_is_running && !io->WantTextInput) {
        if (event.key.keysym.sym == SDLK_w)
            onMoveDrone(0, -1, 0, 0);
        // MoveDroneByAngle(0, 0.25, 0, 0);
        if (event.key.keysym.sym == SDLK_s)
            onMoveDrone(0, 1, 0, 0);
        // MoveDroneByAngle(0, -0.25, 0, 0);
        if (event.key.keysym.sym == SDLK_a)
            onMoveDrone(0, 0, -1, 0);
        // MoveDroneByAngle(0, 0, -0.25, 0);
        if (event.key.keysym.sym == SDLK_d)
            onMoveDrone(0, 0, 1, 0);
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
    float min_value = *std::min_element(buffer.begin(), buffer.end());
    float max_value = *std::max_element(buffer.begin(), buffer.end());

    // Display a legend
    ImGui::Text("Min: %.2f | Max: %.2f | Avg: %.2f", min_value, max_value,
                std::accumulate(buffer.begin(), buffer.end(), 0.0f) / static_cast<float>(buffer.size()));

    // Plot the graph
    ImGui::PlotLines("", buffer.data(), static_cast<int>(buffer.size()), 0, nullptr, min_value, max_value, ImVec2(0, 150));

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    ImVec2 graph_pos = ImGui::GetItemRectMin();
    ImVec2 graph_size = ImGui::GetItemRectSize();

    // Highlight selected data point
    if (buffer_pos >= 0 && buffer_pos < buffer.size()) {
        float x = graph_pos.x + ((float)buffer_pos / (static_cast<float>(buffer.size()) - 1)) * graph_size.x;
        float y = graph_pos.y + graph_size.y - ((buffer[buffer_pos] - min_value) / (max_value - min_value) * graph_size.y);
        draw_list->AddLine(ImVec2(x, graph_pos.y), ImVec2(x, graph_pos.y + graph_size.y), IM_COL32(255, 0, 0, 255), 2.0f);
        draw_list->AddCircleFilled(ImVec2(x, y), 3.0f, IM_COL32(0, 255, 125, 255));

        // Dynamic tooltip
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Value: %.2f\nIndex: %d", buffer[buffer_pos], buffer_pos);
        }
    }
}

template<typename T>
void ImguiHandler::DrawGrid(const std::vector<std::vector<T>> &grid, const std::string color_status) {
    ImVec2 cursor_pos = ImGui::GetCursorScreenPos();

    // Function to map a value to a color
    auto max_exploration_time = static_cast<float>(parameters_.GetExplorationTime());
    std::function<ImVec4(double)> value_to_color;
    if (color_status == "exploration_interpolated") {
        value_to_color = [&max_exploration_time](double value) -> ImVec4 {
            float normalized_value = std::clamp(static_cast<float>(value) / max_exploration_time, 0.0f, 0.6f);
            return {0.6f - normalized_value, 0.6f - normalized_value, 0.3f, 1.0f};
        };
    } else if (color_status == "fire_interpolated") {
        value_to_color = [](double value) -> ImVec4 {
            return {0.0f + static_cast<float>(value), 1.0f - static_cast<float>(value), 0.0f, 1.0f};
        };
    } else if (color_status == "exploration_interpolated_2") {
        value_to_color = [](double value) -> ImVec4 {
            return {0.2f, 0.0f + static_cast<float>(value), 0.2f, 1.0f};
        };
    } else if (color_status == "total_view") {
        value_to_color = [](double value) -> ImVec4 {
            float normalized_value = std::clamp(static_cast<float>(value), 0.0f, 1.0f);
            return {0.6f - normalized_value, 0.6f - normalized_value, 0.3f, 1.0f};
        };
    }

    for (int y = 0; y < grid.size(); ++y) {
        for (int x = 0; x < grid[y].size(); ++x) {
            ImVec4 color;
            ImVec2 p_min;
            ImVec2 p_max;
            if (color_status == "exploration_interpolated"
                || color_status == "fire_interpolated"
                || color_status == "total_view"
                || color_status == "exploration_interpolated_2") {
                color = value_to_color(grid[y][x]);
            } else {
                if (color_status == "fire") {
                    color = grid[y][x] > 0 ? ImVec4(1.0f, 0.0f, 0.0f, 1.0f) : ImVec4(0.0f, 1.0f, 0.0f, 1.0f);
                } else if (color_status == "terrain") {
                    color = FireModelRenderer::GetMappedColor(grid[y][x]);
                } else {
                    color = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
                }
            }
            p_min = ImVec2(cursor_pos.x + x * 5.0, cursor_pos.y + y * 5.0);
            p_max = ImVec2(cursor_pos.x + (x + 1) * 5.0, cursor_pos.y + (y + 1) * 5.0);

            ImGui::GetWindowDrawList()->AddRectFilled(p_min, p_max, IM_COL32(color.x * 255, color.y * 255, color.z * 255, color.w * 255));
        }
    }
}

void ImguiHandler::OpenBrowser(const std::string& url) {
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
