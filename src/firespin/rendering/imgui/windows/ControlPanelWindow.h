//
// ControlPanelWindow.h - Merged Control Panel for GUI_RL mode
//
// Combines SimulationControlsWindow and RLStatusWindow into a single,
// unified window with compact header, progress bars, and 5 tabs:
//   Agents | Environment | Simulation | Log | Settings
//

#ifndef ROSHAN_CONTROLPANELWINDOW_H
#define ROSHAN_CONTROLPANELWINDOW_H

#include "IWindow.h"
#include "../UITypes.h"
#include "../UIState.h"
#include "../components/LogConsoleWidget.h"
#include "../components/DroneInfoWidget.h"
#include "../components/RewardPlot.h"
#include "../components/GridVisualizer.h"
#include "../components/ConfigTable.h"
#include "../components/ProgressBar.h"
#include "../components/StatusIndicator.h"
#include "firespin/model_parameters.h"
#include "firespin/firemodel_gridmap.h"
#include "firespin/rendering/firemodel_renderer.h"
#include "reinforcementlearning/agents/fly_agent.h"
#include "src/utils.h"
#include "externals/pybind11/include/pybind11/pybind11.h"
#include <memory>
#include <functional>
#include <vector>
#include <string>
#include <cstring>
#include <cmath>

namespace py = pybind11;

namespace ui {

class __attribute__((visibility("hidden"))) ControlPanelWindow : public IWindow {
public:
    ControlPanelWindow(FireModelParameters& parameters,
                       std::shared_ptr<GridMap> gridmap,
                       std::shared_ptr<FireModelRenderer> renderer,
                       Mode mode)
        : parameters_(parameters)
        , gridmap_(std::move(gridmap))
        , renderer_(std::move(renderer))
        , mode_(mode) {}

    void Render() override {
        if (!visible_ || mode_ != Mode::GUI_RL || !startupComplete_) return;

        py::dict rlStatus = getRLStatus_();

        ImGui::SetNextWindowSize(ImVec2(540, 680), ImGuiCond_FirstUseEver);
        ImGui::Begin("Control Panel", &visible_);

        RenderHeader(rlStatus);
        ImGui::Spacing();
        RenderProgressSection(rlStatus);
        ImGui::Spacing();
        RenderMetricsOverview(rlStatus);
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        RenderTabs(rlStatus);

        ImGui::End();
    }

    bool IsVisible() const override { return visible_; }
    void SetVisible(bool visible) override { visible_ = visible; }
    const char* GetName() const override { return "Control Panel"; }

    void SetStartupComplete(bool complete) { startupComplete_ = complete; }

    // RL callbacks
    void SetRLStatusCallbacks(std::function<py::dict()> get, std::function<void(py::dict)> set) {
        getRLStatus_ = std::move(get);
        setRLStatus_ = std::move(set);
    }

    void SetResetDronesCallback(std::function<void()> cb) { onResetDrones_ = std::move(cb); }
    void SetStartFiresCallback(std::function<void()> cb) { onStartFires_ = std::move(cb); }
    void SetFileDialogCallback(std::function<void(const std::string&)> cb) { onOpenFileDialog_ = std::move(cb); }
    void SetResetGridMapCallback(std::function<void(std::vector<std::vector<int>>*, bool)> cb) { onResetGridMap_ = std::move(cb); }

    // Update references
    void SetGridMap(std::shared_ptr<GridMap> gridmap) { gridmap_ = std::move(gridmap); }
    void SetRenderer(std::shared_ptr<FireModelRenderer> renderer) { renderer_ = std::move(renderer); }
    void SetDrones(std::shared_ptr<std::vector<std::shared_ptr<FlyAgent>>> drones) { drones_ = std::move(drones); }

    // Simulation control pointers
    void SetUpdateSimulation(bool* update) { updateSimulation_ = update; }
    void SetRenderSimulation(bool* render) { renderSimulation_ = render; }
    void SetDelay(int* delay) { delay_ = delay; }
    void SetFramerate(float framerate) { framerate_ = framerate; }
    void SetRunningTime(double time) { runningTime_ = time; }
    void SetRasterData(std::vector<std::vector<int>>* data) { rasterData_ = data; }

    // User input/output for ROSHAN-AI
    void SetUserInput(std::string* input) { userInput_ = input; }
    void SetModelOutput(std::string* output) { modelOutput_ = output; }

    // Log reader
    void SetLogReader(LogReader* reader) { logReader_ = reader; }
    void SetConsoleResetFlag(bool* flag) { resetConsole_ = flag; }

    LogConsoleWidget& GetLogConsole() { return logConsole_; }

private:
    // ===== Compact Header: [TRAIN][EVAL] Status [Start] [Sim][Render][Reset] =====
    void RenderHeader(py::dict& rlStatus) {
        auto rlMode = rlStatus["rl_mode"].cast<std::string>();
        auto agentIsRunning = rlStatus["agent_is_running"].cast<bool>();
        bool isTrain = (rlMode == "train");

        float buttonHeight = 30.0f;

        // TRAIN button
        if (isTrain) {
            ImGui::PushStyleColor(ImGuiCol_Button, colors::kButtonActive);
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, colors::kButtonActiveHovered);
        } else {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.2f, 0.22f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.28f, 0.28f, 0.32f, 1.0f));
        }
        if (ImGui::Button("TRAIN", ImVec2(72, buttonHeight)) && !isTrain) {
            if (!parameters_.eval_fly_policy_) {
                ImGui::OpenPopup("Switch Mode");
            }
        }
        ImGui::PopStyleColor(2);

        ImGui::SameLine(0, 4.0f);

        // EVAL button
        if (!isTrain) {
            ImGui::PushStyleColor(ImGuiCol_Button, colors::kButtonActive);
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, colors::kButtonActiveHovered);
        } else {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.2f, 0.22f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.28f, 0.28f, 0.32f, 1.0f));
        }
        if (ImGui::Button("EVAL", ImVec2(72, buttonHeight)) && isTrain) {
            if (!parameters_.eval_fly_policy_) {
                ImGui::OpenPopup("Switch Mode");
            }
        }
        ImGui::PopStyleColor(2);

        // Mode switch confirmation popup
        if (ImGui::BeginPopupModal("Switch Mode", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::Text("Switch to %s mode?", isTrain ? "eval" : "train");
            ImGui::Separator();

            if (ImGui::Button("OK", ImVec2(120, 0))) {
                py::dict status = getRLStatus_();
                std::string newMode = isTrain ? "eval" : "train";
                status["rl_mode"] = newMode;
                setRLStatus_(status);
                // Reset training timer on mode switch
                trainingStartTime_ = ImGui::GetTime();
                ImGui::CloseCurrentPopup();
            }
            ImGui::SameLine();
            if (ImGui::Button("Cancel", ImVec2(120, 0))) {
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }

        ImGui::SameLine(0, 14.0f);

        // Status badge
        StatusIndicator::DrawBadge(agentIsRunning ? "Running" : "Stopped",
                                   agentIsRunning ? StatusState::Active : StatusState::Idle);

        ImGui::SameLine(0, 6.0f);

        // Start/Stop RL button
        {
            bool colored = false;
            if (agentIsRunning) {
                ImGui::PushStyleColor(ImGuiCol_Button, colors::kButtonActive);
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, colors::kButtonActiveHovered);
                ImGui::PushStyleColor(ImGuiCol_ButtonActive, colors::kButtonActivePressed);
                colored = true;
            }
            if (ImGui::Button(agentIsRunning ? "Stop" : "Start", ImVec2(50, buttonHeight))) {
                rlStatus[py::str("agent_is_running")] = py::bool_(!agentIsRunning);
                setRLStatus_(rlStatus);
            }
            if (colored) ImGui::PopStyleColor(3);
        }

        ImGui::SameLine(0, 16.0f);

        // Simulation control buttons (compact)
        {
            bool simActive = updateSimulation_ && *updateSimulation_;
            bool colored = false;
            if (simActive) {
                ImGui::PushStyleColor(ImGuiCol_Button, colors::kButtonActive);
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, colors::kButtonActiveHovered);
                ImGui::PushStyleColor(ImGuiCol_ButtonActive, colors::kButtonActivePressed);
                colored = true;
            }
            if (ImGui::Button("Sim", ImVec2(38, buttonHeight)) && updateSimulation_) {
                *updateSimulation_ = !*updateSimulation_;
            }
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("%s simulation", simActive ? "Stop" : "Start");
            if (colored) ImGui::PopStyleColor(3);
        }

        ImGui::SameLine(0, 4.0f);

        {
            bool rendActive = renderSimulation_ && *renderSimulation_;
            bool colored = false;
            if (rendActive) {
                ImGui::PushStyleColor(ImGuiCol_Button, colors::kButtonActive);
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, colors::kButtonActiveHovered);
                ImGui::PushStyleColor(ImGuiCol_ButtonActive, colors::kButtonActivePressed);
                colored = true;
            }
            if (ImGui::Button("Rend", ImVec2(44, buttonHeight)) && renderSimulation_) {
                *renderSimulation_ = !*renderSimulation_;
            }
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("%s rendering", rendActive ? "Stop" : "Start");
            if (colored) ImGui::PopStyleColor(3);
        }

        ImGui::SameLine(0, 4.0f);

        if (ImGui::Button("Reset", ImVec2(48, buttonHeight))) {
            if (onResetGridMap_ && rasterData_) {
                onResetGridMap_(rasterData_, false);
            }
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Reset GridMap to initial state");
    }

    // ===== Progress Section with Speed Metrics =====
    void RenderProgressSection(py::dict& rlStatus) {
        auto obsCollected = rlStatus["obs_collected"].cast<int>();
        auto minUpdate = rlStatus["min_update"].cast<int>();
        auto autoTrain = rlStatus["auto_train"].cast<bool>();

        // Calculate training speed
        double now = ImGui::GetTime();
        if (now - lastSpeedUpdate_ > 0.5) {
            int currentObs = obsCollected;
            if (lastSpeedUpdate_ > 0 && now > lastSpeedUpdate_) {
                stepsPerSec_ = static_cast<float>(currentObs - lastObsCount_) / static_cast<float>(now - lastSpeedUpdate_);
                if (stepsPerSec_ < 0) stepsPerSec_ = 0;
            }
            lastSpeedUpdate_ = now;
            lastObsCount_ = currentObs;
        }

        // Episode progress (if auto-train is enabled)
        if (autoTrain) {
            auto trainEpisode = rlStatus["train_episode"].cast<int>();
            auto trainEpisodes = rlStatus["train_episodes"].cast<int>();
            ProgressBar::DrawWithValues("Episode", trainEpisode + 1, trainEpisodes, ImVec2(-1, 18));
        }

        // Horizon progress
        ProgressBar::DrawWithValues("Horizon", obsCollected, minUpdate, ImVec2(-1, 18));

        // Environment steps
        int currentSteps = parameters_.GetCurrentEnvSteps();
        int totalSteps = parameters_.total_env_steps_;
        ProgressBar::DrawWithValues("Env Steps", currentSteps, totalSteps, ImVec2(-1, 18));

        // Speed and ETA
        ImGui::Text("Speed: %.0f steps/sec", stepsPerSec_);
        if (stepsPerSec_ > 1.0f) {
            int remaining = minUpdate - obsCollected;
            if (remaining > 0) {
                int etaSeconds = static_cast<int>(static_cast<float>(remaining) / stepsPerSec_);
                ImGui::SameLine();
                ImGui::TextColored(colors::kHint, "| ETA: %s", FormatTime(etaSeconds).c_str());
            }
        }
    }

    // ===== Metrics Overview =====
    void RenderMetricsOverview(const py::dict& rlStatus) {
        auto trainStep = rlStatus["train_step"].cast<int>();
        auto policyUpdates = rlStatus["policy_updates"].cast<int>();
        auto currentEpisode = rlStatus["current_episode"].cast<int>();

        ImGui::Text("Steps: %d | Updates: %d | Episode: %d", trainStep, policyUpdates, currentEpisode);

        if (currentEpisode > 100) {
            auto objective = rlStatus["objective"].cast<double>();
            auto bestObjective = rlStatus["best_objective"].cast<double>();

            ImGui::Text("Objective: ");
            ImGui::SameLine(0, 0);
            ImVec4 objColor = objective >= 0 ? colors::kStatusActive : colors::kStatusError;
            ImGui::TextColored(objColor, "%.2f", objective);
            ImGui::SameLine();
            ImGui::Text("| Best: ");
            ImGui::SameLine(0, 0);
            ImGui::TextColored(colors::kHighlight, "%.2f", bestObjective);
        }

        // Training elapsed timer
        double elapsed = ImGui::GetTime() - trainingStartTime_;
        ImGui::Text("Training Time: %s", FormatTime(static_cast<int>(elapsed)).c_str());
    }

    // ===== Tabs =====
    void RenderTabs(py::dict& rlStatus) {
        if (!ImGui::BeginTabBar("ControlPanelTabs")) return;

        RenderAgentsTab(rlStatus);
        RenderEnvironmentTab();
        RenderSimulationTab();
        RenderLogTab();
        RenderSettingsTab(rlStatus);

        ImGui::EndTabBar();
    }

    // Tab 1: Agents
    void RenderAgentsTab(py::dict& rlStatus) {
        if (!ImGui::BeginTabItem("Agents")) {
            if (drones_ && !drones_->empty()) {
                for (const auto& drone : *drones_) {
                    drone->SetActive(false);
                }
            }
            return;
        }

        if (!drones_ || drones_->empty()) {
            ImGui::TextDisabled("No agents available");
            ImGui::EndTabItem();
            return;
        }

        RenderDroneSelector();
        auto& selectedDrone = (*drones_)[droneUI_.selectedDroneIndex];
        selectedDrone->SetActive(true);
        parameters_.active_drone_ = droneUI_.selectedDroneIndex;

        ImGui::Separator();

        RenderAgentRewardBars();

        ImGui::Separator();

        ImGui::Checkbox("Show View Maps", &droneUI_.showExplorationMap);
        ImGui::SameLine();
        ImGui::Checkbox("Show Drone View", &droneUI_.showInputImages);

        if (droneUI_.showExplorationMap) {
            RenderExplorationMapWindow(selectedDrone);
        }

        ImGui::BeginChild("DroneInfoScroll", ImVec2(0, ImGui::GetTextLineHeightWithSpacing() * 15),
                          true, ImGuiWindowFlags_HorizontalScrollbar);

        DroneInfoWidget::DrawStateInfo(selectedDrone);
        DroneInfoWidget::DrawNetworkInputInfo(selectedDrone);

        if (droneUI_.showInputImages) {
            RenderDroneViewTables(selectedDrone);
        }

        ImGui::EndChild();

        if (ImGui::CollapsingHeader("Rewards", ImGuiTreeNodeFlags_DefaultOpen)) {
            auto rewards = selectedDrone->GetEpisodeRewards().getBuffer();
            auto rewardsPos = static_cast<int>(selectedDrone->GetEpisodeRewards().getHead());
            RewardPlot::DrawWithLabel("Episodic Rewards", rewards, rewardsPos);

            DroneInfoWidget::DrawRewardComponents(selectedDrone);
        }

        ImGui::EndTabItem();
    }

    void RenderAgentRewardBars() {
        if (!drones_ || drones_->empty()) return;

        float minReward = 0.0f;
        float maxReward = 0.0f;
        std::vector<float> currentRewards;

        for (const auto& drone : *drones_) {
            auto rewards = drone->GetEpisodeRewards().getBuffer();
            float lastReward = rewards.empty() ? 0.0f : rewards.back();
            currentRewards.push_back(lastReward);
            minReward = std::min(minReward, lastReward);
            maxReward = std::max(maxReward, lastReward);
        }

        float range = maxReward - minReward;
        if (range < 1.0f) {
            minReward = -1.0f;
            maxReward = 1.0f;
        }

        for (size_t i = 0; i < drones_->size(); ++i) {
            const auto& drone = (*drones_)[i];
            std::string label = drone->GetAgentSubType() + "_" + std::to_string(drone->GetId());

            StatusState state = drone->GetDroneInGrid() ? StatusState::Active : StatusState::Error;
            StatusIndicator::DrawDot(state, 4.0f);
            ImGui::SameLine();

            ImGui::Text("%s", label.c_str());
            ImGui::SameLine(120);

            float reward = currentRewards[i];
            ImGui::PushID(static_cast<int>(i));
            ProgressBar::DrawSignedBar("##reward", reward, std::max(std::abs(minReward), std::abs(maxReward)),
                                       ImVec2(-1, 14));
            ImGui::PopID();
        }
    }

    // Tab 2: Environment
    void RenderEnvironmentTab() {
        if (!ImGui::BeginTabItem("Environment")) return;

        ImGui::BeginChild("EnvControlsScroll", ImVec2(0, 0), false);

        if (ImGui::CollapsingHeader("Fire Controls", ImGuiTreeNodeFlags_DefaultOpen)) {
            auto pattern = parameters_.GetFirePattern();
            bool showClusterParams = (pattern == FirePattern::Cluster || pattern == FirePattern::Random);
            bool showNumClusters = (pattern != FirePattern::Scattered);
            ConfigTable table("##FireControlsTable");
            table.Combo("Fire Pattern", &parameters_.fire_pattern_index_, kFirePatternNames, kFirePatternCount)
                .SliderFloat("Fire Percentage", &parameters_.fire_percentage_, -0.001f, 1.0f);
            if (showClusterParams) {
                table.SliderFloat("Fire Spread Probability", &parameters_.fire_spread_prob_, -0.001f, 1.0f)
                     .SliderFloat("Fire Noise", &parameters_.fire_noise_, -0.001f, 1.0f);
            }
            if (showNumClusters) {
                table.SliderInt("Number of Fire Clusters", &parameters_.num_fire_clusters_, 0, 20);
            }
            table.Button("Start New Fires", "StartFires", [this]() {
                    if (onStartFires_) onStartFires_();
                })
                .Tooltip("Click to start some fires");
        }

        if (ImGui::CollapsingHeader("Starting Positions", ImGuiTreeNodeFlags_DefaultOpen)) {
            ConfigTable("##StartControlTable")
                .SliderFloat("Groundstation Start %", &parameters_.groundstation_start_percentage_, 0.0f, 1.0f)
                .Tooltip("GS vs Random starting position ratio")
                .Checkbox("Adaptive Starting Position", &parameters_.adaptive_start_position_)
                .Tooltip("Calculate starting position based on current objective");
        }

        if (ImGui::CollapsingHeader("Goal Configuration", ImGuiTreeNodeFlags_DefaultOpen)) {
            ConfigTable("##GoalControlTable")
                .SliderFloat("Fire Goal Percentage", &parameters_.fire_goal_percentage_, 0.0f, 1.0f)
                .Tooltip("Percentage of goals set to fire locations")
                .Checkbox("Adaptive Goal Positions", &parameters_.adaptive_goal_position_)
                .Tooltip("Calculate goal position based on current objective");
        }

        if (ImGui::CollapsingHeader("Agent Behavior")) {
            ConfigTable("##FlyAgentControl")
                .Checkbox("Use Simple Policy", &parameters_.use_simple_policy_)
                .Tooltip("Agent doesn't stop at first goal during evaluation")
                .Checkbox("Water Limit", &parameters_.use_water_limit_)
                .Tooltip("Agent must recharge at the Groundstation");
        }

        ImGui::EndChild();
        ImGui::EndTabItem();
    }

    // Tab 3: Simulation (from SimulationControlsWindow tabs)
    void RenderSimulationTab() {
        if (!ImGui::BeginTabItem("Simulation")) return;

        ImGui::BeginChild("SimTabScroll", ImVec2(0, 0), false);

        if (ImGui::CollapsingHeader("Sim Info", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (gridmap_) {
                ImGui::Text("Particles: %d", gridmap_->GetNumParticles());
                ImGui::Text("Cells: %d", gridmap_->GetNumCells());
                ImGui::Text("Burning: %d", gridmap_->GetNumBurningCells());
                ImGui::Text("Burned: %d", gridmap_->GetNumBurnedCells());
                ImGui::Text("Burned: %.1f%%", gridmap_->PercentageBurned() * 100.0);
                ImGui::Text("Running Time: %s", FormatTime(static_cast<int>(runningTime_)).c_str());
                ImGui::Text("Size: %.1fkm x %.1fkm",
                            static_cast<double>(gridmap_->GetRows() * parameters_.GetCellSize()) / 1000.0,
                            static_cast<double>(gridmap_->GetCols() * parameters_.GetCellSize()) / 1000.0);
            }
            ImGui::Text("%.3f ms/frame (%.1f FPS)", 1000.0f / framerate_, framerate_);
        }

        if (ImGui::CollapsingHeader("Speed")) {
            if (delay_) {
                ImGui::SliderInt("Delay (ms)", delay_, 0, 500);
            }
            ImGui::SliderScalar("dt", ImGuiDataType_Double, &parameters_.dt_,
                                &parameters_.min_dt_, &parameters_.max_dt_, "%.8f", 1.0f);
        }

        if (ImGui::CollapsingHeader("Render Options")) {
            if (ImGui::BeginTable("RenderOptionsTable", 2, ImGuiTableFlags_SizingStretchSame)) {
                ImGui::TableNextColumn();
                if (ImGui::Checkbox("Render Grid", &parameters_.render_grid_)) {
                    if (renderer_) renderer_->SetFullRedraw();
                }

                ImGui::TableNextColumn();
                if (ImGui::Checkbox("Render Noise", &parameters_.has_noise_)) {
                    if (renderer_) {
                        renderer_->SetInitCellNoise();
                        renderer_->SetFullRedraw();
                    }
                }

                ImGui::TableNextColumn();
                if (ImGui::Checkbox("Lingering", &parameters_.lingering_)) {
                    if (renderer_) renderer_->SetFullRedraw();
                }

                ImGui::TableNextColumn();
                ImGui::Checkbox("Small Drones", &parameters_.show_small_drones_);

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Checkbox("Show Drone Circles", &parameters_.show_drone_circles_);

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                if (ImGui::Checkbox("Episode Termination Indicator", &parameters_.episode_termination_indicator_)) {
                    if (renderer_) renderer_->SetFlashScreen(parameters_.episode_termination_indicator_);
                }

                ImGui::TableNextColumn();
                if (ImGui::Checkbox("Render Particles", &parameters_.render_particles_)) {
                    if (renderer_) renderer_->SetFullRedraw();
                }

                ImGui::TableNextColumn();
                if (ImGui::Checkbox("Render Terrain Transition", &parameters_.render_terrain_transition)) {
                    if (renderer_) renderer_->SetFullRedraw();
                }

                ImGui::EndTable();
            }
        }

        ImGui::EndChild();
        ImGui::EndTabItem();
    }

    // Tab 4: Log
    void RenderLogTab() {
        if (!ImGui::BeginTabItem("Log")) return;

        double now = ImGui::GetTime();
        if (now >= nextReadTime_ && logReader_) {
            std::vector<std::string> newLines = logReader_->readNewLines();
            for (const auto& line : newLines) {
                logConsole_.AddLine(line.c_str());
            }
            nextReadTime_ = now + 0.5;
        }

        logConsole_.Draw("scrolling", 25.0f);

        if (ImGui::Button("Clear")) {
            logConsole_.Clear();
        }
        ImGui::SameLine();
        if (ImGui::Button("Reload All") || (resetConsole_ && *resetConsole_)) {
            if (logReader_) logReader_->reset();
            logConsole_.Clear();
            if (resetConsole_) *resetConsole_ = false;
        }

        ImGui::EndTabItem();
    }

    // Tab 5: Settings (includes ROSHAN-AI as collapsible section)
    void RenderSettingsTab(const py::dict& rlStatus) {
        if (!ImGui::BeginTabItem("Settings")) return;

        auto modelPath = rlStatus["model_path"].cast<std::string>();
        auto modelName = rlStatus["model_name"].cast<std::string>();

        if (ImGui::CollapsingHeader("Model Configuration")) {
            if (ImGui::Selectable("Model Path")) {
                if (onOpenFileDialog_) onOpenFileDialog_("model_path");
            }
            ImGui::SameLine();
            ImGui::TextColored(colors::kHint, "%s", modelPath.c_str());

            if (ImGui::Selectable("Model Name")) {
                if (onOpenFileDialog_) onOpenFileDialog_("model_name");
            }
            ImGui::SameLine();
            ImGui::TextColored(colors::kHint, "%s", modelName.c_str());

            auto autoTrain = rlStatus["auto_train"].cast<bool>();
            if (autoTrain) {
                ImGui::Separator();
                auto trainEpisodes = rlStatus["train_episodes"].cast<int>();
                auto maxEval = rlStatus["max_eval"].cast<int>();
                auto maxTrain = rlStatus["max_train"].cast<int>();

                ImGui::Text("Total Episodes: %d", trainEpisodes);
                ImGui::Text("Steps before Eval: %d", maxTrain);
                ImGui::Text("Eval Episodes: %d", maxEval);
            }
        }

        if (ImGui::CollapsingHeader("Keyboard Shortcuts")) {
            ImGui::BulletText("WASD - Move drone (manual control)");
            ImGui::BulletText("Mouse Wheel - Zoom in/out");
            ImGui::BulletText("Right Drag - Pan camera");
            ImGui::BulletText("Left Click - Ignite/Extinguish cell");
            ImGui::BulletText("Middle Click - Cell info popup");
            ImGui::BulletText("Pos1 - Home Position");
        }

        // ROSHAN-AI (folded into Settings)
        if (parameters_.llm_support_) {
            if (ImGui::CollapsingHeader("ROSHAN-AI")) {
                static char inputText[512] = "";
                ImGui::Text("Ask ROSHAN-AI a question:");
                ImGui::InputText("##ai_input", inputText, IM_ARRAYSIZE(inputText));

                if (ImGui::Button("Send")) {
                    if (userInput_) *userInput_ = inputText;
                    memset(inputText, 0, sizeof(inputText));
                }

                ImGui::BeginChild("ai_scrolling", ImVec2(0, ImGui::GetTextLineHeightWithSpacing() * 15), true);
                ImGui::PushTextWrapPos(0.0f);
                if (modelOutput_) ImGui::TextWrapped("%s", modelOutput_->c_str());
                ImGui::PopTextWrapPos();
                ImGui::EndChild();
            }
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        if (ImGui::Button("Reset Drones")) {
            if (onResetDrones_) onResetDrones_();
        }

        ImGui::EndTabItem();
    }

    // ===== Helper Methods =====
    void RenderDroneSelector() {
        for (const auto& drone : *drones_) {
            drone->SetActive(false);
        }

        std::vector<std::string> droneNames;
        for (const auto& drone : *drones_) {
            droneNames.push_back(drone->GetAgentSubType() + "_" + std::to_string(drone->GetId()));
        }

        ImGui::SetNextItemWidth(150);
        ImGui::Combo("##SelectDrone", &droneUI_.selectedDroneIndex,
            [](void* data, int idx, const char** outText) {
                auto& names = *static_cast<std::vector<std::string>*>(data);
                if (idx < 0 || idx >= static_cast<int>(names.size())) return false;
                *outText = names[idx].c_str();
                return true;
            }, &droneNames, static_cast<int>(droneNames.size()));

        ImGui::SameLine();
        ImGui::Checkbox("Manual Control", &parameters_.manual_control_);
    }

    void RenderExplorationMapWindow(const std::shared_ptr<FlyAgent>& drone) {
        if (!ImGui::Begin("Exploration & Fire Map")) {
            ImGui::End();
            return;
        }

        static bool showExploredMap = true;
        static bool showFireMap = false;
        static bool showStepExplore = false;
        static bool showTotalDroneView = false;
        static bool interpolated = true;

        ImGui::SliderInt("##size_slider", &parameters_.exploration_map_show_size_, 5, 200);
        ImGui::SameLine();
        ImGui::Checkbox("Interpolated", &interpolated);

        ImGui::Checkbox("TotalDroneView", &showTotalDroneView);
        ImGui::SameLine();
        ImGui::Checkbox("ExploredMapTotal", &showExploredMap);
        ImGui::Checkbox("StepExploreMap", &showStepExplore);
        ImGui::SameLine();
        ImGui::Checkbox("FireMap", &showFireMap);

        ImGui::Separator();

        if (ImGui::BeginTable("ViewTable", 1, ImGuiTableFlags_NoBordersInBody)) {
            float mapSize = static_cast<float>(parameters_.exploration_map_show_size_);
            float rowHeight = 5.0f * mapSize;

            if (showTotalDroneView && gridmap_) {
                ImGui::TableNextRow(ImGuiTableRowFlags_None, rowHeight);
                auto view = gridmap_->GetInterpolatedDroneView(
                    drone->GetGridPosition(), drone->GetViewRange(),
                    parameters_.exploration_map_show_size_, interpolated);
                GridVisualizer::Draw(*view, GridColorMode::TotalView,
                    static_cast<float>(parameters_.GetExplorationTime()));
                AddSeparator();
            }

            if (showExploredMap && gridmap_) {
                ImGui::TableNextRow(ImGuiTableRowFlags_None, rowHeight);
                auto map = gridmap_->GetExploredMap(parameters_.exploration_map_show_size_, interpolated);
                GridVisualizer::Draw(*map, GridColorMode::ExplorationInterpolated,
                    static_cast<float>(parameters_.GetExplorationTime()));
                AddSeparator();
            }

            if (showStepExplore && gridmap_) {
                ImGui::TableNextRow(ImGuiTableRowFlags_None, rowHeight);
                auto map = gridmap_->GetStepExploredMap(parameters_.exploration_map_show_size_, interpolated);
                GridVisualizer::Draw(*map, GridColorMode::ExplorationInterpolated2);
                AddSeparator();
            }

            if (showFireMap && gridmap_) {
                ImGui::TableNextRow(ImGuiTableRowFlags_None, rowHeight);
                auto map = gridmap_->GetFireMap(parameters_.exploration_map_show_size_, interpolated);
                GridVisualizer::Draw(*map, GridColorMode::FireInterpolated);
                AddSeparator();
            }

            ImGui::EndTable();
        }

        ImGui::End();
    }

    void RenderDroneViewTables(const std::shared_ptr<FlyAgent>& drone) {
        if (!gridmap_) return;

        if (ImGui::BeginTable("MapsTable", 2, ImGuiTableFlags_Borders)) {
            auto droneView = gridmap_->GetDroneView(drone->GetGridPosition(), drone->GetViewRange());

            ImGui::TableNextColumn();
            ImGui::Text("Terrain View");
            GridVisualizer::DrawTerrain((*droneView)[0], FireModelRenderer::GetMappedColor);

            ImGui::TableNextColumn();
            ImGui::Text("Fire View");
            GridVisualizer::Draw((*droneView)[1], GridColorMode::Fire);

            ImGui::EndTable();
        }

        auto viewRange = drone->GetViewRange();
        ImGui::Dummy(ImVec2(0.0f, static_cast<float>(viewRange) * 5.0f));
        ImGui::Separator();
    }

    static void AddSeparator() {
        ImGui::TableNextRow();
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
    }

    static std::string FormatTime(int totalSeconds) {
        if (totalSeconds < 60) {
            return std::to_string(totalSeconds) + "s";
        } else if (totalSeconds < 3600) {
            int mins = totalSeconds / 60;
            int secs = totalSeconds % 60;
            return std::to_string(mins) + "m " + std::to_string(secs) + "s";
        } else {
            int hours = totalSeconds / 3600;
            int mins = (totalSeconds % 3600) / 60;
            int secs = totalSeconds % 60;
            char buffer[32];
            snprintf(buffer, sizeof(buffer), "%02d:%02d:%02d", hours, mins, secs);
            return std::string(buffer);
        }
    }

    FireModelParameters& parameters_;
    std::shared_ptr<GridMap> gridmap_;
    std::shared_ptr<FireModelRenderer> renderer_;
    std::shared_ptr<std::vector<std::shared_ptr<FlyAgent>>> drones_;
    Mode mode_;

    bool visible_ = true;
    bool startupComplete_ = false;
    double nextReadTime_ = 0.0;

    // Speed tracking
    double lastSpeedUpdate_ = 0.0;
    int lastObsCount_ = 0;
    float stepsPerSec_ = 0.0f;

    // Training elapsed timer
    double trainingStartTime_ = 0.0;

    // Simulation controls
    bool* updateSimulation_ = nullptr;
    bool* renderSimulation_ = nullptr;
    int* delay_ = nullptr;
    float framerate_ = 60.0f;
    double runningTime_ = 0.0;
    std::vector<std::vector<int>>* rasterData_ = nullptr;

    LogConsoleWidget logConsole_;
    DroneUIState droneUI_;

    std::string* userInput_ = nullptr;
    std::string* modelOutput_ = nullptr;
    LogReader* logReader_ = nullptr;
    bool* resetConsole_ = nullptr;

    std::function<py::dict()> getRLStatus_;
    std::function<void(py::dict)> setRLStatus_;
    std::function<void()> onResetDrones_;
    std::function<void()> onStartFires_;
    std::function<void(const std::string&)> onOpenFileDialog_;
    std::function<void(std::vector<std::vector<int>>*, bool)> onResetGridMap_;
};

} // namespace ui

#endif // ROSHAN_CONTROLPANELWINDOW_H
