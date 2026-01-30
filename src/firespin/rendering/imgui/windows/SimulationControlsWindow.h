//
// SimulationControlsWindow.h - Simulation controls window
//
// Displays simulation info, speed controls, and render options.
// Extracted from ImGuiSimulationControls method in firemodel_imgui.cpp.
//

#ifndef ROSHAN_SIMULATIONCONTROLSWINDOW_H
#define ROSHAN_SIMULATIONCONTROLSWINDOW_H

#include "IWindow.h"
#include "../UITypes.h"
#include "../components/ConfigTable.h"
#include "firespin/firemodel_gridmap.h"
#include "firespin/rendering/firemodel_renderer.h"
#include "firespin/model_parameters.h"
#include <memory>
#include <functional>

namespace ui {

class SimulationControlsWindow : public IWindow {
public:
    SimulationControlsWindow(FireModelParameters& parameters,
                             std::shared_ptr<GridMap> gridmap,
                             std::shared_ptr<FireModelRenderer> renderer)
        : parameters_(parameters)
        , gridmap_(std::move(gridmap))
        , renderer_(std::move(renderer)) {}

    void Render() override {
        if (!visible_) return;

        ImGui::Begin("Simulation Controls", &visible_, window_flags::kScrollable);

        RenderControlButtons();
        RenderTabs();

        ImGui::End();
    }

    bool IsVisible() const override { return visible_; }
    void SetVisible(bool visible) override { visible_ = visible; }
    const char* GetName() const override { return "Simulation Controls"; }

    // Setters for runtime data
    void SetUpdateSimulation(bool* update) { updateSimulation_ = update; }
    void SetRenderSimulation(bool* render) { renderSimulation_ = render; }
    void SetDelay(int* delay) { delay_ = delay; }
    void SetFramerate(float framerate) { framerate_ = framerate; }
    void SetRunningTime(double time) { runningTime_ = time; }
    void SetRasterData(std::vector<std::vector<int>>* data) { rasterData_ = data; }
    void SetResetCallback(std::function<void(std::vector<std::vector<int>>*, bool)> cb) { onReset_ = std::move(cb); }

    // Update references (needed when gridmap/renderer change)
    void SetGridMap(std::shared_ptr<GridMap> gridmap) { gridmap_ = std::move(gridmap); }
    void SetRenderer(std::shared_ptr<FireModelRenderer> renderer) { renderer_ = std::move(renderer); }

private:
    void RenderControlButtons() {
        // Start/Stop Simulation Button
        bool buttonColor = false;
        if (updateSimulation_ && *updateSimulation_) {
            ImGui::PushStyleColor(ImGuiCol_Button, colors::kButtonActive);
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, colors::kButtonActiveHovered);
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, colors::kButtonActivePressed);
            buttonColor = true;
        }
        if (ImGui::Button(*updateSimulation_ ? "Stop Simulation" : "Start Simulation")) {
            *updateSimulation_ = !*updateSimulation_;
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Click to %s the simulation.", *updateSimulation_ ? "stop" : "start");
        if (buttonColor) {
            ImGui::PopStyleColor(3);
        }
        ImGui::SameLine();

        // Start/Stop Rendering Button
        buttonColor = false;
        if (renderSimulation_ && *renderSimulation_) {
            ImGui::PushStyleColor(ImGuiCol_Button, colors::kButtonActive);
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, colors::kButtonActiveHovered);
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, colors::kButtonActivePressed);
            buttonColor = true;
        }
        if (ImGui::Button(*renderSimulation_ ? "Stop Rendering" : "Start Rendering")) {
            *renderSimulation_ = !*renderSimulation_;
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Click to %s rendering the simulation.", *renderSimulation_ ? "stop" : "start");
        if (buttonColor) {
            ImGui::PopStyleColor(3);
        }
        ImGui::SameLine();

        // Reset GridMap Button
        if (ImGui::Button("Reset GridMap")) {
            if (onReset_ && rasterData_) {
                onReset_(rasterData_, false);
            }
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Resets the GridMap to the initial state of the currently loaded map.");
    }

    void RenderTabs() {
        if (ImGui::BeginTabBar("SimStatus")) {
            RenderSimulationInfoTab();
            RenderSimulationSpeedTab();
            RenderRenderOptionsTab();
            ImGui::EndTabBar();
        }
    }

    void RenderSimulationInfoTab() {
        if (ImGui::BeginTabItem("Simulation Info")) {
            ImGui::Text("Simulation Analysis");

            if (gridmap_) {
                std::string analysisText;
                analysisText += "Number of particles: " + std::to_string(gridmap_->GetNumParticles()) + "\n";
                analysisText += "Number of cells: " + std::to_string(gridmap_->GetNumCells()) + "\n";
                analysisText += "Number of burning cells: " + std::to_string(gridmap_->GetNumBurningCells()) + "\n";
                analysisText += "Number of burned cells: " + std::to_string(gridmap_->GetNumBurnedCells()) + "\n";
                analysisText += "Percentage burned: " + std::to_string(gridmap_->PercentageBurned() * 100) + " %\n";
                analysisText += "Running Time: " + FormatTime(static_cast<int>(runningTime_)) + "\n";
                analysisText += "Height: " + std::to_string(gridmap_->GetRows() * parameters_.GetCellSize() / 1000) + "km | ";
                analysisText += "Width: " + std::to_string(gridmap_->GetCols() * parameters_.GetCellSize() / 1000) + "km";
                ImGui::TextWrapped("%s", analysisText.c_str());
            }

            ImGui::Spacing();
            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / framerate_, framerate_);
            ImGui::EndTabItem();
        }
    }

    void RenderSimulationSpeedTab() {
        if (ImGui::BeginTabItem("Simulation Speed")) {
            ImGui::Spacing();
            ImGui::Text("Simulation Delay");
            if (delay_) {
                ImGui::SliderInt("Delay (ms)", delay_, 0, 500);
            }
            ImGui::Spacing();
            ImGui::Text("Simulation Speed");
            ImGui::SliderScalar("dt", ImGuiDataType_Double, &parameters_.dt_,
                                &parameters_.min_dt_, &parameters_.max_dt_, "%.8f", 1.0f);
            ImGui::EndTabItem();
        }
    }

    void RenderRenderOptionsTab() {
        if (ImGui::BeginTabItem("Render Options")) {
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
            ImGui::EndTabItem();
        }
    }

    static std::string FormatTime(int seconds) {
        int hours = seconds / 3600;
        int minutes = (seconds % 3600) / 60;
        int secs = seconds % 60;
        char buffer[32];
        snprintf(buffer, sizeof(buffer), "%02d:%02d:%02d", hours, minutes, secs);
        return std::string(buffer);
    }

    FireModelParameters& parameters_;
    std::shared_ptr<GridMap> gridmap_;
    std::shared_ptr<FireModelRenderer> renderer_;

    bool visible_ = false;
    bool* updateSimulation_ = nullptr;
    bool* renderSimulation_ = nullptr;
    int* delay_ = nullptr;
    float framerate_ = 60.0f;
    double runningTime_ = 0.0;
    std::vector<std::vector<int>>* rasterData_ = nullptr;
    std::function<void(std::vector<std::vector<int>>*, bool)> onReset_;
};

} // namespace ui

#endif // ROSHAN_SIMULATIONCONTROLSWINDOW_H
