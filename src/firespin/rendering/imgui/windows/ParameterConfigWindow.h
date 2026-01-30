//
// ParameterConfigWindow.h - Simulation parameter configuration window
//
// Displays and allows editing of simulation parameters like particle settings,
// cell properties, and wind configuration.
// Extracted from ShowParameterConfig method in firemodel_imgui.cpp.
//

#ifndef ROSHAN_PARAMETERCONFIGWINDOW_H
#define ROSHAN_PARAMETERCONFIGWINDOW_H

#include "IWindow.h"
#include "../UITypes.h"
#include "firespin/model_parameters.h"
#include "firespin/wind.h"
#include <memory>

namespace ui {

class ParameterConfigWindow : public IWindow {
public:
    ParameterConfigWindow(FireModelParameters& parameters, std::shared_ptr<Wind> wind)
        : parameters_(parameters)
        , wind_(std::move(wind)) {}

    void Render() override {
        if (!visible_) return;

        ImGui::Begin("Simulation Parameters", &visible_, window_flags::kScrollable);

        if (parameters_.emit_convective_) {
            RenderVirtualParticlesSection();
        }

        if (parameters_.emit_radiation_) {
            RenderRadiationParticlesSection();
        }

        if (parameters_.map_is_uniform_) {
            RenderCellTerrainSection();
        }

        RenderWindSection();

        ImGui::End();
    }

    bool IsVisible() const override { return visible_; }
    void SetVisible(bool visible) override { visible_ = visible; }
    const char* GetName() const override { return "Parameter Config"; }

    void SetWind(std::shared_ptr<Wind> wind) { wind_ = std::move(wind); }

private:
    void RenderVirtualParticlesSection() {
        ImGui::SeparatorText("Virtual Particles");
        if (ImGui::TreeNodeEx("##Virtual Particles",
                              ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_SpanAvailWidth)) {
            ImGui::Text("Particle Lifetime");
            ImGui::SliderScalar("tau_mem", ImGuiDataType_Double, &parameters_.virtualparticle_tau_mem_,
                                &kTauMin, &kTauMax, "%.3f", 1.0f);

            ImGui::Text("Hotness");
            ImGui::SliderScalar("Y_st", ImGuiDataType_Double, &parameters_.virtualparticle_y_st_,
                                &kYStMin, &kYStMax, "%.3f", 1.0f);

            ImGui::Text("Ignition Threshold");
            ImGui::SliderScalar("Y_lim", ImGuiDataType_Double, &parameters_.virtualparticle_y_lim_,
                                &kYLimMin, &kYLimMax, "%.3f", 1.0f);

            ImGui::Text("Height of Emission");
            ImGui::SliderScalar("Lt", ImGuiDataType_Double, &parameters_.Lt_,
                                &kLtMin, &kLtMax, "%.3f", 1.0f);

            ImGui::Text("Scaling Factor");
            ImGui::SliderScalar("Fl", ImGuiDataType_Double, &parameters_.virtualparticle_fl_,
                                &kFlMin, &kFlMax, "%.3f", 1.0f);

            ImGui::Text("Constant");
            ImGui::SliderScalar("C0", ImGuiDataType_Double, &parameters_.virtualparticle_c0_,
                                &kC0Min, &kC0Max, "%.3f", 1.0f);

            ImGui::TreePop();
            ImGui::Spacing();
        }
    }

    void RenderRadiationParticlesSection() {
        ImGui::SeparatorText("Radiation Particles");
        if (ImGui::TreeNodeEx("##Radiation Particles", ImGuiTreeNodeFlags_SpanAvailWidth)) {
            ImGui::Text("Radiation Hotness");
            ImGui::SliderScalar("Y_st_r", ImGuiDataType_Double, &parameters_.radiationparticle_y_st_,
                                &kYStMin, &kYStMax, "%.3f", 1.0f);

            ImGui::Text("Radiation Ignition Threshold");
            ImGui::SliderScalar("Y_lim_r", ImGuiDataType_Double, &parameters_.radiationparticle_y_lim_,
                                &kYLimMin, &kYLimMax, "%.3f", 1.0f);

            ImGui::TreePop();
            ImGui::Spacing();
        }
    }

    void RenderCellTerrainSection() {
        ImGui::SeparatorText("Cell (Terrain)");
        if (ImGui::TreeNodeEx("##CellTerrain",
                              ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_SpanAvailWidth)) {
            ImGui::Spacing();
            ImGui::Text("Cell Size");
            ImGui::SameLine();
            ImGui::TextColored(colors::kHint, "(?)");
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("always press [Reset GridMap] manually after changing these values");

            ImGui::SliderScalar("##Cell Size", ImGuiDataType_Double, &parameters_.cell_size_,
                                &kCellSizeMin, &kCellSizeMax, "%.3f", 1.0f);

            ImGui::Text("Cell Ignition Threshold");
            ImGui::SliderScalar("##Cell Ignition Threshold", ImGuiDataType_Double,
                                &parameters_.cell_ignition_threshold_,
                                &kIgnitionMin, &kIgnitionMax, "%.3f", 1.0f);

            ImGui::Text("Cell Burning Duration");
            ImGui::SliderScalar("##Cell Burning Duration", ImGuiDataType_Double,
                                &parameters_.cell_burning_duration_,
                                &kBurningMin, &kBurningMax, "%.3f", 1.0f);

            ImGui::TreePop();
            ImGui::Spacing();
        }
    }

    void RenderWindSection() {
        ImGui::SeparatorText("Wind");
        if (ImGui::TreeNodeEx("##Wind",
                              ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_SpanAvailWidth)) {
            bool updateWind = false;

            ImGui::Text("Wind Speed");
            if (ImGui::SliderScalar("##Wind Speed", ImGuiDataType_Double, &parameters_.wind_uw_,
                                    &kWindSpeedMin, &kWindSpeedMax, "%.3f", 1.0f))
                updateWind = true;

            ImGui::Text("A");
            if (ImGui::SliderScalar("##A", ImGuiDataType_Double, &parameters_.wind_a_,
                                    &kWindAMin, &kWindAMax, "%.3f", 1.0f))
                updateWind = true;

            ImGui::Text("Wind Angle");
            if (ImGui::SliderScalar("##Wind Angle", ImGuiDataType_Double, &parameters_.wind_angle_,
                                    &kWindAngleMin, &kWindAngleMax, "%.1f", 1.0f))
                updateWind = true;

            if (updateWind && wind_) {
                wind_->UpdateWind();
            }

            ImGui::TreePop();
            ImGui::Spacing();
        }
    }

    FireModelParameters& parameters_;
    std::shared_ptr<Wind> wind_;
    bool visible_ = false;

    // Slider range constants
    static constexpr double kTauMin = 0.01;
    static constexpr double kTauMax = 100.0;
    static constexpr double kYStMin = 0.0;
    static constexpr double kYStMax = 1.0;
    static constexpr double kYLimMin = 0.1;
    static constexpr double kYLimMax = 0.3;
    static constexpr double kFlMin = 0.0;
    static constexpr double kFlMax = 10.0;
    static constexpr double kC0Min = 1.5;
    static constexpr double kC0Max = 2.0;
    static constexpr double kLtMin = 10.0;
    static constexpr double kLtMax = 100.0;
    static constexpr double kBurningMin = 1.0;
    static constexpr double kBurningMax = 200.0;
    static constexpr double kIgnitionMin = 1.0;
    static constexpr double kIgnitionMax = 500.0;
    static constexpr double kCellSizeMin = 1.0;
    static constexpr double kCellSizeMax = 100.0;
    static constexpr double kWindSpeedMin = 0.0;
    static constexpr double kWindSpeedMax = 35.0;
    static constexpr double kWindAMin = 0.2;
    static constexpr double kWindAMax = 0.5;
    static constexpr double kWindAngleMin = 0.0;
    static constexpr double kWindAngleMax = 6.28318530718; // 2*PI
};

} // namespace ui

#endif // ROSHAN_PARAMETERCONFIGWINDOW_H
