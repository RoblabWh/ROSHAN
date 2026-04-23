//
// PlannerInfoWidget.h - PlannerAgent information display component for ImGui
//
// Mirrors DroneInfoWidget but for the high-level planner: raw global state
// (fire summary, wind, drone water levels) plus a schema-driven Network Input
// section that reflects CreatePlannerAgentSchema() exactly.
//

#ifndef ROSHAN_PLANNERINFOWIDGET_H
#define ROSHAN_PLANNERINFOWIDGET_H

#include "imgui.h"
#include "ProgressBar.h"
#include "SchemaTable.h"
#include "StatusIndicator.h"
#include "../UITypes.h"
#include "src/reinforcementlearning/feature_schema.h"
#include "reinforcementlearning/agents/planner_agent.h"
#include <cmath>
#include <memory>

namespace ui {

class PlannerInfoWidget {
public:
    static void DrawHeader(const std::shared_ptr<PlannerAgent>& planner) {
        if (!planner) return;
        StatusIndicator::DrawDot(StatusState::Active, 6.0f);
        ImGui::SameLine();
        ImGui::TextColored(colors::kHighlight, "PlannerAgent_%d", planner->GetId());
    }

    static void DrawStateInfo(const std::shared_ptr<PlannerAgent>& planner) {
        if (!planner) return;

        if (!ImGui::CollapsingHeader("Planner State", ImGuiTreeNodeFlags_DefaultOpen)) {
            return;
        }

        const AgentState s = planner->GetLastState();

        if (ImGui::BeginTable("PlannerStateTable", 2,
                              ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
            ImGui::TableSetupColumn("Label", ImGuiTableColumnFlags_WidthFixed, 200.0f);
            ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableHeadersRow();

            Row("fire_count (log1p-norm)", "%.6f", s.fire_count);

            ImGui::TableNextRow();
            ImGui::TableNextColumn(); ImGui::TextUnformatted("fire_centroid");
            ImGui::TableNextColumn();
            // Sentinel emitted by BuildAgentState when no fires are burning;
            // see planner_agent.cpp:301-303.
            if (IsCentroidSentinel(s.fire_centroid)) {
                ImGui::TextDisabled("(none)");
            } else {
                ImGui::Text("(%.6f, %.6f)", s.fire_centroid.first, s.fire_centroid.second);
            }

            Row("wind_vector", "(%.4f, %.4f)", s.wind_vector.first, s.wind_vector.second);

            const int n_fires = s.fire_positions
                                    ? std::max(0, static_cast<int>(s.fire_positions->size()) - 1)
                                    : 0;
            Row("fires (excl. groundstation)", "%d", n_fires);

            ImGui::EndTable();
        }

        DrawWaterLevels(planner);
    }

    // Schema-driven network input view. Pass nullptr for `schema` to show a placeholder.
    static void DrawNetworkInputInfo(const std::shared_ptr<PlannerAgent>& planner,
                                     const FeatureSchema* schema) {
        if (!planner) return;

        ImGui::Spacing();
        if (!ImGui::CollapsingHeader("Planner Network Input", ImGuiTreeNodeFlags_DefaultOpen)) {
            return;
        }
        SchemaTable::Draw(schema, planner->GetLastState());
    }

private:
    static bool IsCentroidSentinel(const std::pair<double, double>& c) {
        return c.first <= -1.5 && c.second <= -1.5;
    }

    static void DrawWaterLevels(const std::shared_ptr<PlannerAgent>& planner) {
        const AgentState s = planner->GetLastState();
        if (!s.drone_water_levels || s.drone_water_levels->empty()) return;

        ImGui::Spacing();
        if (!ImGui::CollapsingHeader("Drone Water Levels", ImGuiTreeNodeFlags_DefaultOpen)) {
            return;
        }

        if (!ImGui::BeginTable("WaterLevelsTable", 3,
                               ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) return;
        ImGui::TableSetupColumn("Drone", ImGuiTableColumnFlags_WidthFixed, 80.0f);
        ImGui::TableSetupColumn("Fill", ImGuiTableColumnFlags_WidthFixed, 80.0f);
        ImGui::TableSetupColumn("Bar", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableHeadersRow();

        const auto& levels = *s.drone_water_levels;
        for (size_t i = 0; i < levels.size(); ++i) {
            const float v = static_cast<float>(levels[i]);
            ImGui::TableNextRow();
            ImGui::TableNextColumn(); ImGui::Text("%zu", i);
            ImGui::TableNextColumn(); ImGui::Text("%.3f", v);
            ImGui::TableNextColumn();
            ImGui::PushStyleColor(ImGuiCol_PlotHistogram, colors::kProgressFill);
            ImGui::PushStyleColor(ImGuiCol_FrameBg, colors::kProgressBg);
            ImGui::ProgressBar(v, ImVec2(-1, 12), "");
            ImGui::PopStyleColor(2);
        }
        ImGui::EndTable();
    }

    template<typename... Args>
    static void Row(const char* label, const char* fmt, Args... args) {
        ImGui::TableNextRow();
        ImGui::TableNextColumn(); ImGui::TextUnformatted(label);
        ImGui::TableNextColumn(); ImGui::Text(fmt, args...);
    }
};

} // namespace ui

#endif // ROSHAN_PLANNERINFOWIDGET_H
