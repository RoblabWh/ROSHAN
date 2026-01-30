//
// DroneInfoWidget.h - Drone information display component for ImGui
//
// Displays drone state information, network inputs, and distances in table format.
// Extracted from PyConfig method in firemodel_imgui.cpp.
// Enhanced with status indicators and better visual styling.
//

#ifndef ROSHAN_DRONEINFOWIDGET_H
#define ROSHAN_DRONEINFOWIDGET_H

#include "imgui.h"
#include "ConfigTable.h"
#include "StatusIndicator.h"
#include "ProgressBar.h"
#include "../UITypes.h"
#include "reinforcementlearning/agents/fly_agent.h"
#include <memory>
#include <string>
#include <cmath>

namespace ui {

class DroneInfoWidget {
public:
    // Draw drone header with status indicator
    static void DrawHeader(const std::shared_ptr<FlyAgent>& drone) {
        if (!drone) return;

        // Status indicator
        StatusState state = drone->GetDroneInGrid() ? StatusState::Active : StatusState::Error;
        StatusIndicator::DrawDot(state, 6.0f);
        ImGui::SameLine();

        // Agent name with color
        ImGui::TextColored(colors::kHighlight, "%s_%d",
                          drone->GetAgentSubType().c_str(),
                          drone->GetId());

        // Inline status badge
        ImGui::SameLine(ImGui::GetWindowWidth() - 80);
        StatusIndicator::DrawBadge(drone->GetDroneInGrid() ? "In Grid" : "Out",
                                   drone->GetDroneInGrid() ? StatusState::Active : StatusState::Error);
    }

    // Draw drone state information table
    static void DrawStateInfo(const std::shared_ptr<FlyAgent>& drone) {
        if (!drone) return;

        // Collapsible section header
        if (!ImGui::CollapsingHeader("Drone State Information", ImGuiTreeNodeFlags_DefaultOpen)) {
            return;
        }

        if (ImGui::BeginTable("DroneInfoTable", 2, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
            ImGui::TableSetupColumn("Label", ImGuiTableColumnFlags_WidthFixed, 200.0f);
            ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableHeadersRow();

            // Agent Type with status
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("Agent Type");
            ImGui::TableNextColumn();
            StatusState state = drone->GetDroneInGrid() ? StatusState::Active : StatusState::Error;
            StatusIndicator::DrawDot(state, 4.0f);
            ImGui::SameLine();
            ImGui::Text("%s", drone->GetAgentSubType().c_str());

            // Out of Area Counter
            TableRow("GetOutOfAreaCounter", "%d", drone->GetOutOfAreaCounter());

            // Goal Position
            auto goalPos = drone->GetGoalPosition();
            TableRow("GetGoalPosition", "(%.6f, %.6f)", goalPos.first, goalPos.second);

            // Real Position
            auto realPos = drone->GetRealPosition();
            TableRow("GetRealPosition", "(%.6f, %.6f)", realPos.first, realPos.second);

            // Grid Position
            auto gridPos = drone->GetGridPosition();
            TableRow("GetGridPosition", "(%d, %d)", gridPos.first, gridPos.second);

            // Grid Position Double
            auto gridPosDouble = drone->GetLastState().GetGridPositionDouble();
            TableRow("GetGridPositionDouble", "(%.6f, %.6f)", gridPosDouble.first, gridPosDouble.second);

            // Grid Position Double Normalized
            auto gridPosDoubleNorm = drone->GetLastState().GetGridPositionDoubleNorm();
            TableRow("GetGridPositionDoubleNorm", "(%.6f, %.6f)", gridPosDoubleNorm.first, gridPosDoubleNorm.second);

            // Position Normalized Around Center
            auto posNormCenter = drone->GetLastState().GetPositionNormAroundCenter();
            TableRow("GetPositionNormAroundCenter", "(%.6f, %.6f)", posNormCenter.first, posNormCenter.second);

            // Distance to Nearest Boundary
            TableRow("GetDistanceToNearestBoundaryNorm", "%.6f",
                     drone->GetLastState().GetDistanceToNearestBoundaryNorm());

            // Drone In Grid
            TableRow("GetDroneInGrid", "%s", drone->GetDroneInGrid() ? "true" : "false");

            // Newly Explored Cells
            TableRow("GetNewlyExploredCells", "%d", drone->GetNewlyExploredCells());

            // Revisited Cells
            TableRow("GetLastTimeStepRevisitedCellsOfAllAgents", "%d", drone->GetRevisitedCells());

            ImGui::EndTable();
        }
    }

    // Draw network input information table
    static void DrawNetworkInputInfo(const std::shared_ptr<FlyAgent>& drone) {
        if (!drone) return;

        ImGui::Spacing();

        // Collapsible section
        if (!ImGui::CollapsingHeader("Network Input", ImGuiTreeNodeFlags_DefaultOpen)) {
            return;
        }

        if (ImGui::BeginTable("NetworkInputTable", 2, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
            ImGui::TableSetupColumn("Label", ImGuiTableColumnFlags_WidthFixed, 200.0f);
            ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableHeadersRow();

            // Velocity Normalized
            auto velocityNorm = drone->GetLastState().GetVelocityNorm();
            TableRow("VelocityNorm", "%.6f, %.6f", velocityNorm.first, velocityNorm.second);

            // Delta Goal
            auto deltaGoal = drone->GetLastState().GetDeltaGoal();
            TableRow("DeltaGoal", "(%.6f, %.6f)", deltaGoal.first, deltaGoal.second);

            // Cos/Sin to Goal
            auto cosSinGoal = drone->GetLastState().GetCosSinToGoal();
            TableRow("CosSinToGoal", "%.6f, %.6f", cosSinGoal.first, cosSinGoal.second);

            // Speed
            TableRow("Speed", "%.6f", drone->GetLastState().GetSpeed());

            // Distance to Goal
            TableRow("DistanceToGoal", "%.6f", drone->GetLastState().GetDistanceToGoal());

            // Distances to Other Agents
            auto distances = drone->GetLastState().GetDistancesToOtherAgents();
            auto masks = drone->GetLastState().GetDistancesMask();
            for (size_t i = 0; i < distances.size(); ++i) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Distance_%d", static_cast<int>(i));
                ImGui::TableNextColumn();
                ImGui::Text("[dx,dy]: (%.6f, %.6f)\n[dvx, dvy]: (%.6f, %.6f)\n(Mask: %d)",
                            distances[i][0], distances[i][1],
                            distances[i][2], distances[i][3],
                            static_cast<bool>(masks[i]));
            }

            ImGui::EndTable();
        }
    }

    // Draw reward components table with visual bars
    static void DrawRewardComponents(const std::shared_ptr<FlyAgent>& drone) {
        if (!drone) return;

        ImGui::Spacing();

        // Collapsible section
        if (!ImGui::CollapsingHeader("Reward Components", ImGuiTreeNodeFlags_DefaultOpen)) {
            return;
        }

        auto rewardComponents = drone->GetRewardComponents();

        // Find max abs value for scaling
        float maxAbsVal = 0.1f;  // Minimum to avoid division by zero
        for (const auto& [key, value] : rewardComponents) {
            maxAbsVal = std::max(maxAbsVal, std::abs(static_cast<float>(value)));
        }

        if (ImGui::BeginTable("RewardComponentTable", 3, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
            ImGui::TableSetupColumn("Component", ImGuiTableColumnFlags_WidthFixed, 150.0f);
            ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthFixed, 80.0f);
            ImGui::TableSetupColumn("Bar", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableHeadersRow();

            for (const auto& [key, value] : rewardComponents) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("%s", key.c_str());

                ImGui::TableNextColumn();
                // Color code the value
                ImVec4 valColor = value >= 0 ? colors::kStatusActive : colors::kStatusError;
                ImGui::TextColored(valColor, "%+.4f", value);

                ImGui::TableNextColumn();
                // Draw a mini progress bar
                float normalized = std::abs(static_cast<float>(value)) / maxAbsVal;
                ImVec4 barColor = value >= 0 ? colors::kStatusActive : colors::kStatusError;

                ImGui::PushStyleColor(ImGuiCol_PlotHistogram, barColor);
                ImGui::PushStyleColor(ImGuiCol_FrameBg, colors::kProgressBg);
                ImGui::ProgressBar(normalized, ImVec2(-1, 12), "");
                ImGui::PopStyleColor(2);
            }

            ImGui::EndTable();
        }
    }

    // Draw a compact summary view of drone stats
    static void DrawCompactSummary(const std::shared_ptr<FlyAgent>& drone) {
        if (!drone) return;

        auto gridPos = drone->GetGridPosition();
        auto goalPos = drone->GetGoalPosition();
        float distToGoal = drone->GetLastState().GetDistanceToGoal();

        // Single line summary
        StatusState state = drone->GetDroneInGrid() ? StatusState::Active : StatusState::Error;
        StatusIndicator::DrawDot(state, 4.0f);
        ImGui::SameLine();
        ImGui::Text("%s_%d | Pos:(%d,%d) | Goal:(%.1f,%.1f) | Dist:%.2f",
                   drone->GetAgentSubType().c_str(),
                   drone->GetId(),
                   gridPos.first, gridPos.second,
                   goalPos.first, goalPos.second,
                   distToGoal);
    }

private:
    // Helper to add a table row with formatted value
    template<typename... Args>
    static void TableRow(const char* label, const char* format, Args... args) {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Text("%s", label);
        ImGui::TableNextColumn();
        ImGui::Text(format, args...);
    }
};

} // namespace ui

#endif // ROSHAN_DRONEINFOWIDGET_H
