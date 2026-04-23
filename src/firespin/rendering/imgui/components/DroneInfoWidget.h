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
#include "SchemaTable.h"
#include "StatusIndicator.h"
#include "ProgressBar.h"
#include "../UITypes.h"
#include "src/reinforcementlearning/feature_schema.h"
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

            const auto& last = drone->GetLastState();

            TableRow("Out of Area Counter", "%d", drone->GetOutOfAreaCounter());

            auto goalPos = drone->GetGoalPosition();
            TableRow("Goal Position", "(%.6f, %.6f)", goalPos.first, goalPos.second);

            auto realPos = drone->GetRealPosition();
            TableRow("Real Position", "(%.6f, %.6f)", realPos.first, realPos.second);

            auto gridPos = drone->GetGridPosition();
            TableRow("Grid Position", "(%d, %d)", gridPos.first, gridPos.second);

            auto gridPosDouble = state_features::GridPositionDouble(last);
            TableRow("Grid Position (double)", "(%.6f, %.6f)", gridPosDouble.first, gridPosDouble.second);

            auto gridPosDoubleNorm = state_features::GridPositionDoubleNorm(last);
            TableRow("Grid Position Norm", "(%.6f, %.6f)", gridPosDoubleNorm.first, gridPosDoubleNorm.second);

            auto posNormCenter = state_features::PositionNormAroundCenter(last);
            TableRow("Position Norm (around center)", "(%.6f, %.6f)", posNormCenter.first, posNormCenter.second);

            auto goalNorm = state_features::GoalPositionNorm(last);
            TableRow("Goal Position Norm", "(%.6f, %.6f)", goalNorm.first, goalNorm.second);

            auto orient = state_features::OrientationToGoal(last);
            TableRow("Orientation To Goal", "(%.6f, %.6f)", orient.first, orient.second);

            TableRow("Distance To Nearest Boundary Norm", "%.6f",
                     state_features::DistanceToNearestBoundaryNorm(last));

            TableRow("Drone In Grid", "%s", drone->GetDroneInGrid() ? "true" : "false");

            TableRow("Newly Explored Cells", "%d", drone->GetNewlyExploredCells());

            TableRow("Revisited Cells (last step)", "%d", drone->GetRevisitedCells());

            ImGui::EndTable();
        }
    }

    // Draw network input information table.
    // Schema-driven: iterates the supplied FeatureSchema and renders each group
    // by invoking the same extract lambdas the observation batcher uses.
    // Pass nullptr for `schema` to show a placeholder.
    static void DrawNetworkInputInfo(const std::shared_ptr<FlyAgent>& drone,
                                     const FeatureSchema* schema) {
        if (!drone) return;

        ImGui::Spacing();

        if (!ImGui::CollapsingHeader("Network Input", ImGuiTreeNodeFlags_DefaultOpen)) {
            return;
        }

        SchemaTable::Draw(schema, drone->GetLastState());
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
        float distToGoal = state_features::DistanceToGoal(drone->GetLastState());

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
