//
// RewardPlot.h - Reward buffer visualization component for ImGui
//
// Renders a reward history plot with min/max/avg legend and highlighted current position.
// Extracted from DrawBuffer static method in firemodel_imgui.cpp.
//

#ifndef ROSHAN_REWARDPLOT_H
#define ROSHAN_REWARDPLOT_H

#include "imgui.h"
#include "../UITypes.h"
#include <vector>
#include <algorithm>
#include <numeric>

namespace ui {

class RewardPlot {
public:
    // Draw a reward buffer plot
    static void Draw(const std::vector<float>& buffer, int bufferPos, float height = 150.0f) {
        if (buffer.empty()) {
            ImGui::Text("No rewards data available.");
            return;
        }

        float minValue = *std::min_element(buffer.begin(), buffer.end());
        float maxValue = *std::max_element(buffer.begin(), buffer.end());
        float avgValue = std::accumulate(buffer.begin(), buffer.end(), 0.0f) / static_cast<float>(buffer.size());

        // Display legend
        ImGui::Text("Min: %.2f | Max: %.2f | Avg: %.2f", minValue, maxValue, avgValue);

        // Plot the graph
        ImGui::PlotLines("", buffer.data(), static_cast<int>(buffer.size()),
                         0, nullptr, minValue, maxValue, ImVec2(0, height));

        // Highlight the current position
        if (bufferPos >= 0 && bufferPos < static_cast<int>(buffer.size())) {
            ImDrawList* drawList = ImGui::GetWindowDrawList();
            ImVec2 graphPos = ImGui::GetItemRectMin();
            ImVec2 graphSize = ImGui::GetItemRectSize();

            float x = graphPos.x + (static_cast<float>(bufferPos) / (static_cast<float>(buffer.size()) - 1)) * graphSize.x;

            // Avoid division by zero
            float range = maxValue - minValue;
            float y = graphPos.y + graphSize.y;
            if (range > 0.0001f) {
                y = graphPos.y + graphSize.y - ((buffer[bufferPos] - minValue) / range * graphSize.y);
            }

            // Draw vertical line at current position
            drawList->AddLine(ImVec2(x, graphPos.y), ImVec2(x, graphPos.y + graphSize.y),
                              colors::kPlotLine, 2.0f);

            // Draw point at current value
            drawList->AddCircleFilled(ImVec2(x, y), 3.0f, colors::kPlotPoint);

            // Tooltip on hover
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Value: %.2f\nIndex: %d", buffer[bufferPos], bufferPos);
            }
        }
    }

    // Draw with label
    static void DrawWithLabel(const char* label, const std::vector<float>& buffer,
                              int bufferPos, float height = 150.0f) {
        ImGui::TextColored(colors::kHighlight, "%s", label);
        Draw(buffer, bufferPos, height);
    }
};

} // namespace ui

#endif // ROSHAN_REWARDPLOT_H
