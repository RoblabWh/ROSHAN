//
// StatusIndicator.h - Colored status dot/badge component for ImGui
//
// Provides visual status indicators with:
// - Colored circles/dots for status states
// - Optional labels and tooltips
// - Support for different status types
//

#ifndef ROSHAN_STATUSINDICATOR_H
#define ROSHAN_STATUSINDICATOR_H

#include "imgui.h"
#include "../UITypes.h"
#include <string>

namespace ui {

class StatusIndicator {
public:
    // Draw a simple colored status dot
    static void DrawDot(StatusState state, float radius = 5.0f) {
        ImVec4 color = colors::GetStatusColor(state);
        ImVec2 pos = ImGui::GetCursorScreenPos();
        ImDrawList* drawList = ImGui::GetWindowDrawList();

        // Center the dot vertically with text
        float textHeight = ImGui::GetTextLineHeight();
        pos.y += textHeight * 0.5f;
        pos.x += radius;

        drawList->AddCircleFilled(pos, radius, ImGui::ColorConvertFloat4ToU32(color));

        // Add a subtle outline
        drawList->AddCircle(pos, radius, IM_COL32(0, 0, 0, 100), 0, 1.0f);

        // Reserve space
        ImGui::Dummy(ImVec2(radius * 2.0f + 4.0f, textHeight));
    }

    // Draw a status dot with a label
    static void DrawWithLabel(const char* label, StatusState state, float radius = 5.0f) {
        DrawDot(state, radius);
        ImGui::SameLine();
        ImGui::Text("%s", label);
    }

    // Draw a status dot with tooltip on hover
    static void DrawWithTooltip(StatusState state, const char* tooltip, float radius = 5.0f) {
        ImVec2 startPos = ImGui::GetCursorPos();
        DrawDot(state, radius);

        // Check hover for tooltip
        ImVec2 endPos = ImGui::GetCursorPos();
        ImGui::SetCursorPos(startPos);
        ImGui::InvisibleButton("##status_btn", ImVec2(radius * 2.0f + 4.0f, ImGui::GetTextLineHeight()));
        if (ImGui::IsItemHovered() && tooltip) {
            ImGui::SetTooltip("%s", tooltip);
        }
        ImGui::SetCursorPos(endPos);
    }

    // Draw an inline status indicator (same line as previous item)
    static void DrawInline(StatusState state, float radius = 4.0f) {
        ImGui::SameLine();
        ImVec4 color = colors::GetStatusColor(state);
        ImVec2 pos = ImGui::GetCursorScreenPos();
        ImDrawList* drawList = ImGui::GetWindowDrawList();

        float textHeight = ImGui::GetTextLineHeight();
        pos.y += textHeight * 0.5f;
        pos.x += radius;

        drawList->AddCircleFilled(pos, radius, ImGui::ColorConvertFloat4ToU32(color));
        drawList->AddCircle(pos, radius, IM_COL32(0, 0, 0, 100), 0, 1.0f);

        ImGui::Dummy(ImVec2(radius * 2.0f + 4.0f, textHeight));
    }

    // Draw a status badge (pill-shaped with text)
    static void DrawBadge(const char* text, StatusState state) {
        ImVec4 bgColor = colors::GetStatusColor(state);
        ImVec4 textColor = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);  // White text

        ImGui::PushStyleColor(ImGuiCol_Button, bgColor);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, bgColor);
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, bgColor);
        ImGui::PushStyleColor(ImGuiCol_Text, textColor);

        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 10.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(8.0f, 2.0f));

        ImGui::SmallButton(text);

        ImGui::PopStyleVar(2);
        ImGui::PopStyleColor(4);
    }

    // Draw multiple status dots in a row (for multi-agent status)
    static void DrawMultiple(const std::vector<StatusState>& states, float radius = 4.0f, float spacing = 2.0f) {
        ImVec2 startPos = ImGui::GetCursorScreenPos();
        ImDrawList* drawList = ImGui::GetWindowDrawList();
        float textHeight = ImGui::GetTextLineHeight();

        float x = startPos.x + radius;
        float y = startPos.y + textHeight * 0.5f;

        for (size_t i = 0; i < states.size(); ++i) {
            ImVec4 color = colors::GetStatusColor(states[i]);
            drawList->AddCircleFilled(ImVec2(x, y), radius, ImGui::ColorConvertFloat4ToU32(color));
            drawList->AddCircle(ImVec2(x, y), radius, IM_COL32(0, 0, 0, 100), 0, 1.0f);
            x += radius * 2.0f + spacing;
        }

        float totalWidth = static_cast<float>(states.size()) * (radius * 2.0f + spacing);
        ImGui::Dummy(ImVec2(totalWidth, textHeight));
    }

    // Get status state from boolean conditions
    static StatusState GetState(bool isActive, bool hasError = false) {
        if (hasError) return StatusState::Error;
        if (isActive) return StatusState::Active;
        return StatusState::Idle;
    }
};

} // namespace ui

#endif // ROSHAN_STATUSINDICATOR_H
