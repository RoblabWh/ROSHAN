//
// ProgressBar.h - Styled progress bar component for ImGui
//
// Provides customizable progress bars with:
// - Configurable colors and sizes
// - Optional percentage/value text overlay
// - Smooth animations
//

#ifndef ROSHAN_PROGRESSBAR_H
#define ROSHAN_PROGRESSBAR_H

#include "imgui.h"
#include "../UITypes.h"
#include <string>
#include <cstdio>

namespace ui {

class ProgressBar {
public:
    // Draw a basic progress bar with optional label
    static void Draw(const char* label, float progress, const ImVec2& size = ImVec2(-1, 0)) {
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, colors::kProgressFill);
        ImGui::PushStyleColor(ImGuiCol_FrameBg, colors::kProgressBg);

        if (label && label[0] != '\0' && label[0] != '#') {
            ImGui::Text("%s", label);
            ImGui::SameLine();
        }

        ImGui::ProgressBar(progress, size);

        ImGui::PopStyleColor(2);
    }

    // Draw a progress bar with current/total values displayed
    static void DrawWithValues(const char* label, int current, int total,
                               const ImVec2& size = ImVec2(-1, 0)) {
        float progress = total > 0 ? static_cast<float>(current) / static_cast<float>(total) : 0.0f;

        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, colors::kProgressFill);
        ImGui::PushStyleColor(ImGuiCol_FrameBg, colors::kProgressBg);

        if (label && label[0] != '\0' && label[0] != '#') {
            ImGui::Text("%s", label);
        }

        // Create overlay text
        char overlayBuf[64];
        snprintf(overlayBuf, sizeof(overlayBuf), "%d / %d", current, total);

        ImGui::ProgressBar(progress, size, overlayBuf);

        ImGui::PopStyleColor(2);
    }

    // Draw a progress bar with percentage displayed
    static void DrawWithPercent(const char* label, float progress,
                                const ImVec2& size = ImVec2(-1, 0)) {
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, colors::kProgressFill);
        ImGui::PushStyleColor(ImGuiCol_FrameBg, colors::kProgressBg);

        if (label && label[0] != '\0' && label[0] != '#') {
            ImGui::Text("%s", label);
        }

        char overlayBuf[32];
        snprintf(overlayBuf, sizeof(overlayBuf), "%.1f%%", progress * 100.0f);

        ImGui::ProgressBar(progress, size, overlayBuf);

        ImGui::PopStyleColor(2);
    }

    // Draw a horizontal bar (for reward comparison, etc.)
    static void DrawHorizontalBar(const char* label, float value, float minVal, float maxVal,
                                  const ImVec4& fillColor = colors::kProgressFill,
                                  const ImVec2& size = ImVec2(-1, 16)) {
        float normalized = 0.0f;
        if (maxVal > minVal) {
            normalized = (value - minVal) / (maxVal - minVal);
            normalized = normalized < 0.0f ? 0.0f : (normalized > 1.0f ? 1.0f : normalized);
        }

        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, fillColor);
        ImGui::PushStyleColor(ImGuiCol_FrameBg, colors::kProgressBg);

        if (label && label[0] != '\0' && label[0] != '#') {
            ImGui::Text("%s", label);
            ImGui::SameLine();
        }

        char overlayBuf[32];
        snprintf(overlayBuf, sizeof(overlayBuf), "%.2f", value);

        ImGui::ProgressBar(normalized, size, overlayBuf);

        ImGui::PopStyleColor(2);
    }

    // Draw a colored horizontal bar based on value sign (green for positive, red for negative)
    static void DrawSignedBar(const char* label, float value, float maxAbsVal,
                              const ImVec2& size = ImVec2(-1, 16)) {
        float normalized = maxAbsVal > 0 ? std::abs(value) / maxAbsVal : 0.0f;
        normalized = normalized > 1.0f ? 1.0f : normalized;

        ImVec4 fillColor = value >= 0 ? colors::kStatusActive : colors::kStatusError;

        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, fillColor);
        ImGui::PushStyleColor(ImGuiCol_FrameBg, colors::kProgressBg);

        if (label && label[0] != '\0' && label[0] != '#') {
            ImGui::Text("%s", label);
            ImGui::SameLine();
        }

        char overlayBuf[32];
        snprintf(overlayBuf, sizeof(overlayBuf), "%+.2f", value);

        ImGui::ProgressBar(normalized, size, overlayBuf);

        ImGui::PopStyleColor(2);
    }
};

} // namespace ui

#endif // ROSHAN_PROGRESSBAR_H
