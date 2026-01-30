//
// ThemeManager.h - Theme definitions and application for ImGui UI
//
// Provides Dark and Light themes with modern styling including:
// - Rounded corners
// - Consistent spacing
// - Color-coordinated palettes
//

#ifndef ROSHAN_THEMEMANAGER_H
#define ROSHAN_THEMEMANAGER_H

#include "imgui.h"

namespace ui {

enum class Theme { Dark, Light };

class ThemeManager {
public:
    static void ApplyTheme(Theme theme) {
        currentTheme_ = theme;
        ApplyCommonStyle();

        switch (theme) {
            case Theme::Dark:
                ApplyDarkTheme();
                break;
            case Theme::Light:
                ApplyLightTheme();
                break;
        }
    }

    static Theme GetCurrentTheme() { return currentTheme_; }

    static void ApplyDarkTheme() {
        ImGuiStyle& style = ImGui::GetStyle();
        ImVec4* colors = style.Colors;

        // Background colors
        colors[ImGuiCol_WindowBg] = ImVec4(0.10f, 0.10f, 0.12f, 1.00f);
        colors[ImGuiCol_ChildBg] = ImVec4(0.08f, 0.08f, 0.10f, 1.00f);
        colors[ImGuiCol_PopupBg] = ImVec4(0.12f, 0.12f, 0.14f, 0.98f);

        // Border colors
        colors[ImGuiCol_Border] = ImVec4(0.28f, 0.28f, 0.32f, 1.00f);
        colors[ImGuiCol_BorderShadow] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);

        // Frame colors
        colors[ImGuiCol_FrameBg] = ImVec4(0.16f, 0.16f, 0.18f, 1.00f);
        colors[ImGuiCol_FrameBgHovered] = ImVec4(0.22f, 0.22f, 0.26f, 1.00f);
        colors[ImGuiCol_FrameBgActive] = ImVec4(0.28f, 0.28f, 0.32f, 1.00f);

        // Title colors
        colors[ImGuiCol_TitleBg] = ImVec4(0.08f, 0.08f, 0.10f, 1.00f);
        colors[ImGuiCol_TitleBgActive] = ImVec4(0.12f, 0.12f, 0.14f, 1.00f);
        colors[ImGuiCol_TitleBgCollapsed] = ImVec4(0.08f, 0.08f, 0.10f, 0.75f);

        // Menu bar
        colors[ImGuiCol_MenuBarBg] = ImVec4(0.12f, 0.12f, 0.14f, 1.00f);

        // Scrollbar
        colors[ImGuiCol_ScrollbarBg] = ImVec4(0.08f, 0.08f, 0.10f, 1.00f);
        colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.28f, 0.28f, 0.32f, 1.00f);
        colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.35f, 0.35f, 0.40f, 1.00f);
        colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.40f, 0.40f, 0.46f, 1.00f);

        // Check mark
        colors[ImGuiCol_CheckMark] = ImVec4(0.33f, 0.67f, 0.86f, 1.00f);

        // Slider
        colors[ImGuiCol_SliderGrab] = ImVec4(0.33f, 0.67f, 0.86f, 1.00f);
        colors[ImGuiCol_SliderGrabActive] = ImVec4(0.40f, 0.74f, 0.93f, 1.00f);

        // Button
        colors[ImGuiCol_Button] = ImVec4(0.20f, 0.40f, 0.60f, 1.00f);
        colors[ImGuiCol_ButtonHovered] = ImVec4(0.28f, 0.50f, 0.72f, 1.00f);
        colors[ImGuiCol_ButtonActive] = ImVec4(0.35f, 0.60f, 0.85f, 1.00f);

        // Header (used for collapsing headers, tree nodes, selectables)
        colors[ImGuiCol_Header] = ImVec4(0.20f, 0.40f, 0.60f, 0.60f);
        colors[ImGuiCol_HeaderHovered] = ImVec4(0.28f, 0.50f, 0.72f, 0.80f);
        colors[ImGuiCol_HeaderActive] = ImVec4(0.35f, 0.60f, 0.85f, 1.00f);

        // Separator
        colors[ImGuiCol_Separator] = ImVec4(0.28f, 0.28f, 0.32f, 1.00f);
        colors[ImGuiCol_SeparatorHovered] = ImVec4(0.33f, 0.67f, 0.86f, 0.78f);
        colors[ImGuiCol_SeparatorActive] = ImVec4(0.33f, 0.67f, 0.86f, 1.00f);

        // Resize grip
        colors[ImGuiCol_ResizeGrip] = ImVec4(0.28f, 0.28f, 0.32f, 0.40f);
        colors[ImGuiCol_ResizeGripHovered] = ImVec4(0.33f, 0.67f, 0.86f, 0.67f);
        colors[ImGuiCol_ResizeGripActive] = ImVec4(0.33f, 0.67f, 0.86f, 0.95f);

        // Tab
        colors[ImGuiCol_Tab] = ImVec4(0.16f, 0.16f, 0.18f, 1.00f);
        colors[ImGuiCol_TabHovered] = ImVec4(0.33f, 0.67f, 0.86f, 0.80f);
        colors[ImGuiCol_TabActive] = ImVec4(0.24f, 0.48f, 0.72f, 1.00f);
        colors[ImGuiCol_TabUnfocused] = ImVec4(0.12f, 0.12f, 0.14f, 1.00f);
        colors[ImGuiCol_TabUnfocusedActive] = ImVec4(0.18f, 0.36f, 0.54f, 1.00f);

        // Plot
        colors[ImGuiCol_PlotLines] = ImVec4(0.61f, 0.61f, 0.61f, 1.00f);
        colors[ImGuiCol_PlotLinesHovered] = ImVec4(1.00f, 0.43f, 0.35f, 1.00f);
        colors[ImGuiCol_PlotHistogram] = ImVec4(0.33f, 0.67f, 0.86f, 1.00f);
        colors[ImGuiCol_PlotHistogramHovered] = ImVec4(0.40f, 0.74f, 0.93f, 1.00f);

        // Table
        colors[ImGuiCol_TableHeaderBg] = ImVec4(0.14f, 0.14f, 0.16f, 1.00f);
        colors[ImGuiCol_TableBorderStrong] = ImVec4(0.28f, 0.28f, 0.32f, 1.00f);
        colors[ImGuiCol_TableBorderLight] = ImVec4(0.20f, 0.20f, 0.24f, 1.00f);
        colors[ImGuiCol_TableRowBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
        colors[ImGuiCol_TableRowBgAlt] = ImVec4(0.14f, 0.14f, 0.16f, 0.40f);

        // Text
        colors[ImGuiCol_Text] = ImVec4(0.92f, 0.92f, 0.94f, 1.00f);
        colors[ImGuiCol_TextDisabled] = ImVec4(0.50f, 0.50f, 0.54f, 1.00f);
        colors[ImGuiCol_TextSelectedBg] = ImVec4(0.33f, 0.67f, 0.86f, 0.35f);

        // Drag drop
        colors[ImGuiCol_DragDropTarget] = ImVec4(0.33f, 0.67f, 0.86f, 0.95f);

        // Nav
        colors[ImGuiCol_NavHighlight] = ImVec4(0.33f, 0.67f, 0.86f, 1.00f);
        colors[ImGuiCol_NavWindowingHighlight] = ImVec4(1.00f, 1.00f, 1.00f, 0.70f);
        colors[ImGuiCol_NavWindowingDimBg] = ImVec4(0.80f, 0.80f, 0.80f, 0.20f);

        // Modal
        colors[ImGuiCol_ModalWindowDimBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.60f);
    }

    static void ApplyLightTheme() {
        ImGuiStyle& style = ImGui::GetStyle();
        ImVec4* colors = style.Colors;

        // Background colors
        colors[ImGuiCol_WindowBg] = ImVec4(0.96f, 0.96f, 0.97f, 1.00f);
        colors[ImGuiCol_ChildBg] = ImVec4(0.94f, 0.94f, 0.95f, 1.00f);
        colors[ImGuiCol_PopupBg] = ImVec4(0.98f, 0.98f, 0.99f, 0.98f);

        // Border colors
        colors[ImGuiCol_Border] = ImVec4(0.78f, 0.78f, 0.82f, 1.00f);
        colors[ImGuiCol_BorderShadow] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);

        // Frame colors
        colors[ImGuiCol_FrameBg] = ImVec4(0.90f, 0.90f, 0.92f, 1.00f);
        colors[ImGuiCol_FrameBgHovered] = ImVec4(0.84f, 0.84f, 0.88f, 1.00f);
        colors[ImGuiCol_FrameBgActive] = ImVec4(0.78f, 0.78f, 0.84f, 1.00f);

        // Title colors
        colors[ImGuiCol_TitleBg] = ImVec4(0.88f, 0.88f, 0.90f, 1.00f);
        colors[ImGuiCol_TitleBgActive] = ImVec4(0.82f, 0.82f, 0.86f, 1.00f);
        colors[ImGuiCol_TitleBgCollapsed] = ImVec4(0.88f, 0.88f, 0.90f, 0.75f);

        // Menu bar
        colors[ImGuiCol_MenuBarBg] = ImVec4(0.92f, 0.92f, 0.94f, 1.00f);

        // Scrollbar
        colors[ImGuiCol_ScrollbarBg] = ImVec4(0.94f, 0.94f, 0.95f, 1.00f);
        colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.70f, 0.70f, 0.74f, 1.00f);
        colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.60f, 0.60f, 0.66f, 1.00f);
        colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.50f, 0.50f, 0.56f, 1.00f);

        // Check mark
        colors[ImGuiCol_CheckMark] = ImVec4(0.20f, 0.55f, 0.80f, 1.00f);

        // Slider
        colors[ImGuiCol_SliderGrab] = ImVec4(0.20f, 0.55f, 0.80f, 1.00f);
        colors[ImGuiCol_SliderGrabActive] = ImVec4(0.15f, 0.45f, 0.70f, 1.00f);

        // Button
        colors[ImGuiCol_Button] = ImVec4(0.20f, 0.55f, 0.80f, 1.00f);
        colors[ImGuiCol_ButtonHovered] = ImVec4(0.25f, 0.62f, 0.88f, 1.00f);
        colors[ImGuiCol_ButtonActive] = ImVec4(0.15f, 0.45f, 0.70f, 1.00f);

        // Header
        colors[ImGuiCol_Header] = ImVec4(0.20f, 0.55f, 0.80f, 0.40f);
        colors[ImGuiCol_HeaderHovered] = ImVec4(0.20f, 0.55f, 0.80f, 0.60f);
        colors[ImGuiCol_HeaderActive] = ImVec4(0.20f, 0.55f, 0.80f, 0.80f);

        // Separator
        colors[ImGuiCol_Separator] = ImVec4(0.78f, 0.78f, 0.82f, 1.00f);
        colors[ImGuiCol_SeparatorHovered] = ImVec4(0.20f, 0.55f, 0.80f, 0.78f);
        colors[ImGuiCol_SeparatorActive] = ImVec4(0.20f, 0.55f, 0.80f, 1.00f);

        // Resize grip
        colors[ImGuiCol_ResizeGrip] = ImVec4(0.70f, 0.70f, 0.74f, 0.40f);
        colors[ImGuiCol_ResizeGripHovered] = ImVec4(0.20f, 0.55f, 0.80f, 0.67f);
        colors[ImGuiCol_ResizeGripActive] = ImVec4(0.20f, 0.55f, 0.80f, 0.95f);

        // Tab
        colors[ImGuiCol_Tab] = ImVec4(0.88f, 0.88f, 0.90f, 1.00f);
        colors[ImGuiCol_TabHovered] = ImVec4(0.20f, 0.55f, 0.80f, 0.80f);
        colors[ImGuiCol_TabActive] = ImVec4(0.30f, 0.60f, 0.85f, 1.00f);
        colors[ImGuiCol_TabUnfocused] = ImVec4(0.92f, 0.92f, 0.94f, 1.00f);
        colors[ImGuiCol_TabUnfocusedActive] = ImVec4(0.75f, 0.85f, 0.95f, 1.00f);

        // Plot
        colors[ImGuiCol_PlotLines] = ImVec4(0.40f, 0.40f, 0.44f, 1.00f);
        colors[ImGuiCol_PlotLinesHovered] = ImVec4(1.00f, 0.35f, 0.25f, 1.00f);
        colors[ImGuiCol_PlotHistogram] = ImVec4(0.20f, 0.55f, 0.80f, 1.00f);
        colors[ImGuiCol_PlotHistogramHovered] = ImVec4(0.15f, 0.45f, 0.70f, 1.00f);

        // Table
        colors[ImGuiCol_TableHeaderBg] = ImVec4(0.85f, 0.85f, 0.88f, 1.00f);
        colors[ImGuiCol_TableBorderStrong] = ImVec4(0.70f, 0.70f, 0.74f, 1.00f);
        colors[ImGuiCol_TableBorderLight] = ImVec4(0.80f, 0.80f, 0.84f, 1.00f);
        colors[ImGuiCol_TableRowBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
        colors[ImGuiCol_TableRowBgAlt] = ImVec4(0.88f, 0.88f, 0.90f, 0.40f);

        // Text
        colors[ImGuiCol_Text] = ImVec4(0.12f, 0.12f, 0.14f, 1.00f);
        colors[ImGuiCol_TextDisabled] = ImVec4(0.50f, 0.50f, 0.54f, 1.00f);
        colors[ImGuiCol_TextSelectedBg] = ImVec4(0.20f, 0.55f, 0.80f, 0.35f);

        // Drag drop
        colors[ImGuiCol_DragDropTarget] = ImVec4(0.20f, 0.55f, 0.80f, 0.95f);

        // Nav
        colors[ImGuiCol_NavHighlight] = ImVec4(0.20f, 0.55f, 0.80f, 1.00f);
        colors[ImGuiCol_NavWindowingHighlight] = ImVec4(0.00f, 0.00f, 0.00f, 0.70f);
        colors[ImGuiCol_NavWindowingDimBg] = ImVec4(0.20f, 0.20f, 0.20f, 0.20f);

        // Modal
        colors[ImGuiCol_ModalWindowDimBg] = ImVec4(0.20f, 0.20f, 0.20f, 0.50f);
    }

private:
    static void ApplyCommonStyle() {
        ImGuiStyle& style = ImGui::GetStyle();

        // Rounding
        style.WindowRounding = 6.0f;
        style.ChildRounding = 4.0f;
        style.FrameRounding = 4.0f;
        style.PopupRounding = 4.0f;
        style.ScrollbarRounding = 4.0f;
        style.GrabRounding = 4.0f;
        style.TabRounding = 4.0f;

        // Padding and spacing
        style.WindowPadding = ImVec2(10.0f, 10.0f);
        style.FramePadding = ImVec2(8.0f, 4.0f);
        style.CellPadding = ImVec2(6.0f, 4.0f);
        style.ItemSpacing = ImVec2(8.0f, 6.0f);
        style.ItemInnerSpacing = ImVec2(6.0f, 4.0f);
        style.IndentSpacing = 20.0f;
        style.ScrollbarSize = 14.0f;
        style.GrabMinSize = 12.0f;

        // Borders
        style.WindowBorderSize = 1.0f;
        style.ChildBorderSize = 1.0f;
        style.PopupBorderSize = 1.0f;
        style.FrameBorderSize = 0.0f;
        style.TabBorderSize = 0.0f;

        // Alignment
        style.WindowTitleAlign = ImVec2(0.5f, 0.5f);
        style.WindowMenuButtonPosition = ImGuiDir_Right;
        style.ColorButtonPosition = ImGuiDir_Right;
        style.ButtonTextAlign = ImVec2(0.5f, 0.5f);
        style.SelectableTextAlign = ImVec2(0.0f, 0.0f);

        // Anti-aliasing
        style.AntiAliasedLines = true;
        style.AntiAliasedFill = true;
    }

    inline static Theme currentTheme_ = Theme::Dark;
};

} // namespace ui

#endif // ROSHAN_THEMEMANAGER_H
