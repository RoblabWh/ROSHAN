//
// UITypes.h - Color constants, enums, and type definitions for ImGui UI
//

#ifndef ROSHAN_UITYPES_H
#define ROSHAN_UITYPES_H

#include "imgui.h"
#include <cstdint>

namespace ui {

// Spacing constants for consistent UI layout
namespace spacing {
    constexpr float kWindowPadding = 10.0f;
    constexpr float kFramePadding = 8.0f;
    constexpr float kItemSpacing = 8.0f;
    constexpr float kItemInnerSpacing = 6.0f;
    constexpr float kSectionSpacing = 15.0f;
    constexpr float kIndentSpacing = 20.0f;
}

// Status indicator states for drone/agent visualization
enum class StatusState {
    Active,     // Green - running normally
    Idle,       // Yellow - waiting/paused
    Error,      // Red - error/failed
    Inactive    // Gray - disabled
};

namespace colors {
    // Error/Warning/Info colors (for logs and status)
    constexpr ImVec4 kError = {1.0f, 0.314f, 0.314f, 1.0f};      // Red for errors
    constexpr ImVec4 kWarning = {1.0f, 0.706f, 0.0f, 1.0f};      // Orange for warnings
    constexpr ImVec4 kDebug = {0.706f, 0.706f, 1.0f, 1.0f};      // Light blue for debug
    constexpr ImVec4 kInfo = {0.392f, 0.902f, 0.392f, 1.0f};     // Green for info

    // Log line colors (ImU32 format for LogConsole)
    constexpr ImU32 kLogError = IM_COL32(255, 80, 80, 255);
    constexpr ImU32 kLogWarning = IM_COL32(255, 180, 0, 255);
    constexpr ImU32 kLogDebug = IM_COL32(180, 180, 255, 255);
    constexpr ImU32 kLogInfo = IM_COL32(100, 230, 100, 255);

    // UI highlight color (blue theme)
    constexpr ImVec4 kHighlight = {0.33f, 0.67f, 0.86f, 1.0f};

    // Button colors (active state - blue theme)
    constexpr ImVec4 kButtonActive = {0.35f, 0.6f, 0.85f, 1.0f};
    constexpr ImVec4 kButtonActiveHovered = {0.45f, 0.7f, 0.95f, 1.0f};
    constexpr ImVec4 kButtonActivePressed = {0.25f, 0.5f, 0.75f, 1.0f};

    // Startup wizard button colors
    constexpr ImVec4 kWizardButton = {0.45f, 0.6f, 0.85f, 1.0f};
    constexpr ImVec4 kWizardButtonHovered = {0.45f, 0.7f, 0.95f, 1.0f};
    constexpr ImVec4 kWizardButtonPressed = {0.45f, 0.5f, 0.75f, 1.0f};

    // Cell popup title color
    constexpr ImVec4 kPopupTitle = {0.5f, 0.5f, 1.0f, 1.0f};

    // Tooltip hint color (grey)
    constexpr ImVec4 kHint = {0.5f, 0.5f, 0.5f, 1.0f};

    // Plot highlight colors
    constexpr ImU32 kPlotLine = IM_COL32(255, 0, 0, 255);
    constexpr ImU32 kPlotPoint = IM_COL32(0, 255, 125, 255);

    // Status indicator colors
    constexpr ImVec4 kStatusActive = {0.2f, 0.8f, 0.3f, 1.0f};    // Green
    constexpr ImVec4 kStatusIdle = {1.0f, 0.8f, 0.0f, 1.0f};      // Yellow
    constexpr ImVec4 kStatusError = {1.0f, 0.3f, 0.3f, 1.0f};     // Red
    constexpr ImVec4 kStatusInactive = {0.5f, 0.5f, 0.5f, 1.0f};  // Gray

    // Progress bar colors
    constexpr ImVec4 kProgressFill = {0.33f, 0.67f, 0.86f, 1.0f};
    constexpr ImVec4 kProgressBg = {0.2f, 0.2f, 0.24f, 1.0f};

    // Mini-map colors
    constexpr ImU32 kMapVegetation = IM_COL32(34, 139, 34, 255);     // Forest green
    constexpr ImU32 kMapBurning = IM_COL32(255, 100, 0, 255);        // Orange-red
    constexpr ImU32 kMapBurned = IM_COL32(60, 60, 60, 255);          // Dark gray
    constexpr ImU32 kMapWater = IM_COL32(65, 105, 225, 255);         // Royal blue
    constexpr ImU32 kMapSealed = IM_COL32(180, 180, 180, 255);       // Light gray
    constexpr ImU32 kMapDrone = IM_COL32(255, 255, 0, 255);          // Yellow
    constexpr ImU32 kMapDroneOutline = IM_COL32(0, 0, 0, 255);       // Black

    // Helper to get status color from state
    inline ImVec4 GetStatusColor(StatusState state) {
        switch (state) {
            case StatusState::Active: return kStatusActive;
            case StatusState::Idle: return kStatusIdle;
            case StatusState::Error: return kStatusError;
            case StatusState::Inactive: default: return kStatusInactive;
        }
    }
}

// Startup wizard phases
enum class StartupPhase {
    ModeSelection,      // Select train/eval mode
    ModelPathCheck,     // Check if model folder is empty
    TrainingSetup,      // Configure training parameters
    MapSelection,       // Select map to load
    Complete            // Startup complete, normal operation
};

// Grid color modes for DrawGrid
enum class GridColorMode {
    Terrain,
    Fire,
    ExplorationInterpolated,
    FireInterpolated,
    ExplorationInterpolated2,
    TotalView
};

// Helper function to get color from log line content
inline ImU32 ColorFromLogLine(const std::string& line) {
    if (line.find("ERROR") != std::string::npos) return colors::kLogError;
    if (line.find("WARNING") != std::string::npos) return colors::kLogWarning;
    if (line.find("DEBUG") != std::string::npos) return colors::kLogDebug;
    return colors::kLogInfo;
}

// Window flags commonly used
namespace window_flags {
    constexpr ImGuiWindowFlags kScrollable =
        ImGuiWindowFlags_HorizontalScrollbar | ImGuiWindowFlags_AlwaysVerticalScrollbar;

    constexpr ImGuiWindowFlags kModal =
        ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse |
        ImGuiWindowFlags_NoMove | ImGuiWindowFlags_AlwaysAutoResize;

    constexpr ImGuiWindowFlags kPopup =
        ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoTitleBar;
}

} // namespace ui

#endif // ROSHAN_UITYPES_H
