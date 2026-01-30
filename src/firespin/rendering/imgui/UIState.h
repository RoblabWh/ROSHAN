//
// UIState.h - Encapsulated shared state for ImGui UI
//

#ifndef ROSHAN_UISTATE_H
#define ROSHAN_UISTATE_H

#include "UITypes.h"
#include "ThemeManager.h"
#include <set>
#include <map>
#include <string>
#include <utility>

namespace ui {

// Window visibility state - replaces the 15+ boolean flags
struct WindowVisibility {
    bool demoWindow = false;
    bool simulationControls = false;
    bool rlStatus = true;
    bool parameterConfig = false;
    bool noiseConfig = false;
};

// Theme preferences
struct ThemeState {
    Theme currentTheme = Theme::Dark;
};

// Startup state
struct StartupState {
    StartupPhase phase = StartupPhase::ModeSelection;
    bool modelStartupComplete = false;
    bool trainModeSelected = false;
    bool checkModelFolderEmpty = false;
    bool resetConsole = false;
};

// File dialog state
struct FileDialogState {
    bool open = false;
    bool loadMapFromDisk = false;
    bool saveMapToDisk = false;
    bool browserSelectionFlag = false;
    bool modelPathSelection = false;
    bool modelModeSelection = false;
    bool modelLoadSelection = false;
    bool initGridmap = false;
    std::string pathKey;
};

// Cell popup state
struct PopupState {
    std::set<std::pair<int, int>> activePopups;
    std::map<std::pair<int, int>, bool> popupHasBeenOpened;

    void AddPopup(int x, int y) {
        auto pos = std::make_pair(x, y);
        activePopups.insert(pos);
        popupHasBeenOpened.insert({pos, false});
    }

    void RemovePopup(int x, int y) {
        auto pos = std::make_pair(x, y);
        popupHasBeenOpened.erase(pos);
        activePopups.erase(pos);
    }

    bool HasPopup(int x, int y) const {
        return activePopups.count(std::make_pair(x, y)) > 0;
    }
};

// Drone UI state
struct DroneUIState {
    int selectedDroneIndex = 0;
    bool showExplorationMap = false;
    bool showInputImages = false;
};

// Console state
struct ConsoleState {
    bool resetConsole = false;
    double nextReadTime = 0.0;
};

// Central UI state container
struct UIState {
    WindowVisibility visibility;
    StartupState startup;
    FileDialogState fileDialog;
    PopupState popups;
    DroneUIState droneUI;
    ConsoleState console;
    ThemeState theme;

    // Reset to default state
    void Reset() {
        visibility = WindowVisibility{};
        startup = StartupState{};
        fileDialog = FileDialogState{};
        popups = PopupState{};
        droneUI = DroneUIState{};
        console = ConsoleState{};
        theme = ThemeState{};
    }

    // Called when startup is complete and default mode is selected
    void OnDefaultModeSelected() {
        startup.modelStartupComplete = true;
        visibility.simulationControls = true;
        visibility.parameterConfig = false;
    }

    // Check if in startup mode
    bool IsInStartup() const {
        return !startup.modelStartupComplete;
    }
};

} // namespace ui

#endif // ROSHAN_UISTATE_H
