//
// UIManager.h - Central UI orchestrator
//
// Manages all ImGui windows and replaces ImguiHandler.
// Provides the same interface as ImguiHandler for easy integration.
//

#ifndef ROSHAN_UIMANAGER_H
#define ROSHAN_UIMANAGER_H

#include "UITypes.h"
#include "UIState.h"
#include "UICallbacks.h"
#include "ThemeManager.h"
#include "MenuBar.h"
#include "windows/SimulationControlsWindow.h"
#include "windows/ParameterConfigWindow.h"
#include "windows/StartupWizard.h"
#include "windows/FileDialogWindow.h"
#include "windows/CellPopupManager.h"
#include "windows/RLStatusWindow.h"

#include "firespin/model_parameters.h"
#include "firespin/firemodel_gridmap.h"
#include "firespin/rendering/firemodel_renderer.h"
#include "firespin/wind.h"
#include "src/corine/dataset_handler.h"
#include "reinforcementlearning/agents/fly_agent.h"
#include "src/utils.h"
#include "externals/pybind11/include/pybind11/pybind11.h"

#include <SDL.h>
#include <memory>
#include <functional>
#include <string>

namespace py = pybind11;

namespace ui {

class UIManager {
public:
    UIManager(Mode mode, FireModelParameters& parameters);

    // Initialize the UI (called after callbacks are set)
    void Init();

    // Main rendering functions (called every frame)
    void ImGuiSimulationControls(const std::shared_ptr<GridMap>& gridmap,
                                  std::vector<std::vector<int>>& currentRasterData,
                                  const std::shared_ptr<FireModelRenderer>& modelRenderer,
                                  bool& updateSimulation, bool& renderSimulation,
                                  int& delay, float framerate, double runningTime);

    void Config(const std::shared_ptr<FireModelRenderer>& modelRenderer,
                std::vector<std::vector<int>>& currentRasterData,
                const std::shared_ptr<Wind>& wind);

    void FileHandling(const std::shared_ptr<DatasetHandler>& datasetHandler,
                      std::vector<std::vector<int>>& currentRasterData);

    void PyConfig(std::string& userInput, std::string& modelOutput,
                  const std::shared_ptr<GridMap>& gridmap,
                  const std::shared_ptr<std::vector<std::shared_ptr<FlyAgent>>>& drones,
                  const std::shared_ptr<FireModelRenderer>& modelRenderer);

    void ImGuiModelMenu(std::vector<std::vector<int>>& currentRasterData);

    void ShowPopups(const std::shared_ptr<GridMap>& gridmap,
                    std::vector<std::vector<int>>& currentRasterData);

    bool ImGuiOnStartup(const std::shared_ptr<FireModelRenderer>& modelRenderer,
                        std::vector<std::vector<int>>& currentRasterData);

    void ShowParameterConfig(const std::shared_ptr<Wind>& wind);

    void HandleEvents(SDL_Event event, ImGuiIO* io,
                      const std::shared_ptr<GridMap>& gridmap,
                      const std::shared_ptr<FireModelRenderer>& modelRenderer,
                      const std::shared_ptr<DatasetHandler>& datasetHandler,
                      std::vector<std::vector<int>>& currentRasterData);

    // Update on RL status change
    void updateOnRLStatusChange();

    // Set default mode selected
    void DefaultModeSelected();

    // Open browser helper
    static void OpenBrowser(const std::string& url);

    // Callbacks - same interface as ImguiHandler
    std::function<void()> onResetDrones;
    std::function<void()> onSetUniformRasterData;
    std::function<void(std::vector<std::vector<int>>*, bool)> onResetGridMap;
    std::function<void()> onFillRasterWithEnum;
    std::function<void(int, double, double, int)> onMoveDrone;
    std::function<void(CellState, int, int)> onSetNoise;
    std::function<void()> startFires;
    std::function<py::dict()> onGetRLStatus;
    std::function<void(py::dict)> onSetRLStatus;

private:
    void SetupWindows();
    void SetupCallbacks();

    FireModelParameters& parameters_;
    Mode mode_;

    // State
    UIState state_;
    bool showDemoWindow_ = false;
    bool browserSelectionFlag_ = false;
    bool resetConsole_ = false;

    // Log reader
    LogReader logReader_;

    // Pointers to external data (set during rendering calls)
    std::vector<std::vector<int>>* currentRasterData_ = nullptr;
    std::string* userInput_ = nullptr;
    std::string* modelOutput_ = nullptr;

    // Windows
    std::unique_ptr<SimulationControlsWindow> simulationControls_;
    std::unique_ptr<ParameterConfigWindow> parameterConfig_;
    std::unique_ptr<StartupWizard> startupWizard_;
    std::unique_ptr<FileDialogWindow> fileDialog_;
    std::unique_ptr<CellPopupManager> cellPopups_;
    std::unique_ptr<RLStatusWindow> rlStatus_;
    std::unique_ptr<MenuBar> menuBar_;

    // Shared pointers (updated during rendering)
    std::shared_ptr<GridMap> gridmap_;
    std::shared_ptr<FireModelRenderer> renderer_;
    std::shared_ptr<Wind> wind_;
    std::shared_ptr<DatasetHandler> datasetHandler_;
    std::shared_ptr<std::vector<std::shared_ptr<FlyAgent>>> drones_;
};

} // namespace ui

#endif // ROSHAN_UIMANAGER_H
