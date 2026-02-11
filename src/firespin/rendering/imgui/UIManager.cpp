//
// UIManager.cpp - Central UI orchestrator implementation
//

#include "UIManager.h"
#include <iostream>

namespace ui {

UIManager::UIManager(Mode mode, FireModelParameters& parameters)
    : parameters_(parameters)
    , mode_(mode)
    , logReader_("") {
    SetupWindows();
}

void UIManager::Init() {
    // Apply the default theme
    ThemeManager::ApplyTheme(state_.theme.currentTheme);

    // Note: model_path may not be available yet at init time.
    // The log reader path is set later via updateOnRLStatusChange().
    if (onGetRLStatus) {
        SetupCallbacks();
    }
}

void UIManager::SetupWindows() {
    // Create windows - they will be fully configured in SetupCallbacks
    simulationControls_ = std::make_unique<SimulationControlsWindow>(parameters_, nullptr, nullptr);
    parameterConfig_ = std::make_unique<ParameterConfigWindow>(parameters_, nullptr);
    startupWizard_ = std::make_unique<StartupWizard>(parameters_, nullptr, mode_);
    fileDialog_ = std::make_unique<FileDialogWindow>(parameters_, nullptr);
    cellPopups_ = std::make_unique<CellPopupManager>(parameters_, nullptr);
    rlStatus_ = std::make_unique<RLStatusWindow>(parameters_, nullptr, nullptr, mode_);
    controlPanel_ = std::make_unique<ControlPanelWindow>(parameters_, nullptr, nullptr, mode_);
    menuBar_ = std::make_unique<MenuBar>(parameters_, mode_);
}

void UIManager::SetupCallbacks() {
    // Wire up callbacks to windows

    // SimulationControlsWindow
    simulationControls_->SetResetCallback(onResetGridMap);

    // StartupWizard
    startupWizard_->SetRLStatusCallbacks(onGetRLStatus, onSetRLStatus);
    startupWizard_->SetGridMapCallbacks(onSetUniformRasterData, onResetGridMap);
    startupWizard_->SetOnComplete([this]() {
        state_.startup.modelStartupComplete = true;
        state_.visibility.simulationControls = true;
        state_.visibility.controlPanel = true;
        menuBar_->SetModelStartupComplete(true);
        rlStatus_->SetStartupComplete(true);
        controlPanel_->SetStartupComplete(true);
    });

    // FileDialogWindow
    fileDialog_->SetRLStatusCallbacks(onGetRLStatus, onSetRLStatus);
    fileDialog_->SetGridMapCallback(onResetGridMap);
    fileDialog_->SetOnMapLoaded([this]() {
        if (!state_.startup.modelStartupComplete) {
            state_.startup.modelStartupComplete = true;
            state_.visibility.simulationControls = true;
            state_.visibility.controlPanel = true;
            menuBar_->SetModelStartupComplete(true);
            rlStatus_->SetStartupComplete(true);
            controlPanel_->SetStartupComplete(true);
        }
        cellPopups_->RequestGridmapInit();
    });

    // CellPopupManager
    cellPopups_->SetNoiseCallback(onSetNoise);
    cellPopups_->SetResetGridMapCallback(onResetGridMap);
    cellPopups_->SetShowNoiseConfig(&state_.visibility.noiseConfig);

    // RLStatusWindow
    rlStatus_->SetRLStatusCallbacks(onGetRLStatus, onSetRLStatus);
    rlStatus_->SetResetDronesCallback(onResetDrones);
    rlStatus_->SetStartFiresCallback(startFires);
    rlStatus_->SetLogReader(&logReader_);
    rlStatus_->SetConsoleResetFlag(&resetConsole_);
    rlStatus_->SetFileDialogCallback([this](const std::string& key) {
        fileDialog_->OpenModelPath(key);
    });

    // ControlPanelWindow (merged window for GUI_RL mode)
    controlPanel_->SetRLStatusCallbacks(onGetRLStatus, onSetRLStatus);
    controlPanel_->SetResetDronesCallback(onResetDrones);
    controlPanel_->SetStartFiresCallback(startFires);
    controlPanel_->SetResetGridMapCallback(onResetGridMap);
    controlPanel_->SetLogReader(&logReader_);
    controlPanel_->SetConsoleResetFlag(&resetConsole_);
    controlPanel_->SetFileDialogCallback([this](const std::string& key) {
        fileDialog_->OpenModelPath(key);
    });

    // MenuBar
    menuBar_->SetShowControls(&state_.visibility.simulationControls);
    menuBar_->SetShowRLStatus(&state_.visibility.rlStatus);
    menuBar_->SetShowControlPanel(&state_.visibility.controlPanel);
    menuBar_->SetShowParameterConfig(&state_.visibility.parameterConfig);
    menuBar_->SetShowNoiseConfig(&state_.visibility.noiseConfig);
    menuBar_->SetShowDemoWindow(&showDemoWindow_);
    menuBar_->SetThemeCallback([this](Theme theme) {
        state_.theme.currentTheme = theme;
    });

    menuBar_->SetBrowserCallback([]() {
        OpenBrowser("http://localhost:3000/map.html");
    });
    menuBar_->SetBrowserSelectionCallback([this]() {
        browserSelectionFlag_ = true;
    });
    menuBar_->SetLoadMapCallback([this]() {
        fileDialog_->OpenLoadMap();
    });
    menuBar_->SetLoadUniformCallback([this]() {
        if (onSetUniformRasterData) onSetUniformRasterData();
    });
    menuBar_->SetLoadClassesCallback([this]() {
        if (onFillRasterWithEnum) onFillRasterWithEnum();
        if (onResetGridMap && currentRasterData_) onResetGridMap(currentRasterData_, true);
    });
    menuBar_->SetSaveMapCallback([this]() {
        fileDialog_->OpenSaveMap();
    });
    menuBar_->SetResetGridMapCallback([this]() {
        if (onResetGridMap && currentRasterData_) onResetGridMap(currentRasterData_, true);
    });
}

void UIManager::ImGuiSimulationControls(const std::shared_ptr<GridMap>& gridmap,
                                         std::vector<std::vector<int>>& currentRasterData,
                                         const std::shared_ptr<FireModelRenderer>& modelRenderer,
                                         bool& updateSimulation, bool& renderSimulation,
                                         int& delay, float framerate, double runningTime) {
    currentRasterData_ = &currentRasterData;

    // In GUI_RL mode, store sim data on controlPanel_ and skip SimulationControlsWindow
    if (mode_ == Mode::GUI_RL) {
        controlPanel_->SetGridMap(gridmap);
        controlPanel_->SetRenderer(modelRenderer);
        controlPanel_->SetUpdateSimulation(&updateSimulation);
        controlPanel_->SetRenderSimulation(&renderSimulation);
        controlPanel_->SetDelay(&delay);
        controlPanel_->SetFramerate(framerate);
        controlPanel_->SetRunningTime(runningTime);
        controlPanel_->SetRasterData(&currentRasterData);
        return;
    }

    // Pure GUI mode: use SimulationControlsWindow as before
    simulationControls_->SetGridMap(gridmap);
    simulationControls_->SetRenderer(modelRenderer);
    simulationControls_->SetUpdateSimulation(&updateSimulation);
    simulationControls_->SetRenderSimulation(&renderSimulation);
    simulationControls_->SetDelay(&delay);
    simulationControls_->SetFramerate(framerate);
    simulationControls_->SetRunningTime(runningTime);
    simulationControls_->SetRasterData(&currentRasterData);
    simulationControls_->SetVisible(state_.visibility.simulationControls);

    simulationControls_->Render();

    state_.visibility.simulationControls = simulationControls_->IsVisible();
}

void UIManager::Config(const std::shared_ptr<FireModelRenderer>& modelRenderer,
                       std::vector<std::vector<int>>& currentRasterData,
                       const std::shared_ptr<Wind>& wind) {
    currentRasterData_ = &currentRasterData;
    wind_ = wind;
    renderer_ = modelRenderer;

    // Show demo window if requested
    if (showDemoWindow_) {
        ImGui::ShowDemoWindow(&showDemoWindow_);
    }

    // Handle startup wizard
    if (!ImGuiOnStartup(modelRenderer, currentRasterData)) {
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(5, 7));

        if (state_.visibility.parameterConfig) {
            ShowParameterConfig(wind);
        }

        ImGui::PopStyleVar();
    }
}

void UIManager::FileHandling(const std::shared_ptr<DatasetHandler>& datasetHandler,
                             std::vector<std::vector<int>>& currentRasterData) {
    datasetHandler_ = datasetHandler;
    currentRasterData_ = &currentRasterData;

    // Update and render file dialog
    fileDialog_->SetDatasetHandler(datasetHandler);
    fileDialog_->SetRasterData(&currentRasterData);
    fileDialog_->Render();
}

void UIManager::PyConfig(std::string& userInput, std::string& modelOutput,
                         const std::shared_ptr<GridMap>& gridmap,
                         const std::shared_ptr<std::vector<std::shared_ptr<FlyAgent>>>& drones,
                         const std::shared_ptr<FireModelRenderer>& modelRenderer) {
    userInput_ = &userInput;
    modelOutput_ = &modelOutput;
    gridmap_ = gridmap;
    drones_ = drones;
    renderer_ = modelRenderer;

    // In GUI_RL mode, render the merged Control Panel instead of RLStatusWindow
    if (mode_ == Mode::GUI_RL) {
        controlPanel_->SetGridMap(gridmap);
        controlPanel_->SetRenderer(modelRenderer);
        controlPanel_->SetDrones(drones);
        controlPanel_->SetUserInput(&userInput);
        controlPanel_->SetModelOutput(&modelOutput);
        controlPanel_->SetVisible(state_.visibility.controlPanel && state_.startup.modelStartupComplete);

        controlPanel_->Render();

        if (state_.startup.modelStartupComplete) {
            state_.visibility.controlPanel = controlPanel_->IsVisible();
        }
        return;
    }

    // Non-GUI_RL mode: use RLStatusWindow as before
    rlStatus_->SetGridMap(gridmap);
    rlStatus_->SetRenderer(modelRenderer);
    rlStatus_->SetDrones(drones);
    rlStatus_->SetUserInput(&userInput);
    rlStatus_->SetModelOutput(&modelOutput);
    rlStatus_->SetVisible(state_.visibility.rlStatus && state_.startup.modelStartupComplete);

    rlStatus_->Render();

    if (state_.startup.modelStartupComplete) {
        state_.visibility.rlStatus = rlStatus_->IsVisible();
    }
}

void UIManager::ImGuiModelMenu(std::vector<std::vector<int>>& currentRasterData) {
    currentRasterData_ = &currentRasterData;
    menuBar_->Render();
}

void UIManager::ShowPopups(const std::shared_ptr<GridMap>& gridmap,
                           std::vector<std::vector<int>>& currentRasterData) {
    gridmap_ = gridmap;
    currentRasterData_ = &currentRasterData;

    cellPopups_->SetGridMap(gridmap);
    cellPopups_->SetRasterData(&currentRasterData);
    cellPopups_->Render();
}

bool UIManager::ImGuiOnStartup(const std::shared_ptr<FireModelRenderer>& modelRenderer,
                               std::vector<std::vector<int>>& currentRasterData) {
    if (parameters_.skip_gui_init_) {
        // Handle skip_gui_init case
        startupWizard_->SetRenderer(modelRenderer);
        startupWizard_->SetRasterData(&currentRasterData);
        startupWizard_->Render();
        return !state_.startup.modelStartupComplete;
    }

    if (!state_.startup.modelStartupComplete) {
        startupWizard_->SetRenderer(modelRenderer);
        startupWizard_->SetRasterData(&currentRasterData);
        startupWizard_->Render();

        // Check if file dialog needs to be opened
        if (startupWizard_->NeedsFileDialog()) {
            if (startupWizard_->IsModelLoadSelection()) {
                fileDialog_->OpenModelLoad();
            } else {
                fileDialog_->OpenLoadMap();
            }
            startupWizard_->ClearFileDialogFlag();
        }

        return true;
    }

    return false;
}

void UIManager::ShowParameterConfig(const std::shared_ptr<Wind>& wind) {
    wind_ = wind;
    parameterConfig_->SetWind(wind);
    parameterConfig_->SetVisible(state_.visibility.parameterConfig);
    parameterConfig_->Render();
    state_.visibility.parameterConfig = parameterConfig_->IsVisible();
}

void UIManager::HandleEvents(SDL_Event event, ImGuiIO* io,
                             const std::shared_ptr<GridMap>& gridmap,
                             const std::shared_ptr<FireModelRenderer>& modelRenderer,
                             const std::shared_ptr<DatasetHandler>& datasetHandler,
                             std::vector<std::vector<int>>& currentRasterData) {
    gridmap_ = gridmap;
    renderer_ = modelRenderer;
    datasetHandler_ = datasetHandler;
    currentRasterData_ = &currentRasterData;

    // Mouse button down - left click to ignite/extinguish
    if (event.type == SDL_MOUSEBUTTONDOWN && modelRenderer && !io->WantCaptureMouse &&
        event.button.button == SDL_BUTTON_LEFT) {
        int x, y;
        SDL_GetMouseState(&x, &y);
        auto gridPos = modelRenderer->ScreenToGridPosition(x, y);
        x = gridPos.first;
        y = gridPos.second;
        if (x >= 0 && x < gridmap->GetRows() && y >= 0 && y < gridmap->GetCols()) {
            if (gridmap->At(x, y).CanIgnite())
                gridmap->IgniteCell(x, y);
            else if (gridmap->GetCellState(x, y) == CellState::GENERIC_BURNING)
                gridmap->ExtinguishCell(x, y);
        }
    }
    // Mouse wheel - zoom toward cursor
    else if (event.type == SDL_MOUSEWHEEL && modelRenderer && !io->WantCaptureMouse) {
        int mx, my;
        SDL_GetMouseState(&mx, &my);
        if (event.wheel.y > 0)
            modelRenderer->ApplyZoom(1.1, mx, my);
        else if (event.wheel.y < 0)
            modelRenderer->ApplyZoom(0.9, mx, my);
    }
    // Mouse motion - pan with right button
    else if (event.type == SDL_MOUSEMOTION && modelRenderer && !io->WantCaptureMouse) {
        int x, y;
        Uint32 mouseState = SDL_GetMouseState(&x, &y);
        if (mouseState & SDL_BUTTON(SDL_BUTTON_RIGHT)) {
            modelRenderer->ChangeCameraPosition(-event.motion.xrel, -event.motion.yrel);
        }
    }
    // Middle click - open cell popup
    else if (event.type == SDL_MOUSEBUTTONDOWN && modelRenderer &&
             event.button.button == SDL_BUTTON_MIDDLE && !io->WantCaptureMouse) {
        int x, y;
        SDL_GetMouseState(&x, &y);
        auto cellPos = modelRenderer->ScreenToGridPosition(x, y);
        if (cellPos.first >= 0 && cellPos.first < gridmap->GetRows() &&
            cellPos.second >= 0 && cellPos.second < gridmap->GetCols()) {
            cellPopups_->AddPopup(cellPos.first, cellPos.second);
        }
    }
    // Window resize
    else if (event.type == SDL_WINDOWEVENT && modelRenderer) {
        if (event.window.event == SDL_WINDOWEVENT_SIZE_CHANGED) {
            modelRenderer->ResizeEvent();
        }
    }
    // Manual drone control (WASD + Space)
    if (event.type == SDL_KEYDOWN && parameters_.manual_control_ && !io->WantTextInput && onMoveDrone &&
        (event.key.keysym.sym == SDLK_w || event.key.keysym.sym == SDLK_a ||
         event.key.keysym.sym == SDLK_s || event.key.keysym.sym == SDLK_d ||
         event.key.keysym.sym == SDLK_SPACE)) {
        if (event.key.keysym.sym == SDLK_w)
            onMoveDrone(parameters_.active_drone_, -1, 0, 0);
        if (event.key.keysym.sym == SDLK_s)
            onMoveDrone(parameters_.active_drone_, 1, 0, 0);
        if (event.key.keysym.sym == SDLK_a)
            onMoveDrone(parameters_.active_drone_, 0, -1, 0);
        if (event.key.keysym.sym == SDLK_d)
            onMoveDrone(parameters_.active_drone_, 0, 1, 0);
        if (event.key.keysym.sym == SDLK_SPACE)
            onMoveDrone(parameters_.active_drone_, 0, 0, 1);
    }

    // Keyboard navigation (arrows, +/-, Home, F) - only when not typing in ImGui or manual drone control
    if (event.type == SDL_KEYDOWN && modelRenderer && !io->WantTextInput) {
        auto sym = event.key.keysym.sym;

        // Arrow keys for panning (skip if manual control is active to avoid conflict)
        if (!parameters_.manual_control_) {
            if (sym == SDLK_UP)    modelRenderer->ChangeCameraPosition(0, 3);
            if (sym == SDLK_DOWN)  modelRenderer->ChangeCameraPosition(0, -3);
            if (sym == SDLK_LEFT)  modelRenderer->ChangeCameraPosition(3, 0);
            if (sym == SDLK_RIGHT) modelRenderer->ChangeCameraPosition(-3, 0);
        }

        // +/- for zoom
        if (sym == SDLK_PLUS || sym == SDLK_KP_PLUS || sym == SDLK_EQUALS)
            modelRenderer->ApplyZoom(1.15);
        if (sym == SDLK_MINUS || sym == SDLK_KP_MINUS)
            modelRenderer->ApplyZoom(0.85);

        // Home to reset view
        if (sym == SDLK_HOME)
            modelRenderer->ResetCamera();

        // F to cycle focus: Fire -> Drone 0 -> Drone 1 -> ... -> Fire
        if (sym == SDLK_f && gridmap && drones_) {
            int numDrones = static_cast<int>(drones_->size());
            if (focusTarget_ == -1) {
                // Focus on fire
                modelRenderer->FocusOnFire();
                if (numDrones > 0)
                    focusTarget_ = 0;
            } else {
                // Focus on drone
                modelRenderer->FocusOnDrone(focusTarget_, drones_);
                focusTarget_++;
                if (focusTarget_ >= numDrones)
                    focusTarget_ = -1;
            }
        }
    }

    // Browser selection handling
    if (datasetHandler != nullptr) {
        if (datasetHandler->NewDataPointExists() && browserSelectionFlag_) {
            std::vector<std::vector<int>> rasterData;
            datasetHandler->LoadRasterDataFromJSON(rasterData);
            currentRasterData.clear();
            currentRasterData = rasterData;
            browserSelectionFlag_ = false;
            parameters_.map_is_uniform_ = false;
            if (onResetGridMap) onResetGridMap(&currentRasterData, true);
        }
    }
}

void UIManager::updateOnRLStatusChange() {
    py::dict rlStatus = onGetRLStatus();
    if (rlStatus.contains("model_path")) {
        auto modelPath = rlStatus["model_path"].cast<std::string>();
        auto logPath = (std::filesystem::path(modelPath) / "logs" / std::filesystem::path("logging.log")).string();
        logReader_.set_model_path(logPath);
    }
}

void UIManager::DefaultModeSelected() {
    state_.startup.modelStartupComplete = true;
    state_.visibility.simulationControls = true;
    state_.visibility.controlPanel = true;
    state_.visibility.parameterConfig = false;
    menuBar_->SetModelStartupComplete(true);
    rlStatus_->SetStartupComplete(true);
    controlPanel_->SetStartupComplete(true);
}

void UIManager::OpenBrowser(const std::string& url) {
    std::string command;

#if defined(_WIN32)
    command = "start ";
#elif defined(__APPLE__)
    command = "open ";
#elif defined(__linux__)
    command = "xdg-open ";
#endif

    if (system((command + url).c_str()) == -1) {
        std::cerr << "Error opening URL: " << url << std::endl;
    }
}

} // namespace ui
