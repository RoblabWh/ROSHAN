//
// CellPopupManager.h - Cell popup management
//
// Manages popups that appear when clicking on grid cells.
// Extracted from ShowPopups method in firemodel_imgui.cpp.
//

#ifndef ROSHAN_CELLPOPUPMANAGER_H
#define ROSHAN_CELLPOPUPMANAGER_H

#include "IWindow.h"
#include "../UITypes.h"
#include "../UIState.h"
#include "firespin/firemodel_gridmap.h"
#include "firespin/model_parameters.h"
#include <memory>
#include <functional>
#include <set>
#include <map>

namespace ui {

class CellPopupManager : public IWindow {
public:
    CellPopupManager(FireModelParameters& parameters,
                     std::shared_ptr<GridMap> gridmap)
        : parameters_(parameters)
        , gridmap_(std::move(gridmap)) {}

    void Render() override {
        RenderPopups();
        HandlePendingGridmapInit();
    }

    bool IsVisible() const override { return !state_.activePopups.empty(); }
    void SetVisible(bool /*visible*/) override { /* Managed by popup state */ }
    const char* GetName() const override { return "Cell Popups"; }

    // Add a popup at the given cell position
    void AddPopup(int x, int y) {
        state_.AddPopup(x, y);
    }

    // Check if there's a popup at the given position
    bool HasPopup(int x, int y) const {
        return state_.HasPopup(x, y);
    }

    // State access
    PopupState& GetState() { return state_; }

    // Request gridmap initialization (after map load)
    void RequestGridmapInit() { initGridmap_ = true; }

    // Callbacks
    void SetNoiseCallback(std::function<void(CellState, int, int)> cb) {
        onSetNoise_ = std::move(cb);
    }

    void SetResetGridMapCallback(std::function<void(std::vector<std::vector<int>>*, bool)> cb) {
        onResetGridMap_ = std::move(cb);
    }

    void SetGridMap(std::shared_ptr<GridMap> gridmap) { gridmap_ = std::move(gridmap); }
    void SetRasterData(std::vector<std::vector<int>>* data) { rasterData_ = data; }
    void SetShowNoiseConfig(bool* show) { showNoiseConfig_ = show; }

private:
    void RenderPopups() {
        for (auto it = state_.activePopups.begin(); it != state_.activePopups.end();) {
            char popupId[32];
            snprintf(popupId, sizeof(popupId), "Cell %d %d", it->first, it->second);

            // Position new popups at mouse location
            if (!state_.popupHasBeenOpened[*it]) {
                ImGui::SetNextWindowPos(ImGui::GetMousePos());
                state_.popupHasBeenOpened[*it] = true;
            }

            if (ImGui::Begin(popupId, nullptr, window_flags::kPopup)) {
                // Title with close button
                ImGui::TextColored(colors::kPopupTitle, "Cell %d %d", it->first, it->second);
                ImGui::SameLine();

                float windowWidth = ImGui::GetWindowContentRegionMax().x - ImGui::GetWindowContentRegionMin().x;
                float buttonWidth = ImGui::CalcTextSize("X").x + ImGui::GetStyle().FramePadding.x;
                ImGui::SetCursorPosX(windowWidth - buttonWidth);

                if (ImGui::Button("X")) {
                    state_.popupHasBeenOpened.erase(*it);
                    it = state_.activePopups.erase(it);
                    ImGui::End();
                    continue;
                }

                // Cell info from gridmap
                if (gridmap_) {
                    gridmap_->ShowCellInfo(it->first, it->second);
                }

                // Noise config (if enabled via View menu)
                if (showNoiseConfig_ && *showNoiseConfig_) {
                    RenderNoiseConfig(it->first, it->second);
                }
            }
            ImGui::End();
            ++it;
        }
    }

    void RenderNoiseConfig(int cellX, int cellY) {
        static int noiseLevel = 1;
        static int noiseSize = 1;

        ImGui::SliderInt("Noise Level", &noiseLevel, 1, 200);
        ImGui::SliderInt("Noise Size", &noiseSize, 1, 200);

        if (ImGui::Button("Reset Map")) {
            if (onResetGridMap_ && rasterData_) {
                onResetGridMap_(rasterData_, false);
            }
        }

        if (ImGui::Button("Add Noise")) {
            if (gridmap_ && onSetNoise_) {
                CellState state = gridmap_->GetCellState(cellX, cellY);
                onSetNoise_(state, noiseLevel, noiseSize);
            }
        }
    }

    void HandlePendingGridmapInit() {
        if (initGridmap_) {
            initGridmap_ = false;
            if (onResetGridMap_ && rasterData_) {
                onResetGridMap_(rasterData_, true);
            }
        }
    }

    FireModelParameters& parameters_;
    std::shared_ptr<GridMap> gridmap_;
    PopupState state_;

    bool initGridmap_ = false;
    std::vector<std::vector<int>>* rasterData_ = nullptr;
    bool* showNoiseConfig_ = nullptr;

    std::function<void(CellState, int, int)> onSetNoise_;
    std::function<void(std::vector<std::vector<int>>*, bool)> onResetGridMap_;
};

} // namespace ui

#endif // ROSHAN_CELLPOPUPMANAGER_H
