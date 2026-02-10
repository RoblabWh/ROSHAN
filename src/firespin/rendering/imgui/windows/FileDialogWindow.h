//
// FileDialogWindow.h - File dialog handling
//
// Manages file dialogs for loading/saving maps and model paths.
// Extracted from FileHandling method in firemodel_imgui.cpp.
//

#ifndef ROSHAN_FILEDIALOGWINDOW_H
#define ROSHAN_FILEDIALOGWINDOW_H

#include "IWindow.h"
#include "../UIState.h"
#include "firespin/model_parameters.h"
#include "src/corine/dataset_handler.h"
#include "src/utils.h"
#include "externals/ImGuiFileDialog/ImGuiFileDialog.h"
#include "externals/pybind11/include/pybind11/pybind11.h"
#include <memory>
#include <functional>
#include <optional>

namespace py = pybind11;

namespace ui {

class __attribute__((visibility("hidden"))) FileDialogWindow : public IWindow {
public:
    FileDialogWindow(FireModelParameters& parameters,
                     std::shared_ptr<DatasetHandler> datasetHandler)
        : parameters_(parameters)
        , datasetHandler_(std::move(datasetHandler)) {}

    void Render() override {
        if (!state_.open) return;

        ConfigureAndOpenDialog();
        ProcessDialogResult();
    }

    bool IsVisible() const override { return state_.open; }
    void SetVisible(bool visible) override { state_.open = visible; }
    const char* GetName() const override { return "File Dialog"; }

    // State access
    FileDialogState& GetState() { return state_; }

    // Open dialog for specific purposes
    void OpenLoadMap() {
        state_.open = true;
        state_.loadMapFromDisk = true;
    }

    void OpenSaveMap() {
        state_.open = true;
        state_.saveMapToDisk = true;
    }

    void OpenModelPath(const std::string& key) {
        state_.open = true;
        state_.modelPathSelection = true;
        state_.pathKey = key;
    }

    void OpenModelLoad() {
        state_.open = true;
        state_.modelLoadSelection = true;
    }

    // Callbacks
    void SetRLStatusCallbacks(std::function<py::dict()> get, std::function<void(py::dict)> set) {
        getRLStatus_ = std::move(get);
        setRLStatus_ = std::move(set);
    }

    void SetGridMapCallback(std::function<void(std::vector<std::vector<int>>*, bool)> cb) {
        onResetGridMap_ = std::move(cb);
    }

    void SetOnMapLoaded(std::function<void()> cb) { onMapLoaded_ = std::move(cb); }
    void SetOnConsoleReset(std::function<void()> cb) { onConsoleReset_ = std::move(cb); }

    void SetRasterData(std::vector<std::vector<int>>* data) { rasterData_ = data; }
    void SetDatasetHandler(std::shared_ptr<DatasetHandler> handler) { datasetHandler_ = std::move(handler); }

private:
    void ConfigureAndOpenDialog() {
        IGFD::FileDialogConfig config;
        std::optional<std::string> filters;
        std::string title;
        std::string key;

        if (state_.modelPathSelection) {
            auto rootPath = get_project_path("root_path", {});
            config.path = rootPath.string();
            if (state_.pathKey == "model_path") {
                title = "Change Model Path Folder";
                filters.reset();
                key = "ChooseFolderDlgKey";
            } else if (state_.pathKey == "model_name") {
                title = "Choose Model Name";
                key = "ChooseFileDlgKey";
                filters = ".pt";
            }
        } else if (state_.modelLoadSelection) {
            title = "Choose Model to Load";
            auto rootPath = get_project_path("root_path", {});
            config.path = rootPath.string();
            filters = ".pt";
            key = "ChooseFileDlgKey";
        } else {
            title = "Choose File or Filename";
            auto mapsPath = get_project_path("maps_directory", {});
            config.path = mapsPath.string();
            filters = ".tif";
            key = "ChooseFileDlgKey";
        }

        if (filters) {
            ImGuiFileDialog::Instance()->OpenDialog(key, title, filters->c_str(), config);
        } else {
            ImGuiFileDialog::Instance()->OpenDialog(key, title, nullptr, config);
        }

        currentDialogKey_ = key;
    }

    void ProcessDialogResult() {
        if (!ImGuiFileDialog::Instance()->Display(currentDialogKey_)) {
            return;
        }

        if (ImGuiFileDialog::Instance()->IsOk()) {
            std::string filePathName = ImGuiFileDialog::Instance()->GetFilePathName();
            std::string filePath = ImGuiFileDialog::Instance()->GetCurrentPath();
            std::string fileName = ImGuiFileDialog::Instance()->GetCurrentFileName();

            if (state_.loadMapFromDisk) {
                HandleLoadMap(filePathName);
            } else if (state_.saveMapToDisk) {
                HandleSaveMap(filePathName);
            } else if (state_.modelPathSelection) {
                HandleModelPathSelection(filePath, fileName);
            } else if (state_.modelLoadSelection) {
                HandleModelLoadSelection(filePath, fileName);
            }
        }

        ImGuiFileDialog::Instance()->Close();
        ResetState();
    }

    void HandleLoadMap(const std::string& filePathName) {
        if (!datasetHandler_) return;

        datasetHandler_->LoadMap(filePathName);
        std::vector<std::vector<int>> rasterData;
        datasetHandler_->LoadMapDataset(rasterData);

        if (rasterData_) {
            rasterData_->clear();
            *rasterData_ = rasterData;
        }

        parameters_.map_is_uniform_ = false;
        state_.initGridmap = true;

        if (onMapLoaded_) onMapLoaded_();
    }

    void HandleSaveMap(const std::string& filePathName) {
        if (datasetHandler_) {
            datasetHandler_->SaveRaster(filePathName);
        }
    }

    void HandleModelPathSelection(const std::string& filePath, const std::string& fileName) {
        if (!getRLStatus_ || !setRLStatus_) return;

        py::dict rlStatus = getRLStatus_();
        if (state_.pathKey == "model_path") {
            rlStatus[py::str(state_.pathKey)] = py::str(filePath);
        } else if (state_.pathKey == "model_name") {
            rlStatus[py::str("model_path")] = py::str(filePath);
            rlStatus[py::str(state_.pathKey)] = py::str(fileName);
        }
        setRLStatus_(rlStatus);
    }

    void HandleModelLoadSelection(const std::string& filePath, const std::string& fileName) {
        if (!getRLStatus_ || !setRLStatus_) return;

        py::dict rlStatus = getRLStatus_();
        rlStatus[py::str("model_path")] = py::str(filePath);
        rlStatus[py::str("model_name")] = py::str(fileName);
        setRLStatus_(rlStatus);
    }

    void ResetState() {
        state_.open = false;
        state_.loadMapFromDisk = false;
        state_.saveMapToDisk = false;
        state_.modelPathSelection = false;
        state_.modelLoadSelection = false;
        state_.pathKey.clear();
    }

    FireModelParameters& parameters_;
    std::shared_ptr<DatasetHandler> datasetHandler_;
    FileDialogState state_;
    std::string currentDialogKey_;

    std::vector<std::vector<int>>* rasterData_ = nullptr;

    std::function<py::dict()> getRLStatus_;
    std::function<void(py::dict)> setRLStatus_;
    std::function<void(std::vector<std::vector<int>>*, bool)> onResetGridMap_;
    std::function<void()> onMapLoaded_;
    std::function<void()> onConsoleReset_;
};

} // namespace ui

#endif // ROSHAN_FILEDIALOGWINDOW_H
