//
// StartupWizard.h - Startup wizard window
//
// Guides user through mode selection, training setup, and map selection.
// Extracted from ImGuiOnStartup and CheckForModelPathSelection in firemodel_imgui.cpp.
//

#ifndef ROSHAN_STARTUPWIZARD_H
#define ROSHAN_STARTUPWIZARD_H

#include "IWindow.h"
#include "../UITypes.h"
#include "../UIState.h"
#include "firespin/model_parameters.h"
#include "firespin/rendering/firemodel_renderer.h"
#include "externals/pybind11/include/pybind11/pybind11.h"
#include <memory>
#include <functional>
#include <filesystem>

namespace py = pybind11;

namespace ui {

class __attribute__((visibility("hidden"))) StartupWizard : public IWindow {
public:
    StartupWizard(FireModelParameters& parameters,
                  std::shared_ptr<FireModelRenderer> renderer,
                  Mode mode)
        : parameters_(parameters)
        , renderer_(std::move(renderer))
        , mode_(mode) {}

    void Render() override {
        if (state_.phase == StartupPhase::Complete) return;

        // Handle skip_gui_init case
        if (parameters_.skip_gui_init_) {
            RenderModelFolderCheck();
            return;
        }

        switch (state_.phase) {
            case StartupPhase::ModeSelection:
                RenderModeSelection();
                break;
            case StartupPhase::ModelPathCheck:
                RenderModelFolderCheck();
                break;
            case StartupPhase::TrainingSetup:
                RenderTrainingSetup();
                break;
            case StartupPhase::MapSelection:
                RenderMapSelection();
                break;
            case StartupPhase::Complete:
                break;
        }
    }

    bool IsVisible() const override { return state_.phase != StartupPhase::Complete; }
    void SetVisible(bool /*visible*/) override { /* Visibility is controlled by phase */ }
    const char* GetName() const override { return "Startup Wizard"; }

    // Check if startup is complete
    bool IsComplete() const { return state_.phase == StartupPhase::Complete; }

    // Mark startup as complete externally
    void Complete() {
        state_.phase = StartupPhase::Complete;
        if (onComplete_) onComplete_();
    }

    // Callbacks
    void SetRLStatusCallbacks(std::function<py::dict()> get, std::function<void(py::dict)> set) {
        getRLStatus_ = std::move(get);
        setRLStatus_ = std::move(set);
    }

    void SetGridMapCallbacks(std::function<void()> setUniform,
                             std::function<void(std::vector<std::vector<int>>*, bool)> reset) {
        onSetUniformRasterData_ = std::move(setUniform);
        onResetGridMap_ = std::move(reset);
    }

    void SetFileDialogCallback(std::function<void(bool, bool)> openFileDialog) {
        openFileDialog_ = std::move(openFileDialog);
    }

    void SetOnComplete(std::function<void()> cb) { onComplete_ = std::move(cb); }

    void SetRenderer(std::shared_ptr<FireModelRenderer> renderer) { renderer_ = std::move(renderer); }
    void SetRasterData(std::vector<std::vector<int>>* data) { rasterData_ = data; }

    // State access
    StartupState& GetState() { return state_; }
    bool NeedsFileDialog() const { return needsFileDialog_; }
    void ClearFileDialogFlag() { needsFileDialog_ = false; }
    bool IsModelLoadSelection() const { return modelLoadSelection_; }

private:
    void RenderModeSelection() {
        if (mode_ != Mode::GUI_RL) {
            state_.phase = StartupPhase::MapSelection;
            return;
        }

        int width, height;
        SDL_GetRendererOutputSize(renderer_->GetRenderer(), &width, &height);

        ImGui::Begin("Select Mode", nullptr, window_flags::kModal);
        CenterWindow(width, height);

        ImGui::Spacing();
        ImGui::PushStyleColor(ImGuiCol_Button, colors::kWizardButton);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, colors::kWizardButtonHovered);
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, colors::kWizardButtonPressed);

        py::dict rlStatus = getRLStatus_();

        if (ImGui::Button("Train Model", ImVec2(-1, 0))) {
            rlStatus[py::str("rl_mode")] = py::str("train");
            state_.trainModeSelected = true;
            state_.phase = StartupPhase::ModelPathCheck;
        }

        if (ImGui::Button("Load Model", ImVec2(-1, 0))) {
            rlStatus[py::str("rl_mode")] = py::str("eval");
            modelLoadSelection_ = true;
            needsFileDialog_ = true;
            parameters_.check_for_model_folder_empty_ = true;
            state_.phase = StartupPhase::Complete;
        }

        ImGui::Spacing();
        auto resume = rlStatus["resume"].cast<bool>();
        ImGui::Checkbox("Resume Training from Checkpoint?", &resume);
        ImGui::Spacing();

        rlStatus[py::str("resume")] = resume;
        setRLStatus_(rlStatus);

        ImGui::PopStyleColor(3);
        ImGui::End();
    }

    void RenderModelFolderCheck() {
        if (parameters_.check_for_model_folder_empty_) {
            if (state_.trainModeSelected) {
                state_.phase = StartupPhase::TrainingSetup;
            }
            return;
        }

        int width, height;
        SDL_GetRendererOutputSize(renderer_->GetRenderer(), &width, &height);

        py::dict rlStatus = getRLStatus_();
        auto modelPath = rlStatus["model_path"].cast<std::string>();
        auto resume = rlStatus["resume"].cast<bool>();
        auto rlMode = rlStatus["rl_mode"].cast<std::string>();
        bool eval = rlMode == "eval";

        std::filesystem::path modelDir(modelPath);
        if (std::filesystem::exists(modelDir) && std::filesystem::is_directory(modelDir) && !(resume || eval)) {
            if (!std::filesystem::is_empty(modelDir)) {
                ImGui::Begin("Model Folder Check", nullptr, window_flags::kModal);
                CenterWindow(width, height);

                ImGui::Spacing();
                ImGui::Text("Model Folder not empty.");
                ImGui::Text("Delete all files in the folder:");
                ImGui::Text("%s", modelPath.c_str());

                if (ImGui::Button("Delete Files and Continue", ImVec2(-1, 0))) {
                    parameters_.check_for_model_folder_empty_ = true;
                    state_.resetConsole = true;
                }

                if (ImGui::Button("Close Program", ImVec2(-1, 0))) {
                    parameters_.check_for_model_folder_empty_ = true;
                    parameters_.initial_mode_selection_done_ = true;
                    parameters_.exit_carefully_ = true;
                }

                ImGui::End();
                return;
            }
        }

        parameters_.check_for_model_folder_empty_ = true;
        if (state_.trainModeSelected) {
            state_.phase = StartupPhase::TrainingSetup;
        }
    }

    void RenderTrainingSetup() {
        int width, height;
        SDL_GetRendererOutputSize(renderer_->GetRenderer(), &width, &height);

        ImGui::PushStyleColor(ImGuiCol_Button, colors::kWizardButton);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, colors::kWizardButtonHovered);
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, colors::kWizardButtonPressed);

        ImGui::Begin("Training Setup", nullptr, window_flags::kModal);
        CenterWindow(width, height);

        ImGui::Spacing();
        ImGui::Text("Algorithm specific parameters must be changed in the Config files.");
        ImGui::Spacing();

        py::dict rlStatus = getRLStatus_();

        // Hierarchy type selection
        auto hierarchyType = rlStatus["hierarchy_type"].cast<std::string>();
        const char* hierarchyTypes[] = {"fly_agent", "explore_agent", "planner_agent"};
        int currentHierarchyType = hierarchyType == "fly_agent" ? 0 : hierarchyType == "explore_agent" ? 1 : 2;

        ImGui::Text("Select Agent Type");
        if (ImGui::BeginCombo("##Hierarchy Type", hierarchyTypes[currentHierarchyType])) {
            for (int n = 0; n < IM_ARRAYSIZE(hierarchyTypes); n++) {
                bool isSelected = (currentHierarchyType == n);
                if (ImGui::Selectable(hierarchyTypes[n], isSelected))
                    currentHierarchyType = n;
                if (isSelected)
                    ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
        ImGui::Spacing();

        // Number of agents
        auto numAgents = rlStatus["num_agents"].cast<int>();
        ImGui::InputInt("Number of Agents", &numAgents, 1, 10, ImGuiInputTextFlags_CharsDecimal);
        if (ImGui::IsItemHovered()) {
            if (currentHierarchyType == 0) {
                ImGui::SetTooltip("The Number of Agents is the number of drones present in the environment.");
            } else if (currentHierarchyType == 1) {
                ImGui::SetTooltip("The Number of Agents is the number of Explorers the explore_agents deploys.");
            }
        }

        // Update parameters based on selection
        if (currentHierarchyType == 0) {
            parameters_.SetNumberOfDrones(numAgents);
        } else if (currentHierarchyType == 1) {
            parameters_.SetNumberOfExplorers(numAgents);
        } else {
            parameters_.SetNumberOfExtinguishers(numAgents);
        }
        rlStatus[py::str("num_agents")] = numAgents;
        ImGui::Spacing();

        rlStatus[py::str("hierarchy_type")] = hierarchyTypes[currentHierarchyType];

        // RL Algorithm selection
        auto rlAlgorithm = rlStatus["rl_algorithm"].cast<std::string>();
        const char* rlAlgorithms[] = {"PPO", "TD3", "IQL"};
        int currentRLAlgorithm = rlAlgorithm == "PPO" ? 0 : rlAlgorithm == "TD3" ? 1 : 2;

        ImGui::Text("Select RL Algorithm");
        if (ImGui::BeginCombo("##RL Algorithm", rlAlgorithms[currentRLAlgorithm])) {
            for (int n = 0; n < IM_ARRAYSIZE(rlAlgorithms); n++) {
                bool isSelected = (currentRLAlgorithm == n);
                if (ImGui::Selectable(rlAlgorithms[n], isSelected))
                    currentRLAlgorithm = n;
                if (isSelected)
                    ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
        rlStatus[py::str("rl_algorithm")] = rlAlgorithms[currentRLAlgorithm];
        ImGui::Spacing();

        if (ImGui::Button("Proceed to Map Selection", ImVec2(-1, 0))) {
            state_.phase = StartupPhase::MapSelection;
        }

        setRLStatus_(rlStatus);

        ImGui::PopStyleColor(3);
        ImGui::End();
    }

    void RenderMapSelection() {
        int width, height;
        SDL_GetRendererOutputSize(renderer_->GetRenderer(), &width, &height);

        ImGui::Begin("Map Selection", nullptr, window_flags::kModal);
        CenterWindow(width, height);

        ImGui::Spacing();
        ImGui::PushStyleColor(ImGuiCol_Button, colors::kWizardButton);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, colors::kWizardButtonHovered);
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, colors::kWizardButtonPressed);

        if (ImGui::Button("Uniform Vegetation")) {
            if (onSetUniformRasterData_) onSetUniformRasterData_();
            if (onResetGridMap_ && rasterData_) onResetGridMap_(rasterData_, true);
            parameters_.initial_mode_selection_done_ = true;
            Complete();
        }

        ImGui::Separator();

        if (ImGui::Button("Load from File", ImVec2(-1, 0))) {
            needsFileDialog_ = true;
            loadMapFromDisk_ = true;
            parameters_.initial_mode_selection_done_ = true;
        }

        ImGui::PopStyleColor(3);
        ImGui::End();
    }

    void CenterWindow(int screenWidth, int screenHeight) {
        ImVec2 windowSize = ImGui::GetWindowSize();
        ImGui::SetWindowPos(ImVec2((static_cast<float>(screenWidth) - windowSize.x) * 0.5f,
                                   (static_cast<float>(screenHeight) - windowSize.y) * 0.5f));
    }

    FireModelParameters& parameters_;
    std::shared_ptr<FireModelRenderer> renderer_;
    Mode mode_;
    StartupState state_;

    std::vector<std::vector<int>>* rasterData_ = nullptr;
    bool needsFileDialog_ = false;
    bool modelLoadSelection_ = false;
    bool loadMapFromDisk_ = false;

    std::function<py::dict()> getRLStatus_;
    std::function<void(py::dict)> setRLStatus_;
    std::function<void()> onSetUniformRasterData_;
    std::function<void(std::vector<std::vector<int>>*, bool)> onResetGridMap_;
    std::function<void(bool, bool)> openFileDialog_;
    std::function<void()> onComplete_;
};

} // namespace ui

#endif // ROSHAN_STARTUPWIZARD_H
