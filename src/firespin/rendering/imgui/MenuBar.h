//
// MenuBar.h - Main menu bar
//
// Renders the main application menu bar.
// Extracted from ImGuiModelMenu method in firemodel_imgui.cpp.
//

#ifndef ROSHAN_MENUBAR_H
#define ROSHAN_MENUBAR_H

#include "imgui.h"
#include "ThemeManager.h"
#include "firespin/model_parameters.h"
#include <functional>
#include <string>

namespace ui {

class MenuBar {
public:
    MenuBar(FireModelParameters& parameters, Mode mode)
        : parameters_(parameters)
        , mode_(mode) {}

    void Render() {
        if (!modelStartupComplete_) return;

        if (ImGui::BeginMainMenuBar()) {
            RenderFileMenu();
            RenderParticlesMenu();
            RenderViewMenu();
            RenderHelpMenu();
            ImGui::EndMainMenuBar();
        }
    }

    void SetModelStartupComplete(bool complete) { modelStartupComplete_ = complete; }

    // Window visibility controls
    void SetShowControls(bool* show) { showControls_ = show; }
    void SetShowRLStatus(bool* show) { showRLStatus_ = show; }
    void SetShowParameterConfig(bool* show) { showParameterConfig_ = show; }
    void SetShowNoiseConfig(bool* show) { showNoiseConfig_ = show; }
    void SetShowDemoWindow(bool* show) { showDemoWindow_ = show; }
    void SetThemeCallback(std::function<void(Theme)> cb) { onThemeChange_ = std::move(cb); }

    // Callbacks
    void SetBrowserCallback(std::function<void()> cb) { onOpenBrowser_ = std::move(cb); }
    void SetBrowserSelectionCallback(std::function<void()> cb) { onBrowserSelection_ = std::move(cb); }
    void SetLoadMapCallback(std::function<void()> cb) { onLoadMap_ = std::move(cb); }
    void SetLoadUniformCallback(std::function<void()> cb) { onLoadUniform_ = std::move(cb); }
    void SetLoadClassesCallback(std::function<void()> cb) { onLoadClasses_ = std::move(cb); }
    void SetSaveMapCallback(std::function<void()> cb) { onSaveMap_ = std::move(cb); }
    void SetResetGridMapCallback(std::function<void()> cb) { onResetGridMap_ = std::move(cb); }

private:
    void RenderFileMenu() {
        if (ImGui::BeginMenu("File")) {
            if (parameters_.GetCorineLoaded()) {
                if (ImGui::MenuItem("Open Browser")) {
                    if (onOpenBrowser_) onOpenBrowser_();
                }
                if (ImGui::MenuItem("Load Map from Browser Selection")) {
                    if (onBrowserSelection_) onBrowserSelection_();
                }
            }
            if (ImGui::MenuItem("Load Map from Disk")) {
                if (onLoadMap_) onLoadMap_();
            }
            if (ImGui::MenuItem("Load Uniform Map")) {
                if (onLoadUniform_) onLoadUniform_();
            }
            if (ImGui::MenuItem("Load Map with Classes")) {
                if (onLoadClasses_) onLoadClasses_();
            }
            if (ImGui::MenuItem("Save Map")) {
                if (onSaveMap_) onSaveMap_();
            }
            if (ImGui::MenuItem("Reset GridMap")) {
                if (onResetGridMap_) onResetGridMap_();
            }
            ImGui::EndMenu();
        }
    }

    void RenderParticlesMenu() {
        if (ImGui::BeginMenu("Model Particles")) {
            ImGui::MenuItem("Emit Convective Particles", nullptr, &parameters_.emit_convective_);
            ImGui::MenuItem("Emit Radiation Particles", nullptr, &parameters_.emit_radiation_);
            ImGui::EndMenu();
        }
    }

    void RenderViewMenu() {
        if (ImGui::BeginMenu("View")) {
            // Window visibility toggles
            if (showControls_)
                ImGui::MenuItem("Simulation Controls", nullptr, showControls_);
            if (mode_ == Mode::GUI_RL && showRLStatus_)
                ImGui::MenuItem("RL Status", nullptr, showRLStatus_);
            if (showParameterConfig_)
                ImGui::MenuItem("Parameter Config", nullptr, showParameterConfig_);

            ImGui::Separator();

            // Theme submenu
            if (ImGui::BeginMenu("Theme")) {
                bool isDark = ThemeManager::GetCurrentTheme() == Theme::Dark;
                bool isLight = ThemeManager::GetCurrentTheme() == Theme::Light;

                if (ImGui::MenuItem("Dark", nullptr, isDark)) {
                    ThemeManager::ApplyTheme(Theme::Dark);
                    if (onThemeChange_) onThemeChange_(Theme::Dark);
                }
                if (ImGui::MenuItem("Light", nullptr, isLight)) {
                    ThemeManager::ApplyTheme(Theme::Light);
                    if (onThemeChange_) onThemeChange_(Theme::Light);
                }
                ImGui::EndMenu();
            }

            ImGui::EndMenu();
        }
    }

    void RenderHelpMenu() {
        if (ImGui::BeginMenu("Help")) {
            if (parameters_.has_noise_ && showNoiseConfig_)
                ImGui::MenuItem("Noise Config", nullptr, showNoiseConfig_);
            if (showDemoWindow_)
                ImGui::MenuItem("ImGui Help", nullptr, showDemoWindow_);
            ImGui::EndMenu();
        }
    }

    FireModelParameters& parameters_;
    Mode mode_;
    bool modelStartupComplete_ = false;

    bool* showControls_ = nullptr;
    bool* showRLStatus_ = nullptr;
    bool* showParameterConfig_ = nullptr;
    bool* showNoiseConfig_ = nullptr;
    bool* showDemoWindow_ = nullptr;

    std::function<void()> onOpenBrowser_;
    std::function<void()> onBrowserSelection_;
    std::function<void()> onLoadMap_;
    std::function<void()> onLoadUniform_;
    std::function<void()> onLoadClasses_;
    std::function<void()> onSaveMap_;
    std::function<void()> onResetGridMap_;
    std::function<void(Theme)> onThemeChange_;
};

} // namespace ui

#endif // ROSHAN_MENUBAR_H
