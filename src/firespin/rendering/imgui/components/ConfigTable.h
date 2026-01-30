//
// ConfigTable.h - Reusable 2-column ImGui table builder
//
// Eliminates the 9+ duplicated table patterns in the codebase with a fluent builder API.
//
// Usage:
//   ConfigTable("##MyTable")
//       .SliderFloat("Speed", &speed, 0.0f, 100.0f)
//       .Checkbox("Enabled", &enabled)
//       .Text("Status", status_str);
//

#ifndef ROSHAN_CONFIGTABLE_H
#define ROSHAN_CONFIGTABLE_H

#include "imgui.h"
#include "../UITypes.h"
#include <string>
#include <functional>

namespace ui {

class ConfigTable {
public:
    // Begin a new config table with optional flags
    explicit ConfigTable(const char* id,
                         ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg,
                         float labelWidth = 200.0f)
        : id_(id), flags_(flags), labelWidth_(labelWidth), isOpen_(false), autoHeaders_(true), rowIndex_(0) {
        isOpen_ = ImGui::BeginTable(id_, 2, flags_);
        if (isOpen_) {
            ImGui::TableSetupColumn("Label", ImGuiTableColumnFlags_WidthFixed, labelWidth_);
            ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);
        }
    }

    ~ConfigTable() {
        End();
    }

    // Disable copy
    ConfigTable(const ConfigTable&) = delete;
    ConfigTable& operator=(const ConfigTable&) = delete;

    // Show headers row (called automatically before first row if autoHeaders is true)
    ConfigTable& Headers() {
        if (isOpen_) {
            ImGui::TableHeadersRow();
            autoHeaders_ = false;
        }
        return *this;
    }

    // Disable automatic headers
    ConfigTable& NoHeaders() {
        autoHeaders_ = false;
        return *this;
    }

    // Slider for float value
    ConfigTable& SliderFloat(const char* label, float* value, float min, float max, const char* format = "%.3f") {
        if (!isOpen_) return *this;
        EnsureHeaders();
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Text("%s", label);
        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1);
        std::string widgetId = std::string("##") + label;
        ImGui::SliderFloat(widgetId.c_str(), value, min, max, format);
        return *this;
    }

    // Slider for double value (using ImGuiDataType_Double)
    ConfigTable& SliderDouble(const char* label, double* value, double min, double max, const char* format = "%.3f") {
        if (!isOpen_) return *this;
        EnsureHeaders();
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Text("%s", label);
        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1);
        std::string widgetId = std::string("##") + label;
        ImGui::SliderScalar(widgetId.c_str(), ImGuiDataType_Double, value, &min, &max, format);
        return *this;
    }

    // Slider for int value
    ConfigTable& SliderInt(const char* label, int* value, int min, int max) {
        if (!isOpen_) return *this;
        EnsureHeaders();
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Text("%s", label);
        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1);
        std::string widgetId = std::string("##") + label;
        ImGui::SliderInt(widgetId.c_str(), value, min, max);
        return *this;
    }

    // Checkbox
    ConfigTable& Checkbox(const char* label, bool* value, bool centered = true) {
        if (!isOpen_) return *this;
        EnsureHeaders();
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Text("%s", label);
        ImGui::TableNextColumn();
        if (centered) {
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (ImGui::GetContentRegionAvail().x / 2) - 10);
        }
        std::string widgetId = std::string("##") + label;
        ImGui::Checkbox(widgetId.c_str(), value);
        return *this;
    }

    // Button (centered)
    ConfigTable& Button(const char* label, const char* buttonLabel, std::function<void()> onClick, const ImVec2& size = ImVec2(100, 17)) {
        if (!isOpen_) return *this;
        EnsureHeaders();
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Text("%s", label);
        ImGui::TableNextColumn();
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (ImGui::GetContentRegionAvail().x / 2) - size.x / 2);
        std::string widgetId = std::string("##") + buttonLabel;
        if (ImGui::Button(widgetId.c_str(), size)) {
            if (onClick) onClick();
        }
        return *this;
    }

    // Static text display
    ConfigTable& Text(const char* label, const char* value) {
        if (!isOpen_) return *this;
        EnsureHeaders();
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Text("%s", label);
        ImGui::TableNextColumn();
        ImGui::Text("%s", value);
        return *this;
    }

    // Text with formatting
    ConfigTable& TextFormatted(const char* label, const char* format, ...) {
        if (!isOpen_) return *this;
        EnsureHeaders();
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Text("%s", label);
        ImGui::TableNextColumn();
        va_list args;
        va_start(args, format);
        ImGui::TextV(format, args);
        va_end(args);
        return *this;
    }

    // Input int
    ConfigTable& InputInt(const char* label, int* value, int step = 1, int stepFast = 10) {
        if (!isOpen_) return *this;
        EnsureHeaders();
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Text("%s", label);
        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-1);
        std::string widgetId = std::string("##") + label;
        ImGui::InputInt(widgetId.c_str(), value, step, stepFast);
        return *this;
    }

    // Add tooltip to previous item
    ConfigTable& Tooltip(const char* text) {
        if (!isOpen_) return *this;
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("%s", text);
        }
        return *this;
    }

    // Custom row with callback
    ConfigTable& Custom(const char* label, std::function<void()> renderValue) {
        if (!isOpen_) return *this;
        EnsureHeaders();
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Text("%s", label);
        ImGui::TableNextColumn();
        if (renderValue) renderValue();
        return *this;
    }

    // Info row (value spans full width, useful for percentages etc.)
    ConfigTable& InfoRow(const char* text) {
        if (!isOpen_) return *this;
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TableNextColumn();
        ImGui::Text("%s", text);
        rowIndex_++;
        return *this;
    }

    // Section header with separator line
    ConfigTable& SectionHeader(const char* title) {
        if (!isOpen_) return *this;
        EnsureHeaders();
        ImGui::TableNextRow();
        ImGui::TableNextColumn();

        // Draw a colored header spanning both columns
        ImGui::PushStyleColor(ImGuiCol_Text, colors::kHighlight);
        ImGui::Text("%s", title);
        ImGui::PopStyleColor();

        ImGui::TableNextColumn();
        ImGui::Separator();
        rowIndex_++;
        return *this;
    }

    // Collapsible section using TreeNode
    ConfigTable& BeginCollapsible(const char* title, bool defaultOpen = true) {
        if (!isOpen_) return *this;
        EnsureHeaders();
        ImGui::TableNextRow();
        ImGui::TableNextColumn();

        ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_SpanFullWidth;
        if (defaultOpen) flags |= ImGuiTreeNodeFlags_DefaultOpen;

        collapsibleOpen_ = ImGui::TreeNodeEx(title, flags);
        ImGui::TableNextColumn();
        rowIndex_++;
        return *this;
    }

    ConfigTable& EndCollapsible() {
        if (collapsibleOpen_) {
            ImGui::TreePop();
            collapsibleOpen_ = false;
        }
        return *this;
    }

    // Check if current collapsible section is open
    bool IsCollapsibleOpen() const { return collapsibleOpen_; }

    // Check if table is open
    bool IsOpen() const { return isOpen_; }

    // Explicitly end the table (also called by destructor)
    void End() {
        if (isOpen_) {
            ImGui::EndTable();
            isOpen_ = false;
        }
    }

private:
    void EnsureHeaders() {
        if (autoHeaders_ && isOpen_) {
            ImGui::TableHeadersRow();
            autoHeaders_ = false;
        }
    }

    const char* id_;
    ImGuiTableFlags flags_;
    float labelWidth_;
    bool isOpen_;
    bool autoHeaders_;
    bool collapsibleOpen_ = false;
    int rowIndex_;
};

// Simple 2-column table without borders (used in render options)
class SimpleTable {
public:
    explicit SimpleTable(const char* id, int columns = 2,
                         ImGuiTableFlags flags = ImGuiTableFlags_SizingStretchSame)
        : id_(id), isOpen_(false) {
        isOpen_ = ImGui::BeginTable(id_, columns, flags);
    }

    ~SimpleTable() {
        End();
    }

    SimpleTable& NextColumn() {
        if (isOpen_) ImGui::TableNextColumn();
        return *this;
    }

    SimpleTable& NextRow() {
        if (isOpen_) ImGui::TableNextRow();
        return *this;
    }

    bool IsOpen() const { return isOpen_; }

    void End() {
        if (isOpen_) {
            ImGui::EndTable();
            isOpen_ = false;
        }
    }

private:
    const char* id_;
    bool isOpen_;
};

} // namespace ui

#endif // ROSHAN_CONFIGTABLE_H
