//
// LogConsoleWidget.h - Log viewer component for ImGui
//
// Provides a scrolling log console with filtering, auto-scroll,
// and color-coded log levels with background highlights.
//

#ifndef ROSHAN_LOGCONSOLEWIDGET_H
#define ROSHAN_LOGCONSOLEWIDGET_H

#include "imgui.h"
#include "../UITypes.h"
#include <string>

namespace ui {

class LogConsoleWidget {
public:
    LogConsoleWidget() {
        Clear();
    }

    void Clear() {
        buf_.clear();
        lineColors_.clear();
        lineOffsets_.clear();
        lineOffsets_.push_back(0);
    }

    void TrimToLast(int maxLines) {
        if (lineOffsets_.Size <= maxLines) return;

        int keep = maxLines;
        int startIdx = lineOffsets_.Size - keep;
        int startOff = lineOffsets_[startIdx];

        ImGuiTextBuffer newBuf;
        newBuf.append(buf_.begin() + startOff, buf_.end());

        ImVector<int> newOffsets;
        newOffsets.resize(keep);
        for (int i = 0; i < keep; ++i)
            newOffsets[i] = lineOffsets_[startIdx + i] - startOff;

        ImVector<ImU32> newColors;
        newColors.resize(keep);
        for (int i = 0; i < keep; ++i)
            newColors[i] = lineColors_[lineColors_.Size - keep + i];

        buf_ = std::move(newBuf);
        lineOffsets_.swap(newOffsets);
        lineColors_.swap(newColors);
    }

    // Add a line with automatic color detection based on content
    void AddLine(const char* line) {
        AddLine(line, ColorFromLogLine(line));
    }

    // Add a line with explicit color
    void AddLine(const char* line, ImU32 color) {
        int oldSize = buf_.size();
        buf_.append(line);
        if (oldSize == buf_.size() || buf_[buf_.size() - 1] != '\n')
            buf_.append("\n");
        lineOffsets_.push_back(buf_.size());
        lineColors_.push_back(color);

        if (autoScroll_)
            scrollToBottom_ = true;

        if (lineOffsets_.Size > maxLines_)
            TrimToLast(maxLines_);
    }

    void Draw(const char* id, float rows = 20.0f) {
        // Simplified controls: just filter and auto-scroll
        filter_.Draw("Filter", ImGui::GetFontSize() * 16.0f);
        ImGui::SameLine();
        ImGui::Checkbox("Auto Scroll", &autoScroll_);

        // Log content area
        ImGui::BeginChild(id,
                          ImVec2(0, ImGui::GetTextLineHeightWithSpacing() * rows),
                          true,
                          ImGuiWindowFlags_HorizontalScrollbar);

        const int lineCount = lineOffsets_.Size;
        if (lineCount == 0) {
            ImGui::TextDisabled("[No logs available]");
            Clear();
            ImGui::EndChild();
            return;
        }

        const char* bufStart = buf_.begin();

        if (filter_.IsActive()) {
            // Filtering mode: test each line
            for (int lineNo = 0; lineNo < lineOffsets_.Size - 1; ++lineNo) {
                const char* lineBegin = bufStart + lineOffsets_[lineNo];
                const char* lineEnd = bufStart + lineOffsets_[lineNo + 1] - 1;
                if (filter_.PassFilter(lineBegin, lineEnd)) {
                    RenderLogLine(lineNo, lineBegin, lineEnd);
                }
            }
        } else {
            // Use clipper for efficient rendering of large logs
            ImGuiListClipper clipper;
            clipper.Begin(lineOffsets_.Size - 1);
            while (clipper.Step()) {
                for (int lineNo = clipper.DisplayStart; lineNo < clipper.DisplayEnd; ++lineNo) {
                    const char* lineBegin = bufStart + lineOffsets_[lineNo];
                    const char* lineEnd = bufStart + lineOffsets_[lineNo + 1] - 1;
                    RenderLogLine(lineNo, lineBegin, lineEnd);
                }
            }
        }

        // Handle auto-scroll
        const bool wasAtBottom = ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 1.0f;
        if (autoScroll_ && (scrollToBottom_ || wasAtBottom))
            ImGui::SetScrollHereY(1.0f);
        scrollToBottom_ = false;

        ImGui::EndChild();
    }

    // Accessors
    bool GetAutoScroll() const { return autoScroll_; }
    void SetAutoScroll(bool value) { autoScroll_ = value; }
    int GetMaxLines() const { return maxLines_; }
    void SetMaxLines(int value) { maxLines_ = value; }

private:
    void RenderLogLine(int lineNo, const char* lineBegin, const char* lineEnd) {
        // Draw background highlight for errors/warnings
        ImU32 color = lineColors_[lineNo];
        if (color == colors::kLogError || color == colors::kLogWarning) {
            ImVec2 pos = ImGui::GetCursorScreenPos();
            ImVec2 size = ImVec2(ImGui::GetContentRegionAvail().x, ImGui::GetTextLineHeightWithSpacing());
            ImU32 bgColor = (color == colors::kLogError) ?
                IM_COL32(255, 80, 80, 30) : IM_COL32(255, 180, 0, 25);
            ImGui::GetWindowDrawList()->AddRectFilled(pos, ImVec2(pos.x + size.x, pos.y + size.y), bgColor);
        }

        // Log message with color
        ImGui::PushStyleColor(ImGuiCol_Text, color);
        ImGui::TextUnformatted(lineBegin, lineEnd);
        ImGui::PopStyleColor();
    }

    ImGuiTextBuffer buf_;
    ImVector<int> lineOffsets_;
    ImVector<ImU32> lineColors_;
    ImGuiTextFilter filter_;
    bool autoScroll_ = true;
    bool scrollToBottom_ = false;
    int maxLines_ = 200000;
};

} // namespace ui

#endif // ROSHAN_LOGCONSOLEWIDGET_H
