//
// Created by nex on 09.05.24.
//

#ifndef ROSHAN_FIREMODEL_IMGUI_H
#define ROSHAN_FIREMODEL_IMGUI_H

#include "imgui.h"
#include <functional>
#include <set>
#include <map>
#include "firespin/model_parameters.h"
#include "firespin/firemodel_gridmap.h"
#include "firemodel_renderer.h"
#include "src/corine/dataset_handler.h"
#include "externals/ImGuiFileDialog/ImGuiFileDialog.h"
#include "src/utils.h"
#include "externals/pybind11/include/pybind11/pybind11.h"
#include "externals/pybind11/include/pybind11/embed.h"

namespace py = pybind11;

struct LogConsole {
    ImGuiTextBuffer      Buf;
    ImVector<int>        LineOffsets;   // Index of each line start in Buf (always starts with 0)
    ImVector<ImU32>      LineColors;    // Color per line
    ImGuiTextFilter      Filter;
    bool                 AutoScroll = true;
    bool                 ScrollToBottom = false;
    int                  MaxLines = 200000; // ring-capacity

    void Clear() {
        Buf.clear();
        LineColors.clear();
        LineOffsets.clear();
        LineOffsets.push_back(0);
    }

    void TrimToLast(int max_lines) {
        if (LineOffsets.Size <= max_lines) return;
        int keep = max_lines;
        int start_idx = LineOffsets.Size - keep;
        int start_off = LineOffsets[start_idx];

        ImGuiTextBuffer new_buf;
        new_buf.append(Buf.begin() + start_off, Buf.end());

        ImVector<int> new_offsets;
        new_offsets.resize(keep);
        for (int i = 0; i < keep; ++i)
            new_offsets[i] = LineOffsets[start_idx + i] - start_off;

        ImVector<ImU32> new_colors;
        new_colors.resize(keep);
        for (int i = 0; i < keep; ++i)
            new_colors[i] = LineColors[LineColors.Size - keep + i];

        Buf = std::move(new_buf);
        LineOffsets.swap(new_offsets);
        LineColors.swap(new_colors);
    }

    // Append one line; '\n' if it's not present
    void AddLine(const char* line, ImU32 color) {
        int old_size = Buf.size();
        Buf.append(line);
        if (old_size == Buf.size() || Buf[Buf.size()-1] != '\n')
            Buf.append("\n");
        LineOffsets.push_back(Buf.size());
        LineColors.push_back(color);
        if (AutoScroll) ScrollToBottom = true;
        if (LineOffsets.Size > MaxLines)
            TrimToLast(MaxLines);
    }

    void DrawUI(const char* id, float rows = 20.0f) {
        Filter.Draw("Filter", ImGui::GetFontSize() * 16.0f);
        ImGui::SameLine();
        ImGui::Checkbox("Auto Scroll", &AutoScroll);

        ImGui::BeginChild(id,
                          ImVec2(0, ImGui::GetTextLineHeightWithSpacing() * rows),
                          true,
                          ImGuiWindowFlags_HorizontalScrollbar);

        const int line_count = LineOffsets.Size;
        if (line_count == 0) {
            ImGui::TextDisabled("[No logs available]");
            this->Clear();
            ImGui::EndChild();
            return;
        }

        const char* buf_start = Buf.begin();

        if (Filter.IsActive()) {
            // Filtering: must test every line only draw matches
            for (int line_no = 0; line_no < LineOffsets.Size - 1; ++line_no) {
                const char* line_begin = buf_start + LineOffsets[line_no];
                const char* line_end   = buf_start + LineOffsets[line_no + 1] - 1; // exclude '\n'
                if (Filter.PassFilter(line_begin, line_end)) {
                    ImGui::PushStyleColor(ImGuiCol_Text, LineColors[line_no]);
                    ImGui::TextUnformatted(line_begin, line_end);
                    ImGui::PopStyleColor();
                }
            }
        } else {
            // Use clipper to draw only visible lines
            ImGuiListClipper clipper;
            clipper.Begin(LineOffsets.Size - 1);
            while (clipper.Step()) {
                for (int line_no = clipper.DisplayStart; line_no < clipper.DisplayEnd; ++line_no) {
                    const char* line_begin = buf_start + LineOffsets[line_no];
                    const char* line_end   = buf_start + LineOffsets[line_no + 1] - 1;
                    ImGui::PushStyleColor(ImGuiCol_Text, LineColors[line_no]);
                    ImGui::TextUnformatted(line_begin, line_end);
                    ImGui::PopStyleColor();
                }
            }
        }
        const bool was_at_bottom = ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 1.0f;

        if (AutoScroll && (ScrollToBottom || was_at_bottom))
            ImGui::SetScrollHereY(1.0f);
        ScrollToBottom = false;

        ImGui::EndChild();
    }
};

inline ImU32 ColorFromLine(const std::string& s) {
    if (s.find("ERROR")   != std::string::npos) return IM_COL32(255, 80,  80, 255);
    if (s.find("WARNING") != std::string::npos) return IM_COL32(255, 180,  0, 255);
    if (s.find("DEBUG")   != std::string::npos) return IM_COL32(180, 180,255, 255);
    return IM_COL32(100, 230, 100, 255);
}

class __attribute__((visibility("default"))) ImguiHandler {
public:
    ImguiHandler(Mode mode, FireModelParameters &parameters);
    [[maybe_unused]] void Init();
    void ImGuiSimulationControls(const std::shared_ptr<GridMap>& gridmap, std::vector<std::vector<int>> &current_raster_data,
                                 const std::shared_ptr<FireModelRenderer>& model_renderer, bool &update_simulation,
                                 bool &render_simulation, int &delay, float framerate, double running_time);
    void Config(const std::shared_ptr<FireModelRenderer>& model_renderer, std::vector<std::vector<int>> &current_raster_data, const std::shared_ptr<Wind>& wind);
    void FileHandling(const std::shared_ptr<DatasetHandler>& dataset_handler, std::vector<std::vector<int>> &current_raster_data);
    void PyConfig(std::string &user_input, std::string &model_output,
                  const std::shared_ptr<GridMap>& gridmap,
                  const std::shared_ptr<std::vector<std::shared_ptr<FlyAgent>>>& drones,
                  const std::shared_ptr<FireModelRenderer>& model_renderer);
    void ImGuiModelMenu(std::vector<std::vector<int>> &current_raster_data);
    void ShowPopups(const std::shared_ptr<GridMap>& gridmap, std::vector<std::vector<int>> &current_raster_data);
    bool ImGuiOnStartup(const std::shared_ptr<FireModelRenderer>& model_renderer, std::vector<std::vector<int>> &current_raster_data);
    void ShowParameterConfig(const std::shared_ptr<Wind>& wind);
    void HandleEvents(SDL_Event event, ImGuiIO *io, const std::shared_ptr<GridMap>& gridmap, const std::shared_ptr<FireModelRenderer>& model_renderer,
                      const std::shared_ptr<DatasetHandler>& dataset_handler, std::vector<std::vector<int>> &current_raster_data);
    static void OpenBrowser(const std::string& url);
    void updateOnRLStatusChange();
    void DefaultModeSelected() {
        model_startup_ = true;
        show_controls_ = true;
        show_model_parameter_config_ = false;
    }

    // Callbacks
    std::function<void()> onResetDrones;
    std::function<void()> onSetUniformRasterData;
    std::function<void(std::vector<std::vector<int>>*, bool)> onResetGridMap;
    std::function<void()> onFillRasterWithEnum;
    std::function<void(int,double,double,int)> onMoveDrone;
    std::function<void(CellState, int, int)> onSetNoise;
    std::function<void()> startFires;
    std::function<py::dict()> onGetRLStatus;
    std::function<void(py::dict)> onSetRLStatus;

private:
    FireModelParameters &parameters_;

    //Flags
    bool show_demo_window_ = false;
    bool show_controls_ = false;
    bool show_rl_status_ = true;
    bool show_model_parameter_config_ = false;
    bool show_noise_config_ = false;
    bool model_startup_ = false;
    bool open_file_dialog_ = false;
    bool load_map_from_disk_ = false;
    bool save_map_to_disk_ = false;
    bool init_gridmap_ = false;
    bool browser_selection_flag_ = false;  // If set to true, will load a new GridMap from a file.
    bool model_path_selection_ = false;
    bool model_mode_selection_ = false;
    bool model_load_selection_ = false;
    bool train_mode_selected_ = false;
    bool reset_console_ = false;
    std::string path_key_;
    Mode mode_;


    //For the Popup of Cells
    std::set<std::pair<int, int>> popups_;
    std::map<std::pair<int, int>, bool> popup_has_been_opened_;

    LogReader log_reader_;

    //Helper
    template<typename T>
    void DrawGrid(const std::vector<std::vector<T>> &grid, const std::string color_status);
    static void DrawBuffer(std::vector<float> buffer, int buffer_pos);
    void CheckForModelPathSelection(const std::shared_ptr<FireModelRenderer>& model_renderer);
    void RLStatusParser(const py::dict& rl_status);
};

struct LogEntry {
    std::string text;
    ImU32 color;
};

#endif //ROSHAN_FIREMODEL_IMGUI_H
