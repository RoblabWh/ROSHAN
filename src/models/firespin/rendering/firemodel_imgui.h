//
// Created by nex on 09.05.24.
//

#ifndef ROSHAN_FIREMODEL_IMGUI_H
#define ROSHAN_FIREMODEL_IMGUI_H

#include <imgui.h>
#include <functional>
#include <set>
#include <map>
#include "models/firespin/model_parameters.h"
#include "models/firespin/firemodel_gridmap.h"
#include "models/firespin/rendering/firemodel_renderer.h"
#include "src/corine/dataset_handler.h"
#include "externals/ImGuiFileDialog/ImGuiFileDialog.h"
#include "src/utils.h"
#include "externals/pybind11/include/pybind11/pybind11.h"
#include "externals/pybind11/include/pybind11/embed.h"

namespace py = pybind11;

class __attribute__((visibility("default"))) ImguiHandler {
public:
    ImguiHandler(Mode mode, FireModelParameters &parameters);
    void ImGuiSimulationControls(std::shared_ptr<GridMap> gridmap, std::vector<std::vector<int>> &current_raster_data,
                                 std::shared_ptr<FireModelRenderer> model_renderer, bool &update_simulation,
                                 bool &render_simulation, int &delay, float framerate, double running_time);
    void Config(std::shared_ptr<FireModelRenderer> model_renderer, std::vector<std::vector<int>> &current_raster_data, std::shared_ptr<Wind> wind);
    void FileHandling(std::shared_ptr<DatasetHandler> dataset_handler, std::vector<std::vector<int>> &current_raster_data);
    void PyConfig(std::vector<float> rewards, int rewards_pos,std::vector<float> all_rewards,
                  bool &agent_is_running, std::string &user_input, std::string &model_output,
                  std::shared_ptr<std::vector<std::shared_ptr<DroneAgent>>> drones,
                  const std::shared_ptr<FireModelRenderer>& model_renderer);
    void ImGuiModelMenu(std::vector<std::vector<int>> &current_raster_data);
    void ShowPopups(std::shared_ptr<GridMap> gridmap, std::vector<std::vector<int>> &current_raster_data);
    bool ImGuiOnStartup(std::shared_ptr<FireModelRenderer> model_renderer, std::vector<std::vector<int>> &current_raster_data);
    void ShowParameterConfig(std::shared_ptr<Wind> wind);
    void HandleEvents(SDL_Event event, ImGuiIO *io, std::shared_ptr<GridMap> gridmap, std::shared_ptr<FireModelRenderer> model_renderer,
                      std::shared_ptr<DatasetHandler> dataset_handler, std::vector<std::vector<int>> &current_raster_data, bool agent_is_running);
    void OpenBrowser(std::string url);

    // Callbacks
    std::function<void()> onResetDrones;
    std::function<void()> onSetUniformRasterData;
    std::function<void(std::vector<std::vector<int>>*)> onResetGridMap;
    std::function<void()> onFillRasterWithEnum;
    std::function<void(int,double,double,int)> onMoveDrone;
    std::function<void(CellState, int, int)> onSetNoise;
    std::function<void(float)> startFires;
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
    std::string path_key_;
    Mode mode_;
    bool show_input_images_ = false;


    //For the Popup of Cells
    std::set<std::pair<int, int>> popups_;
    std::map<std::pair<int, int>, bool> popup_has_been_opened_;


    //Helper
    void DrawGrid(const std::vector<std::vector<int>>& grid, std::shared_ptr<FireModelRenderer> renderer, float cell_size, bool is_fire_status = false);
    void DrawBuffer(std::vector<float> buffer, int buffer_pos);
};

#endif //ROSHAN_FIREMODEL_IMGUI_H
