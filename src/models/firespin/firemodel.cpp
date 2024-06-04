//
// Created by nex on 08.06.23.
//

#include "firemodel.h"

std::shared_ptr<FireModel> FireModel::instance_ = nullptr;

FireModel::FireModel(std::shared_ptr<SDL_Renderer> renderer, int mode) {
    // Find the project root directory
    auto start_path = std::filesystem::current_path();
    auto project_root = find_project_root(start_path);
    std::filesystem::path  dataset_path;
    if (project_root) {
        dataset_path = *project_root / "assets" / "dataset" / "CLMS_CLCplus_RASTER_2018_010m_eu_03035_V1_1.tif";
        std::cout << "Project root found: " << project_root->string() << std::endl;
        std::cout << "Dataset path: " << dataset_path << std::endl;
    } else {
        std::cerr << "Project root not found starting from: " << start_path << std::endl;
    }
    dataset_handler_ = std::make_shared<DatasetHandler>(dataset_path);

    if (mode == 1) {
        parameters_.SetNumberOfDrones(0);
        python_code_ = false;
    }
    running_time_ = 0;

    model_renderer_ = FireModelRenderer::GetInstance(renderer, parameters_);
    wind_ = std::make_shared<Wind>(parameters_);
    gridmap_ = nullptr;
    rl_handler_ = ReinforcementLearningHandler::GetInstance(parameters_);

    user_input_ = "";
    model_output_ = "Hey, let's talk.";
    this->setupImGui();
}

void FireModel::ResetGridMap(std::vector<std::vector<int>>* rasterData) {
    gridmap_ = std::make_shared<GridMap>(wind_, parameters_, rasterData);
    model_renderer_->SetGridMap(gridmap_);
    rl_handler_->SetGridMap(gridmap_);
    rl_handler_->SetModelRenderer(model_renderer_);

    // Init drones
    rl_handler_->ResetDrones();
    // Starting Conditions for Fires
    if (python_code_){
        rl_handler_->InitFires();
    }

    //Reset simulation time
    running_time_ = 0;
}

void FireModel::SetUniformRasterData() {
    current_raster_data_.clear();
    current_raster_data_ = std::vector<std::vector<int>>(parameters_.GetGridNx(), std::vector<int>(parameters_.GetGridNy(), GENERIC_UNBURNED));
    parameters_.map_is_uniform_ = true;
}

void FireModel::FillRasterWithEnum() {
    current_raster_data_.clear();
    current_raster_data_ = std::vector<std::vector<int>>(parameters_.GetGridNx(), std::vector<int>(parameters_.GetGridNy(), GENERIC_UNBURNED));
    int total_cells = parameters_.GetGridNx() * parameters_.GetGridNy();
    // int num_classes = CELL_STATE_COUNT - 1;
    int num_classes = 11;

    int per_class = total_cells / num_classes;
    int extra = total_cells % num_classes;

    int current_class = 1;
    int count = 0;

    for(int i = 0; i < parameters_.GetGridNx(); ++i) {
        for(int j = 0; j < parameters_.GetGridNy(); ++j) {
            current_raster_data_[i][j] = current_class;
            if (++count == per_class + (current_class < extra)) {
                count = 0;
                ++current_class;
            }
        }
    }
}

void FireModel::Update() {
    // Simulation time step update
    running_time_ += parameters_.GetDt();
    // Update the fire particles and the cell states
    gridmap_->UpdateParticles();
    gridmap_->UpdateCells();
}

std::vector<std::deque<std::shared_ptr<State>>> FireModel::GetObservations() {
    return rl_handler_->GetObservations();
}

std::tuple<std::vector<std::deque<std::shared_ptr<State>>>, std::vector<double>, std::vector<bool>, std::pair<bool, bool>> FireModel::Step(std::vector<std::shared_ptr<Action>> actions){
    std::tuple<std::vector<std::deque<std::shared_ptr<State>>>, std::vector<double>, std::vector<bool>, std::pair<bool, bool>>  result;
    result = rl_handler_->Step(actions);
    std::vector<bool> terminals = std::get<2>(result);
    bool resetGridMap = true;
    // Check if all elements in terminals are true
    for (bool terminal : terminals) {
        if (!terminal) {
            resetGridMap = false;
            break;
        }
    }
    if (resetGridMap) {
        ResetGridMap(&current_raster_data_);
    }
    return rl_handler_->Step(actions);
}

bool FireModel::AgentIsRunning() {
    return agent_is_running_;
}

void FireModel::Render() {
    model_renderer_->Render(rl_handler_->GetDrones());
    model_renderer_->DrawArrow(wind_->GetCurrentAngle() * 180 / M_PI + 45);
}

void FireModel::Initialize() {
    // Outdated code
}

void FireModel::SetWidthHeight(int width, int height) {
    // Outdated code
}

void FireModel::Reset() {
    // Outdated code
}

//** ########################################################################
//                                 ImGui Stuff
//   ######################################################################
// **//

FireModel::~FireModel() {

}

void FireModel::setupImGui() {
    imgui_handler_ = std::make_shared<ImguiHandler>(python_code_, parameters_);
    imgui_handler_->onResetDrones = [this]() {rl_handler_->ResetDrones();};
    imgui_handler_->onResetGridMap = [this](std::vector<std::vector<int>>* rasterData) {ResetGridMap(rasterData);};
    imgui_handler_->onFillRasterWithEnum = [this]() {FillRasterWithEnum();};
    imgui_handler_->onSetUniformRasterData = [this]() {SetUniformRasterData();};
    imgui_handler_->onMoveDrone = [this](int drone_idx, double speed_x, double speed_y, int water_dispense) {return rl_handler_->StepDrone(
            drone_idx, speed_x, speed_y, water_dispense);};
}

void FireModel::ImGuiRendering(std::function<void(bool &, bool &, int &)> controls, bool &update_simulation,
                             bool &render_simulation, int &delay) {
    imgui_handler_->ShowControls(controls, update_simulation, render_simulation, delay);
    imgui_handler_->ImGuiModelMenu(model_renderer_, current_raster_data_);
    imgui_handler_->Config(gridmap_, model_renderer_,
                           current_raster_data_, running_time_, wind_);
    imgui_handler_->PyConfig(rl_handler_->GetRewards().getBuffer(), rl_handler_->GetRewards().getHead(), rl_handler_->GetAllRewards(), agent_is_running_, user_input_, model_output_, rl_handler_->GetDrones(), model_renderer_);
    imgui_handler_->FileHandling(dataset_handler_, current_raster_data_);
    imgui_handler_->ShowPopups(gridmap_, current_raster_data_);
}

void FireModel::HandleEvents(SDL_Event event, ImGuiIO *io) {
    imgui_handler_->HandleEvents(event, io, gridmap_, model_renderer_, dataset_handler_, current_raster_data_, agent_is_running_);
}

void FireModel::ImGuiSimulationSpeed() {
    imgui_handler_->ImGuiSimulationSpeed();
}

std::string FireModel::GetUserInput() {
    std::string tmp_input = user_input_;
    user_input_ = "";
    return tmp_input;
}

void FireModel::GetData(std::string data) {
    model_output_ = data;
}

