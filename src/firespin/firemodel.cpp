//
// Created by nex on 08.06.23.
//

#include "firemodel.h"

#include <utility>

FireModel::FireModel(Mode mode, const std::string& config_path) : mode_(mode)
{
    running_time_ = 0;
    timer_.Start();

    parameters_.init(config_path);
    parameters_.mode_ = mode_;
    dataset_handler_ = std::make_shared<DatasetHandler>(parameters_.corine_dataset_name_);
    parameters_.SetCorineLoaded(dataset_handler_->HasCorineLoaded());
    model_renderer_ = nullptr;
    wind_ = std::make_shared<Wind>(parameters_);
    gridmap_ = nullptr;

    user_input_ = "";
    model_output_ = "Hey, let's talk.";

    this->setupRLHandler();
    last_rl_mode_ = parameters_.init_rl_mode_;
    auto rl_status = rl_handler_->GetRLStatus();
    rl_status[py::str("rl_mode")] = py::str(last_rl_mode_);

    if(mode_ == Mode::GUI || mode_ == Mode::GUI_RL){
        this->setupImGui();
    }
    if (mode_ == Mode::GUI || mode_ == Mode::NoGUI){
        parameters_.check_for_model_folder_empty_ = true;
        parameters_.SetNumberOfDrones(0);
    }
    if (mode_ == Mode::NoGUI_RL) {
        parameters_.check_for_model_folder_empty_ = true;
        rl_handler_->onUpdateRLStatus = []() {}; // Disable the callback to avoid ImGui issues
        std::cout << "Running in NoGUI_RL mode. Agent always runs." << std::endl;
    }
    if (parameters_.skip_gui_init_ && (mode_ != Mode::NoGUI_RL && mode_ != Mode::NoGUI)) {
        imgui_handler_->DefaultModeSelected();
    }
    rl_handler_->SetRLStatus(rl_status);
    std::cout << "Created FireModel" << std::endl;
}

FireModel::~FireModel(){
    gridmap_.reset();
    rl_handler_.reset();
    dataset_handler_.reset();
    model_renderer_.reset();
    wind_.reset();
    std::cout << "FireModel destroyed" << std::endl;
}

void FireModel::InitializeMap() {
    auto maps_folder = get_project_path("maps_directory", {});
    auto map_path = (maps_folder / parameters_.default_map_).string();
    if (mode_ == Mode::NoGUI_RL || mode_ == Mode::NoGUI || parameters_.skip_gui_init_){
        if (!map_path.empty()){
            std::cout << "Loading map from: " << map_path << std::endl;
            this->LoadMap(map_path);
        } else {
            std::cout << "No map path provided, using default map." << std::endl;
            this->SetUniformRasterData();
        }
        parameters_.initial_mode_selection_done_ = true;
    }
}

void FireModel::CheckReset() {
    // Reset the Simulation IF the RL mode changed or the last step(RL) was terminal
    if (reset_next_step_ || (last_rl_mode_ != rl_handler_->GetRLMode())) {
        ResetGridMap(&current_raster_data_);
        reset_next_step_ = false;
        last_rl_mode_ = rl_handler_->GetRLMode();
    }
}

void FireModel::ResetGridMap(std::vector<std::vector<int>>* rasterData, bool full_reset) {
    if (!gridmap_ || full_reset) {
        gridmap_ = std::make_shared<GridMap>(wind_, parameters_, rasterData);
    } else {
        gridmap_->Reset(rasterData);
    }
    wind_->SetRandomAngle();
    parameters_.wind_angle_ = wind_->GetCurrentAngle();
    if (mode_ == Mode::GUI || mode_ == Mode::GUI_RL) {
        model_renderer_->SetGridMap(gridmap_);
        gridmap_->GenerateNoiseMap();
//        gridmap_->SetNoiseGenerated(parameters_.has_noise_);
    }

    if (mode_ == Mode::GUI_RL) {
        rl_handler_->SetModelRenderer(model_renderer_);
    }

    if (mode_ == Mode::GUI_RL || mode_ == Mode::NoGUI_RL) {
        gridmap_->SetGroundstation();
        rl_handler_->SetGridMap(gridmap_);
        fire_generator_ = std::make_shared<FireGenerator>(gridmap_, parameters_);
#ifndef SPEEDTEST
        // Starting Conditions for Fires
        fire_generator_->StartFires();
#endif
        // Init drones
        rl_handler_->ResetEnvironment(mode_);
    }

    //Reset simulation time
    running_time_ = 0;
}

bool FireModel::AgentIsRunning() {
    return rl_handler_->AgentIsRunning();
}

void FireModel::SetRenderer(SDL_Renderer* renderer) {
    model_renderer_ = FireModelRenderer::Create(renderer, parameters_);
}

void FireModel::SetUniformRasterData() {
    current_raster_data_.clear();
    parameters_.SetGridNxNyStd();
    current_raster_data_ = std::vector<std::vector<int>>(parameters_.GetGridNx(), std::vector<int>(parameters_.GetGridNy(), GENERIC_UNBURNED));
    parameters_.map_is_uniform_ = true;
    ResetGridMap(&current_raster_data_, true);
}

void FireModel::FillRasterWithEnum() {
    current_raster_data_.clear();
    parameters_.SetGridNxNyStd();
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
    gridmap_->UpdateCellDiminishing();
#ifdef SPEEDTEST
    TestBurndownHeadless();
#endif
    //this->Test();
}

std::unordered_map<std::string, std::vector<std::deque<std::shared_ptr<State>>>> FireModel::GetObservations() {
    return rl_handler_->GetObservations();
}

StepResult FireModel::Step(const std::string& agent_type, std::vector<std::shared_ptr<Action>> actions){
#ifdef SPEEDTEST
    // Construct a new action for each drone with 0, 0
    std::vector<std::shared_ptr<Action>> actions2;
    for (int i = 0; i < parameters_.GetNumberOfDrones(); ++i) {
        actions2.push_back(std::make_shared<DroneAction>(0, 0, 0));
    }
    actions = actions2;
#endif
    auto result = rl_handler_->Step(agent_type, std::move(actions));
#ifndef SPEEDTEST
    // Check if any element in terminals is true, if so some agent reached a terminal state
    if (result.summary.env_reset) {
        reset_next_step_ = true;
        // Check if died or reached goal
        if (mode_ == Mode::GUI_RL) {
            if (result.summary.any_failed){
                model_renderer_->ShowRedFlash();
            } else {
                model_renderer_->ShowGreenFlash();
            }
        }
    }
#endif
    return result;
}

void FireModel::Render() {
    model_renderer_->Render(rl_handler_->GetDrones());
    model_renderer_->DrawArrow(-wind_->GetCurrentAngle() * 180 / M_PI + 130);
}

//** ########################################################################
//                                 ImGui Stuff
//   ######################################################################
// **//

void FireModel::LoadMap(const std::string& path) {
    dataset_handler_->LoadMap(path);
    std::vector<std::vector<int>> rasterData;
    dataset_handler_->LoadMapDataset(rasterData);
    current_raster_data_.clear();
    current_raster_data_ = rasterData;
    ResetGridMap(&current_raster_data_, true);
}

void FireModel::setupRLHandler() {
    rl_handler_ = ReinforcementLearningHandler::Create(parameters_);
    std::cout << "Created ReinforcementLearning Handler" << std::endl;
}

void FireModel::setupImGui() {
    imgui_handler_ = std::make_shared<ImguiHandler>(mode_, parameters_);
    imgui_handler_->onGetRLStatus = [this]() {return rl_handler_->GetRLStatus();};
    imgui_handler_->onResetDrones = [this]() { rl_handler_->ResetEnvironment(mode_);};
    imgui_handler_->onResetGridMap = [this](std::vector<std::vector<int>>* rasterData, bool full_reset=false) {ResetGridMap(rasterData, full_reset);};
    imgui_handler_->onFillRasterWithEnum = [this]() {FillRasterWithEnum();};
    imgui_handler_->onSetUniformRasterData = [this]() {SetUniformRasterData();};
    imgui_handler_->onMoveDrone = [this](int drone_idx, double speed_x, double speed_y, int water_dispense) {return rl_handler_->StepDroneManual(
            drone_idx, speed_x, speed_y, water_dispense);};
    imgui_handler_->startFires = [this]() { fire_generator_->StartFires(); };
    imgui_handler_->onSetNoise = [this](CellState state, int noise_level, int noise_size) {GridMap::SetCellNoise(state, noise_level, noise_size);};
    imgui_handler_->onSetRLStatus = [this](py::dict status) {rl_handler_->SetRLStatus(std::move(status));};
    rl_handler_->onUpdateRLStatus = [this]() {imgui_handler_->updateOnRLStatusChange();};
}

void FireModel::ImGuiRendering(bool &update_simulation, bool &render_simulation, int &delay, float framerate) {
    imgui_handler_->ImGuiSimulationControls(gridmap_, current_raster_data_, model_renderer_,
                                            update_simulation, render_simulation, delay, framerate,
                                            running_time_);
    imgui_handler_->ImGuiModelMenu(current_raster_data_);
    imgui_handler_->Config(model_renderer_, current_raster_data_, wind_);
    imgui_handler_->PyConfig(user_input_,model_output_, gridmap_, rl_handler_->GetDrones(), model_renderer_);
    imgui_handler_->FileHandling(dataset_handler_, current_raster_data_);
    imgui_handler_->ShowPopups(gridmap_, current_raster_data_);
}

void FireModel::HandleEvents(SDL_Event event, ImGuiIO *io) {
    imgui_handler_->HandleEvents(event, io, gridmap_, model_renderer_, dataset_handler_, current_raster_data_);
}

std::string FireModel::GetUserInput() {
    std::string tmp_input = user_input_;
    user_input_ = "";
    return tmp_input;
}

void FireModel::GetData(std::string data) {
    model_output_ = data;
}

void FireModel::UpdateReward() {
    rl_handler_->UpdateReward();
}

void FireModel::SetRLStatus(pybind11::dict status) {
    rl_handler_->SetRLStatus(status);
}

pybind11::dict FireModel::GetRLStatus() {
    return rl_handler_->GetRLStatus();
}

void FireModel::TestBurndownHeadless() {

    if (!gridmap_->IsBurning()){
        std::cout << "All fires are extinguished or burned down. ";
        // End the simulation and close the program
        timer_.Stop();
        double duration = timer_.GetDurationInMilliseconds();
        std::cout << "Simulation duration(ms): " << duration << std::endl;
        timer_.AppendDuration(duration);
        // Reset the simulation
        ResetGridMap(&current_raster_data_);
        fire_generator_->StartFires();
        timer_.Start();
    }
    if (timer_.GetTimeSteps() == 0){
        std::vector<double> durations = timer_.GetDurations();
        std::cout << "Simulation ended." << std::endl;
        std::cout << "Simulation duration(ms): ";
        for (auto duration : durations)
            std::cout<< duration << ", ";
        std::cout << std::endl;
        std::cout << "Average: " << timer_.GetAverageDuration() << std::endl;
        exit(0);
    }
}

bool FireModel::InitialModeSelectionDone() {
    return parameters_.initial_mode_selection_done_ && parameters_.check_for_model_folder_empty_;
}

bool FireModel::GetEarlyClosing() {
    return !parameters_.exit_carefully_;
}
