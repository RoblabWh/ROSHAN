//
// Created by nex on 08.06.23.
//

#include "firemodel.h"

std::shared_ptr<FireModel> FireModel::instance_ = nullptr;

FireModel::FireModel(Mode mode, const std::string& map_path) : mode_(mode)
{

    running_time_ = 0;
    timer_.Start();

    dataset_handler_ = std::make_shared<DatasetHandler>();
    parameters_.SetCorineLoaded(dataset_handler_->HasCorineLoaded());
    model_renderer_ = nullptr;
    wind_ = std::make_shared<Wind>(parameters_);
    gridmap_ = nullptr;
    rl_handler_ = nullptr;

    user_input_ = "";
    model_output_ = "Hey, let's talk.";
    agent_is_running_ = false;

    this->setupRLHandler();

    if(mode_ == Mode::GUI || mode_ == Mode::GUI_RL){
        this->setupImGui();
    }
    if (mode_ == Mode::GUI || mode_ == Mode::NoGUI){
        parameters_.SetNumberOfDrones(0);
    }
    if ((mode_ == Mode::NoGUI_RL || mode_ == Mode::NoGUI)){
        if (!map_path.empty()){
            std::cout << "Loading map from: " << map_path << std::endl;
            this->LoadMap(map_path);
        } else {
            std::cout << "No map path provided, using default map." << std::endl;
            this->SetUniformRasterData();
        }
        this->StartFires(1);
    }
    if (mode_ == Mode::NoGUI_RL) {
        parameters_.SetNumberOfDrones(1);
        agent_is_running_ = true;
    }
}

void FireModel::ResetGridMap(std::vector<std::vector<int>>* rasterData) {
    gridmap_ = std::make_shared<GridMap>(wind_, parameters_, rasterData);

    if (mode_ == Mode::GUI || mode_ == Mode::GUI_RL) {
        model_renderer_->SetGridMap(gridmap_);
        gridmap_->GenerateNoiseMap();
        gridmap_->SetNoiseGenerated(parameters_.has_noise_);
    }

    if (mode_ == Mode::GUI_RL) {
        rl_handler_->SetModelRenderer(model_renderer_);
    }

    if (mode_ == Mode::GUI_RL || mode_ == Mode::NoGUI_RL) {
        rl_handler_->SetGridMap(gridmap_);
        // Init drones
        rl_handler_->ResetDrones(mode_);
#ifndef SPEEDTEST
        // Starting Conditions for Fires
        rl_handler_->InitFires();
#endif
    }


    //Reset simulation time
    running_time_ = 0;
}

void FireModel::SetRenderer(std::shared_ptr<SDL_Renderer> renderer) {
    model_renderer_ = FireModelRenderer::GetInstance(renderer, parameters_);
}

void FireModel::SetUniformRasterData() {
    current_raster_data_.clear();
    current_raster_data_ = std::vector<std::vector<int>>(parameters_.GetGridNx(), std::vector<int>(parameters_.GetGridNy(), GENERIC_UNBURNED));
    parameters_.map_is_uniform_ = true;
    ResetGridMap(&current_raster_data_);
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
#ifdef SPEEDTEST
    TestBurndownHeadless();
#endif
    //this->Test();
}

std::vector<std::deque<std::shared_ptr<State>>> FireModel::GetObservations() {
    return rl_handler_->GetObservations();
}

std::tuple<std::vector<std::deque<std::shared_ptr<State>>>, std::vector<double>, std::vector<bool>, std::pair<bool, bool>, double> FireModel::Step(std::vector<std::shared_ptr<Action>> actions){
#ifdef SPEEDTEST
    // Construct a new action for each drone with 0, 0
    std::vector<std::shared_ptr<Action>> actions2;
    for (int i = 0; i < parameters_.GetNumberOfDrones(); ++i) {
        actions2.push_back(std::make_shared<DroneAction>(0, 0, 0));
    }
    actions = actions2;
#endif
    std::tuple<std::vector<std::deque<std::shared_ptr<State>>>, std::vector<double>, std::vector<bool>, std::pair<bool, bool>, double>  result;
    result = rl_handler_->Step(actions);
#ifndef SPEEDTEST
    // Check if all elements in terminals are true, if so all agents reached a terminal state
    std::vector<bool> terminals = std::get<2>(result);
    bool resetEnv = std::all_of(terminals.begin(), terminals.end(),
                                [](bool terminal) {return terminal;});
    if (resetEnv)
        ResetGridMap(&current_raster_data_);
#endif
    return result;
}

bool FireModel::AgentIsRunning() {
    return agent_is_running_;
}

void FireModel::Render() {
    model_renderer_->Render(rl_handler_->GetDrones());
    model_renderer_->DrawArrow(wind_->GetCurrentAngle() * 180 / M_PI + 45);
}

//** ########################################################################
//                                 ImGui Stuff
//   ######################################################################
// **//

FireModel::~FireModel() {

}

void FireModel::LoadMap(std::string path) {
    dataset_handler_->LoadMap(std::move(path));
    std::vector<std::vector<int>> rasterData;
    dataset_handler_->LoadMapDataset(rasterData);
    current_raster_data_.clear();
    current_raster_data_ = rasterData;
    ResetGridMap(&current_raster_data_);
}

void FireModel::setupRLHandler() {
    rl_handler_ = ReinforcementLearningHandler::GetInstance(parameters_);
    rl_handler_->startFires = [this](int percentage) {StartFires(percentage);};
}

void FireModel::setupImGui() {
    imgui_handler_ = std::make_shared<ImguiHandler>(mode_, parameters_);
    imgui_handler_->onResetDrones = [this]() {rl_handler_->ResetDrones(mode_);};
    imgui_handler_->onResetGridMap = [this](std::vector<std::vector<int>>* rasterData) {ResetGridMap(rasterData);};
    imgui_handler_->onFillRasterWithEnum = [this]() {FillRasterWithEnum();};
    imgui_handler_->onSetUniformRasterData = [this]() {SetUniformRasterData();};
    imgui_handler_->onMoveDrone = [this](int drone_idx, double speed_x, double speed_y, int water_dispense) {return rl_handler_->StepDrone(
            drone_idx, speed_x, speed_y, water_dispense);};
    imgui_handler_->startFires = [this](int percentage) {StartFires(percentage);};
    imgui_handler_->onSetNoise = [this](CellState state, int noise_level, int noise_size) {gridmap_->SetCellNoise(state, noise_level, noise_size);};
    imgui_handler_->onGetRLStatus = [this]() {return rl_handler_->GetRLStatus();};
    imgui_handler_->onSetRLStatus = [this](py::dict status) {rl_handler_->SetRLStatus(status);};
}

void FireModel::ImGuiRendering(bool &update_simulation, bool &render_simulation, int &delay, float framerate) {
    imgui_handler_->ImGuiSimulationControls(gridmap_, current_raster_data_, model_renderer_,
                                            update_simulation, render_simulation, delay, framerate,
                                            running_time_);
    imgui_handler_->ImGuiModelMenu(current_raster_data_);
    imgui_handler_->Config(model_renderer_, current_raster_data_, wind_);
    imgui_handler_->PyConfig(rl_handler_->GetRewards().getBuffer(), rl_handler_->GetRewards().getHead(), rl_handler_->GetAllRewards(), agent_is_running_, user_input_, model_output_, rl_handler_->GetDrones(), model_renderer_);
    imgui_handler_->FileHandling(dataset_handler_, current_raster_data_);
    imgui_handler_->ShowPopups(gridmap_, current_raster_data_);
}

void FireModel::HandleEvents(SDL_Event event, ImGuiIO *io) {
    imgui_handler_->HandleEvents(event, io, gridmap_, model_renderer_, dataset_handler_, current_raster_data_, agent_is_running_);
}

std::string FireModel::GetUserInput() {
    std::string tmp_input = user_input_;
    user_input_ = "";
    return tmp_input;
}

void FireModel::GetData(std::string data) {
    model_output_ = data;
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
        StartFires(1);
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

void FireModel::StartFires(int percentage) {
    int cells = gridmap_->GetNumCells();
    double perc = percentage * 0.01;
    int fires = static_cast<int>(cells * perc);
    if (!gridmap_->CanStartFires(fires)){
        std::cout << "Map is incapable of burning that much. Please choose a lower percentage." << std::endl;
        return;
    }
    for(int i = 0; i < fires;) {
        std::pair<int, int> point = gridmap_->GetRandomPointInGrid();
        if (gridmap_->CellCanIgnite(point.first, point.second)) {
            gridmap_->IgniteCell(point.first, point.second);
            i++;
        }
    }
    //std::cout << "Fires started: " << fires << std::endl;
}

int FireModel::GetViewRange() {
    return parameters_.GetViewRange();
}

int FireModel::GetTimeSteps() {
    return parameters_.GetTimeSteps();
}
