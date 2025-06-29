//
// Created by nex on 08.06.23.
//

#include "firemodel.h"

#include <utility>

FireModel::FireModel(Mode mode) : mode_(mode)
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

    this->setupRLHandler();

    if(mode_ == Mode::GUI || mode_ == Mode::GUI_RL){
        this->setupImGui();
    }
    if (mode_ == Mode::GUI || mode_ == Mode::NoGUI){
        parameters_.SetNumberOfDrones(0);
    }
    if (mode_ == Mode::NoGUI_RL) {
        parameters_.SetNumberOfDrones(1);
        parameters_.SetAgentIsRunning(true);
        parameters_.initial_mode_selection_done_ = true;
    }
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

void FireModel::InitializeMap(const std::string& map_path) {
    if ((mode_ == Mode::NoGUI_RL || mode_ == Mode::NoGUI)){
        if (!map_path.empty()){
            std::cout << "Loading map from: " << map_path << std::endl;
            this->LoadMap(map_path);
        } else {
            std::cout << "No map path provided, using default map." << std::endl;
            this->SetUniformRasterData();
        }
        this->StartFires(parameters_.fire_percentage_);
    }
}

void FireModel::ResetGridMap(std::vector<std::vector<int>>* rasterData) {
    gridmap_ = std::make_shared<GridMap>(wind_, parameters_, rasterData);
    wind_->SetRandomAngle();
    parameters_.wind_angle_ = wind_->GetCurrentAngle();
    if (mode_ == Mode::GUI || mode_ == Mode::GUI_RL) {
        model_renderer_->SetGridMap(gridmap_);
        gridmap_->GenerateNoiseMap();
        gridmap_->SetNoiseGenerated(parameters_.has_noise_);
    }

    if (mode_ == Mode::GUI_RL) {
        rl_handler_->SetModelRenderer(model_renderer_);
    }

    if (mode_ == Mode::GUI_RL || mode_ == Mode::NoGUI_RL) {
        gridmap_->SetGroundstation();
        rl_handler_->SetGridMap(gridmap_);
#ifndef SPEEDTEST
        // Starting Conditions for Fires
        rl_handler_->InitFires();
#endif
        // Init drones
        rl_handler_->ResetEnvironment(mode_);
    }

    //Reset simulation time
    running_time_ = 0;
}

void FireModel::SetRenderer(SDL_Renderer* renderer) {
    model_renderer_ = FireModelRenderer::Create(renderer, parameters_);
}

void FireModel::SetUniformRasterData() {
    current_raster_data_.clear();
    parameters_.SetGridNxNyStd();
    current_raster_data_ = std::vector<std::vector<int>>(parameters_.GetGridNx(), std::vector<int>(parameters_.GetGridNy(), GENERIC_UNBURNED));
    parameters_.map_is_uniform_ = true;
    ResetGridMap(&current_raster_data_);
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

void FireModel::SimStep(std::vector<std::shared_ptr<Action>> actions){
    rl_handler_->SimStep(std::move(actions));
}

std::tuple<std::unordered_map<std::string, std::vector<std::deque<std::shared_ptr<State>>>>,
        std::vector<double>,
        std::vector<bool>,
        std::unordered_map<std::string, bool>,
        double> FireModel::Step(const std::string& agent_type, std::vector<std::shared_ptr<Action>> actions){
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
    if (std::get<3>(result)["EnvReset"]) {
        ResetGridMap(&current_raster_data_);
        // Check if died or reached goal
        if (mode_ == Mode::GUI_RL) {
            if (std::get<3>(result)["OneAgentDied"]){
                model_renderer_->ShowRedFlash();
            } else {
                model_renderer_->ShowGreenFlash();
            }
        }
    }
#endif
    return result;
}

bool FireModel::AgentIsRunning() {
    return parameters_.GetAgentIsRunning();
}

void FireModel::Render() {
    model_renderer_->Render(rl_handler_->GetDrones());
    model_renderer_->DrawArrow(-wind_->GetCurrentAngle() * 180 / M_PI + 130);
}

//** ########################################################################
//                                 ImGui Stuff
//   ######################################################################
// **//

void FireModel::LoadMap(std::string path) {
    dataset_handler_->LoadMap(std::move(path));
    std::vector<std::vector<int>> rasterData;
    dataset_handler_->LoadMapDataset(rasterData);
    current_raster_data_.clear();
    current_raster_data_ = rasterData;
    ResetGridMap(&current_raster_data_);
}

void FireModel::setupRLHandler() {
    rl_handler_ = ReinforcementLearningHandler::Create(parameters_);
    std::cout << "Created ReinforcementLearning Handler" << std::endl;
    rl_handler_->startFires = [this](float percentage) {StartFires(percentage);};
}

void FireModel::setupImGui() {
    imgui_handler_ = std::make_shared<ImguiHandler>(mode_, parameters_);
    imgui_handler_->onResetDrones = [this]() { rl_handler_->ResetEnvironment(mode_);};
    imgui_handler_->onResetGridMap = [this](std::vector<std::vector<int>>* rasterData) {ResetGridMap(rasterData);};
    imgui_handler_->onFillRasterWithEnum = [this]() {FillRasterWithEnum();};
    imgui_handler_->onSetUniformRasterData = [this]() {SetUniformRasterData();};
    imgui_handler_->onMoveDrone = [this](int drone_idx, double speed_x, double speed_y, int water_dispense) {return rl_handler_->StepDroneManual(
            drone_idx, speed_x, speed_y, water_dispense);};
    imgui_handler_->startFires = [this](float percentage) {StartFires(percentage);};
    imgui_handler_->onSetNoise = [this](CellState state, int noise_level, int noise_size) {gridmap_->SetCellNoise(state, noise_level, noise_size);};
    imgui_handler_->onGetRLStatus = [this]() {return rl_handler_->GetRLStatus();};
    imgui_handler_->onSetRLStatus = [this](py::dict status) {rl_handler_->SetRLStatus(std::move(status));};
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
    imgui_handler_->HandleEvents(event, io, gridmap_, model_renderer_, dataset_handler_, current_raster_data_, parameters_.agent_is_running_);
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
        StartFires(parameters_.fire_percentage_);
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

void FireModel::StartFires(float percentage) {
    int cells = gridmap_->GetNumCells();
    double perc = percentage * 0.01;
    int fires = static_cast<int>(cells * perc);
    if (!gridmap_->CanStartFires(fires)){
        std::cout << "Map is incapable of burning that much. Please choose a lower percentage." << std::endl;
        return;
    }
    if (!parameters_.ignite_single_cells_) {
        this->IgniteFireCluster(fires);
    } else {
        for(int i = 0; i < fires;) {
            std::pair<int, int> point = gridmap_->GetRandomPointInGrid();
            if (gridmap_->CellCanIgnite(point.first, point.second)) {
                gridmap_->IgniteCell(point.first, point.second);
                i++;
            }
        }
    }
}

void FireModel::IgniteFireCluster(int fires) {
    // Starting point
    std::pair<int, int> point = gridmap_->GetRandomPointInGrid();
    std::set<std::pair<int, int>> visited;
    std::queue<std::pair<int, int>> to_visit;

    to_visit.push(point);
    visited.insert(point);

    int ignited = 0;

    while (!to_visit.empty() && ignited < fires) {
        std::pair<int, int> current = to_visit.front();
        to_visit.pop();

        if(gridmap_->CellCanIgnite(current.first, current.second)) {
            gridmap_->IgniteCell(current.first, current.second);
            ignited++;
        }

        // Get Moore Neighborhood
        std::vector<std::pair<int, int>> neighbors = gridmap_->GetMooreNeighborhood(current.first, current.second);
        for (auto neighbor : neighbors) {
            if (visited.find(neighbor) == visited.end() && gridmap_->CellCanIgnite(neighbor.first, neighbor.second)) {
                double randomValue = static_cast<double>(std::rand()) / RAND_MAX;
                double fireProbability = parameters_.fire_spread_prob_ + (static_cast<double>(std::rand()) / RAND_MAX - 0.5) * parameters_.fire_noise_;
                if (randomValue < fireProbability) {
                    to_visit.push(neighbor);
                    visited.insert(neighbor);
                }
            }
        }
    }
}

int FireModel::GetViewRange(const std::string& agent_type) {
    return FireModelParameters::GetViewRange(agent_type);
}

int FireModel::GetTimeSteps() {
    return parameters_.GetTimeSteps();
}

bool FireModel::InitialModeSelectionDone() {
    return parameters_.InitialModeSelectionDone();
}

int FireModel::GetMapSize() {
    return parameters_.GetExplorationMapSize();
}
