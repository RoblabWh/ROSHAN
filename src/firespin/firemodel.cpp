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
    dataset_handler_ = std::make_shared<DatasetHandler>(parameters_.corine_dataset_name_);
    parameters_.SetCorineLoaded(dataset_handler_->HasCorineLoaded());
    model_renderer_ = nullptr;
    wind_ = std::make_shared<Wind>(parameters_);
    gridmap_ = nullptr;

    user_input_ = "";
    model_output_ = "Hey, let's talk.";

    this->setupRLHandler();

    if(mode_ == Mode::GUI || mode_ == Mode::GUI_RL){
        this->setupImGui();
    }
    if (mode_ == Mode::GUI || mode_ == Mode::NoGUI){
        parameters_.check_for_model_folder_empty_ = true;
        parameters_.SetNumberOfDrones(0);
    }
    if (mode_ == Mode::NoGUI_RL) {
        parameters_.SetAgentIsRunning(true);
        parameters_.check_for_model_folder_empty_ = true;
        std::cout << "Running in NoGUI_RL mode. Agent always runs." << std::endl;
    }
    if (parameters_.skip_gui_init_ && (mode_ != Mode::NoGUI_RL && mode_ != Mode::NoGUI)) {
        imgui_handler_->DefaultModeSelected();
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

void FireModel::LoadMap(const std::string& path) {
    dataset_handler_->LoadMap(path);
    std::vector<std::vector<int>> rasterData;
    dataset_handler_->LoadMapDataset(rasterData);
    current_raster_data_.clear();
    current_raster_data_ = rasterData;
    ResetGridMap(&current_raster_data_);
}

void FireModel::setupRLHandler() {
    rl_handler_ = ReinforcementLearningHandler::Create(parameters_);
    rl_handler_->startFires = [this](float percentage) {StartFires(percentage);};
    std::cout << "Created ReinforcementLearning Handler" << std::endl;
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
    std::pair<int, int> point = gridmap_->GetRandomPointInGrid();
    std::set<std::pair<int, int>> visited;
    std::queue<std::pair<int, int>> to_visit;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    to_visit.push(point);
    visited.insert(point);

    int ignited = 0;

    while (!to_visit.empty() && ignited < fires) {
        auto current = to_visit.front();
        to_visit.pop();

        if (gridmap_->CellCanIgnite(current.first, current.second)) {
            gridmap_->IgniteCell(current.first, current.second);
            ignited++;
        }

        // Get Moore Neighborhood
        auto neighbors = gridmap_->GetMooreNeighborhood(current.first, current.second);
        std::vector<std::pair<int, int>> ignitable_neighbors;

        // Collect neighbors we can ignite
        for (auto& neighbor : neighbors) {
            if (visited.find(neighbor) == visited.end() && gridmap_->CellCanIgnite(neighbor.first, neighbor.second)) {
                ignitable_neighbors.push_back(neighbor);
            }
        }

        // If we need more fires and the queue is about to run dry, force-push a neighbor
        if (to_visit.empty() && ignited < fires && !ignitable_neighbors.empty()) {
            // Randomly select one neighbor to guarantee some spread
            std::shuffle(ignitable_neighbors.begin(), ignitable_neighbors.end(), gen);
            to_visit.push(ignitable_neighbors.front());
            visited.insert(ignitable_neighbors.front());
            continue;
        }

        // Otherwise, do regular probabilistic addition
        for (auto& neighbor : ignitable_neighbors) {
            double randomValue = dist(gen);
            double fireProbability = parameters_.fire_spread_prob_ + (dist(gen) - 0.5) * parameters_.fire_noise_;
            if (randomValue < fireProbability) {
                to_visit.push(neighbor);
                visited.insert(neighbor);
            }
        }
    }
}

bool FireModel::InitialModeSelectionDone() {
    return parameters_.initial_mode_selection_done_ && parameters_.check_for_model_folder_empty_;
}

bool FireModel::GetEarlyClosing() {
    return !parameters_.exit_carefully_;
}
