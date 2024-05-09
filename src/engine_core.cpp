//
// Created by nex on 06.06.23.
//

#include "engine_core.h"

std::shared_ptr<EngineCore> EngineCore::instance_ = nullptr;


bool EngineCore::Init(int mode){
    // Setup SDL
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER | SDL_INIT_GAMECONTROLLER) != 0)
    {
        SDL_Log("Failed to initialize SDL: %s\n", SDL_GetError());
        return false;
    }

    // From 2.0.18: Enable native IME.
#ifdef SDL_HINT_IME_SHOW_UI
    SDL_SetHint(SDL_HINT_IME_SHOW_UI, "1");
#endif

    // Create window with SDL_Renderer graphics context
    window_flags_ = (SDL_WindowFlags)(SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI | SDL_WINDOW_OPENGL);
    SDL_Window* window = SDL_CreateWindow("ROSHAN", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, width_, height_, window_flags_);
    if (window == nullptr)
    {
        SDL_Log("Error creating SDL_Window: %s\n", SDL_GetError());
        return false;
    }
    auto windowDeleter = [](SDL_Window* w) { SDL_DestroyWindow(w); };
    window_ = std::shared_ptr<SDL_Window>(window, windowDeleter);

    SDL_MaximizeWindow(window_.get());

    SDL_Renderer* renderer = SDL_CreateRenderer(window_.get(), -1, SDL_RENDERER_ACCELERATED);
    if (renderer == nullptr)
    {
        SDL_Log("Error creating SDL_Renderer: %s\n", SDL_GetError());
        return false;
    }
    auto rendererDeleter = [](SDL_Renderer* r) { SDL_DestroyRenderer(r); };
    renderer_ = std::shared_ptr<SDL_Renderer>(renderer, rendererDeleter);
    SDL_GetRendererOutputSize(renderer_.get(), &width_, &height_);

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO* io = &ImGui::GetIO();
    if (io == nullptr) {
        SDL_Log("Error getting ImGuiIO: %s\n", SDL_GetError());
        return false;
    }
    auto ioDeleter = [](ImGuiIO* i) { ImGui::DestroyContext(); };
    io_ = std::shared_ptr<ImGuiIO>(io, ioDeleter);
    io_->ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    io_->ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();
    style.FrameBorderSize = 1.0f; // Set border size to 1.0f
    style.FrameRounding = 6.0f; // Set rounding to 5.0f

    // Setup Platform/Renderer backends
    if (!ImGui_ImplSDL2_InitForSDLRenderer(window_.get(), renderer_.get())){
        SDL_Log("Error initializing ImGui_ImplSDL2_InitForSDLRenderer: %s\n", SDL_GetError());
        return false;
    }
    if (!ImGui_ImplSDLRenderer2_Init(renderer_.get())){
        SDL_Log("Error initializing ImGui_ImplSDLRenderer2_Init: %s\n", SDL_GetError());
        return false;
    }

    StartServer();
    mode_ = mode;
    model_ = nullptr;
    return is_running_ = true;
}

void EngineCore::Update() {
    if (update_simulation_) {
        SDL_GetRendererOutputSize(renderer_.get(), &width_, &height_);
        model_->SetWidthHeight(width_, height_);
        model_->Update();

        SDL_Delay(delay_);
    }
}

bool EngineCore::ImGuiModelSelection(){
    if (model_ == nullptr) {
        SDL_GetRendererOutputSize(renderer_.get(), &width_, &height_);
        ImVec2 window_size = ImVec2(250, 55);
        ImGui::SetNextWindowSize(window_size);
        ImVec2 appWindowPos = ImVec2((width_ - window_size.x) * 0.5f, (height_ - window_size.y) * 0.5f);
        ImGui::SetNextWindowPos(appWindowPos);

        ImGui::Begin("Model Selection", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove);
        //ImGui::Spacing();

        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.45f, 0.6f, 0.85f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.45f, 0.7f, 0.95f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.45f, 0.5f, 0.75f, 1.0f));

        if (ImGui::Button("FireSPIN", ImVec2(-1, 0))) {
            model_ = FireModel::GetInstance(renderer_, mode_);
        }

        ImGui::PopStyleColor(3);
        ImGui::End();
        return true;
    }
    return false;
}

void EngineCore::ImGuiSimulationControls(bool &update_simulation, bool &render_simulation, int &delay) {
    ImGui::Begin("Simulation Controls", nullptr);
    bool button_color = false;
    if (update_simulation) {
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.35f, 0.6f, 0.85f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.45f, 0.7f, 0.95f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.25f, 0.5f, 0.75f, 1.0f));
        button_color = true;
    }
    if (ImGui::Button(update_simulation ? "Stop Simulation" : "Start Simulation")) {
        update_simulation = !update_simulation;
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Click to %s the simulation.", update_simulation ? "stop" : "start");
    if (button_color) {
        ImGui::PopStyleColor(3);
    }
    ImGui::SameLine();

    button_color = false;
    if (render_simulation) {
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.35f, 0.6f, 0.85f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.45f, 0.7f, 0.95f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.25f, 0.5f, 0.75f, 1.0f));
        button_color = true;
    }
    if (ImGui::Button(render_simulation ? "Stop Rendering" : "Start Rendering")) {
        render_simulation = !render_simulation;
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Click to %s rendering the simulation.", render_simulation ? "stop" : "start");
    if (button_color) {
        ImGui::PopStyleColor(3);
    }
    ImGui::SameLine();

    ImGui::Text("Simulation Delay");
    ImGui::SliderInt("Delay (ms)", &delay, 0, 500);
    ImGui::Spacing();
    model_->ImGuiSimulationSpeed();
    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io_->Framerate, io_->Framerate);
    ImGui::End();
}

void EngineCore::Render() {
    // Start the Dear ImGui frame
    ImGui_ImplSDLRenderer2_NewFrame();
    ImGui_ImplSDL2_NewFrame();
    ImGui::NewFrame();

    if(!ImGuiModelSelection()) {
        std::function<void(bool&, bool&, int&)> controls =
                std::bind(&EngineCore::ImGuiSimulationControls, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
        model_->ImGuiRendering(controls, update_simulation_, render_simulation_, delay_);
    }

    // Rendering
    ImGui::Render();
    SDL_RenderSetScale(renderer_.get(), io_->DisplayFramebufferScale.x, io_->DisplayFramebufferScale.y);
    if (model_ != nullptr && render_simulation_) {
        model_->Render();
    }
    else {
        SDL_Color color = {41, 49, 51, 255};
        SDL_SetRenderDrawColor(renderer_.get(), color.r, color.g, color.b, color.a);
        SDL_RenderClear(renderer_.get());
    }
    ImGui_ImplSDLRenderer2_RenderDrawData(ImGui::GetDrawData());
    SDL_RenderPresent(renderer_.get());
}

void EngineCore::HandleEvents() {
    // Poll and handle events (inputs, window resize, etc.)
    // You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to tell if dear imgui wants to use your inputs.
    // - When io.WantCaptureMouse is true, do not dispatch mouse input data to your main application, or clear/overwrite your copy of the mouse data.
    // - When io.WantCaptureKeyboard is true, do not dispatch keyboard input data to your main application, or clear/overwrite your copy of the keyboard data.
    // Generally you may always pass all inputs to dear imgui, and hide them from your application based on those two flags.
    SDL_Event event;
    while (SDL_PollEvent(&event))
    {
        ImGui_ImplSDL2_ProcessEvent(&event);
        if (event.type == SDL_QUIT)
            is_running_ = false;
        if (event.type == SDL_WINDOWEVENT && event.window.event == SDL_WINDOWEVENT_CLOSE && event.window.windowID == SDL_GetWindowID(window_.get()))
            is_running_ = false;
        if (model_ != nullptr)
            model_->HandleEvents(event, io_.get());
    }
}

void EngineCore::Clean() {
    StopServer();
    // Cleanup
    ImGui_ImplSDLRenderer2_Shutdown();
    ImGui_ImplSDL2_Shutdown();

    // Is already handled by the smart pointer
//    ImGui::DestroyContext();
//    SDL_DestroyRenderer(renderer_.get());
//    SDL_DestroyWindow(window_.get());
    SDL_Quit();
}

void EngineCore::StartServer() {
    int result = std::system("cd ../openstreetmap && npm start");
    if(result == -1){
        std::cerr << "Failed to execute system command. Errno: " << errno << std::endl;
    } else {
        int exit_status = WEXITSTATUS(result);
        std::cout << "System command executed with exit status: " << exit_status << std::endl;
    }
//    TODO This is more elegant but doesn't work when executed in python since the
//    TODO working directory is the python directory and not the cpp directory.
//    TODO Maybe we can change the working directory in python?
//    char abspath[PATH_MAX];
//    ssize_t count = readlink("/proc/self/exe", abspath, PATH_MAX);
//    std::filesystem::path exec_path = std::string(abspath, (count > 0) ? count : 0);
//    std::filesystem::path exec_directory = exec_path.parent_path();
//
//    std::filesystem::path osm_directory = exec_directory / ".." / "openstreetmap";
//
//    if (std::filesystem::exists(osm_directory)) {
//        std::string command = "cd " + osm_directory.string() + " && npm start";
//        int result = std::system(command.c_str());
//        if(result == -1){
//            std::cerr << "Failed to execute system command. Errno: " << errno << std::endl;
//        } else {
//            int exit_status = WEXITSTATUS(result);
//            std::cout << "System command executed with exit status: " << exit_status << std::endl;
//        }
//    } else {
//        std::cerr << "Directory does not exist: " << osm_directory << std::endl;
//    }
}

void EngineCore::StopServer() {
    std::ifstream pidFile("../openstreetmap/server.pid");
    if (!pidFile) {
        std::cerr << "Failed to find server.pid file. Can't close nodejs server\n" << std::endl;
        return;
    }

    int pid;
    pidFile >> pid;
    if (!pidFile) {
        std::cerr << "Failed to read pid from server.pid file. Can't close nodejs server\n" << std::endl;
        return;
    }

    std::string kill_command = "kill " + std::to_string(pid);
    std::cout << kill_command << std::endl;
    std::system(kill_command.c_str());
}


std::vector<std::deque<std::shared_ptr<State>>> EngineCore::GetObservations() {
    return model_->GetObservations();
}

bool EngineCore::AgentIsRunning() {
    if(model_ != nullptr && update_simulation_){
        return model_->AgentIsRunning();
    } else {
        return false;
    }
}

std::tuple<std::vector<std::deque<std::shared_ptr<State>>>, std::vector<double>, std::vector<bool>, std::pair<bool, bool>>
EngineCore::Step(std::vector<std::shared_ptr<Action>> actions) {
    return model_->Step(actions);
}


