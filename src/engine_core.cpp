//
// Created by nex on 06.06.23.
//

#include "engine_core.h"

std::shared_ptr<EngineCore> EngineCore::instance_ = nullptr;


bool EngineCore::Init(int mode, const std::string& map_path){
    mode_ = static_cast<Mode>(mode);
    // Switch case for every mode and print the mode
    switch (mode_) {
        case Mode::GUI:
            std::cout << "Mode: GUI" << std::endl;
            break;
        case Mode::NoGUI:
            std::cout << "Mode: NoGUI" << std::endl;
            break;
        case Mode::GUI_RL:
            std::cout << "Mode: GUI_RL" << std::endl;
            break;
        case Mode::NoGUI_RL:
            std::cout << "Mode: NoGUI_RL" << std::endl;
            break;
        default:
            std::cerr << "Invalid mode: " << mode_ << std::endl;
            return false;
    }

    if (mode_ == Mode::GUI) {
        py::scoped_interpreter guard{};
    }

    bool init = true;
    if (mode_ == Mode::GUI || mode_ == Mode::GUI_RL) {
        init = this->SDLInit() && this->ImGuiInit();
        SDL_GetRendererOutputSize(renderer_.get(), &width_, &height_);
        ImVec2 window_size = ImVec2(250, 55);
        ImGui::SetNextWindowSize(window_size);
        ImVec2 appWindowPos = ImVec2((width_ - window_size.x) * 0.5f, (height_ - window_size.y) * 0.5f);
        ImGui::SetNextWindowPos(appWindowPos);
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.45f, 0.6f, 0.85f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.45f, 0.7f, 0.95f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.45f, 0.5f, 0.75f, 1.0f));
        model_ = FireModel::GetInstance(mode_);
        model_->SetRenderer(renderer_);
        ImGui::PopStyleColor(3);
    } else if (mode_ == Mode::NoGUI || mode_ == Mode::NoGUI_RL) {
        std::cout << "Loading Firemodel without GUI\n";
        model_ = FireModel::GetInstance(mode_, map_path);
        update_simulation_ = true;
        std::cout << "Firemodel loaded\n";

    }
    StartServer();
    return is_running_ = init;
}

void EngineCore::Update() {
    if (update_simulation_) {
        model_->Update();
        std::this_thread::sleep_for(std::chrono::milliseconds(delay_));
        //SDL_Delay(delay_);
    }
}

void EngineCore::Render() {
    if (mode_ == Mode::GUI || mode_ == Mode::GUI_RL) {
        // Start the Dear ImGui frame
        ImGui_ImplSDLRenderer2_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();

        if(model_ != nullptr) {
            model_->ImGuiRendering(update_simulation_, render_simulation_, delay_, io_->Framerate);
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
}

void EngineCore::HandleEvents() {
    // Poll and handle events (inputs, window resize, etc.)
    // You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to tell if dear imgui wants to use your inputs.
    // - When io.WantCaptureMouse is true, do not dispatch mouse input data to your main application, or clear/overwrite your copy of the mouse data.
    // - When io.WantCaptureKeyboard is true, do not dispatch keyboard input data to your main application, or clear/overwrite your copy of the keyboard data.
    // Generally you may always pass all inputs to dear imgui, and hide them from your application based on those two flags.
    if (mode_ == Mode::GUI || mode_ == Mode::GUI_RL) {
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
}

void EngineCore::Clean() {
    StopServer();
    // Cleanup GUI stuff
    if (mode_ == Mode::GUI || mode_ == Mode::GUI_RL) {
        ImGui_ImplSDLRenderer2_Shutdown();
        ImGui_ImplSDL2_Shutdown();
        SDL_Quit();
    }
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
    int ret = std::system(kill_command.c_str());
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

std::tuple<std::vector<std::deque<std::shared_ptr<State>>>, std::vector<double>, std::vector<bool>, std::pair<bool, bool>, double>
EngineCore::Step(std::vector<std::shared_ptr<Action>> actions) {
    return model_->Step(actions);
}

void EngineCore::SendDataToModel(std::string data) {
    if(model_ != nullptr){
        model_->GetData(data);
    }
}

void EngineCore::SendRLStatusToModel(pybind11::dict status) {
    if(model_ != nullptr){
        model_->SetRLStatus(status);
    }
}

pybind11::dict EngineCore::GetRLStatusFromModel() {
    if(model_ != nullptr){
        return model_->GetRLStatus();
    }
    return pybind11::dict();
}

std::string EngineCore::GetUserInput() {
    if(model_ != nullptr){
        return model_->GetUserInput();
    }
    // Return empty string if model is not initialized
    return "";
}

bool EngineCore::InitialModeSelectionDone() {
    return model_->InitialModeSelectionDone();
}

int EngineCore::GetViewRange() {
    if(model_ != nullptr){
        return model_->GetViewRange();
    }
    return 0;
}

int EngineCore::GetTimeSteps() {
    if(model_ != nullptr){
        return model_->GetTimeSteps();
    }
    return 0;
}

void EngineCore::StyleColorsEnemyMouse(ImGuiStyle* dst) {

    ImGuiStyle* style = dst ? dst : &ImGui::GetStyle();
    ImVec4* colors = style->Colors;

    style->Alpha = 1.0;
    style->ChildRounding = 3;
    style->WindowRounding = 3;
    style->GrabRounding = 1;
    style->GrabMinSize = 20;
    style->FrameRounding = 3;


    colors[ImGuiCol_Text] = ImVec4(0.00f, 1.00f, 1.00f, 1.00f);
    colors[ImGuiCol_TextDisabled] = ImVec4(0.00f, 0.40f, 0.41f, 1.00f);
    colors[ImGuiCol_WindowBg] = ImVec4(0.00f, 0.00f, 0.00f, 1.00f);
    colors[ImGuiCol_ChildBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_Border] = ImVec4(0.00f, 1.00f, 1.00f, 0.65f);
    colors[ImGuiCol_BorderShadow] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_FrameBg] = ImVec4(0.44f, 0.80f, 0.80f, 0.18f);
    colors[ImGuiCol_FrameBgHovered] = ImVec4(0.44f, 0.80f, 0.80f, 0.27f);
    colors[ImGuiCol_FrameBgActive] = ImVec4(0.44f, 0.81f, 0.86f, 0.66f);
    colors[ImGuiCol_TitleBg] = ImVec4(0.14f, 0.18f, 0.21f, 0.73f);
    colors[ImGuiCol_TitleBgCollapsed] = ImVec4(0.00f, 0.00f, 0.00f, 0.54f);
    colors[ImGuiCol_TitleBgActive] = ImVec4(0.00f, 1.00f, 1.00f, 0.27f);
    colors[ImGuiCol_MenuBarBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.20f);
    colors[ImGuiCol_ScrollbarBg] = ImVec4(0.22f, 0.29f, 0.30f, 0.71f);
    colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.00f, 1.00f, 1.00f, 0.44f);
    colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.00f, 1.00f, 1.00f, 0.74f);
    colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.00f, 1.00f, 1.00f, 1.00f);
    colors[ImGuiCol_CheckMark] = ImVec4(0.00f, 1.00f, 1.00f, 0.68f);
    colors[ImGuiCol_SliderGrab] = ImVec4(0.00f, 1.00f, 1.00f, 0.36f);
    colors[ImGuiCol_SliderGrabActive] = ImVec4(0.00f, 1.00f, 1.00f, 0.76f);
    colors[ImGuiCol_Button] = ImVec4(0.00f, 0.65f, 0.65f, 0.46f);
    colors[ImGuiCol_ButtonHovered] = ImVec4(0.01f, 1.00f, 1.00f, 0.43f);
    colors[ImGuiCol_ButtonActive] = ImVec4(0.00f, 1.00f, 1.00f, 0.62f);
    colors[ImGuiCol_Header] = ImVec4(0.00f, 1.00f, 1.00f, 0.33f);
    colors[ImGuiCol_HeaderHovered] = ImVec4(0.00f, 1.00f, 1.00f, 0.42f);
    colors[ImGuiCol_HeaderActive] = ImVec4(0.00f, 1.00f, 1.00f, 0.54f);
    colors[ImGuiCol_ResizeGrip] = ImVec4(0.00f, 1.00f, 1.00f, 0.54f);
    colors[ImGuiCol_ResizeGripHovered] = ImVec4(0.00f, 1.00f, 1.00f, 0.74f);
    colors[ImGuiCol_ResizeGripActive] = ImVec4(0.00f, 1.00f, 1.00f, 1.00f);
    colors[ImGuiCol_Button] = ImVec4(0.00f, 0.78f, 0.78f, 0.35f);
    colors[ImGuiCol_PlotLines] = ImVec4(0.00f, 1.00f, 1.00f, 1.00f);
    colors[ImGuiCol_PlotLinesHovered] = ImVec4(0.00f, 1.00f, 1.00f, 1.00f);
    colors[ImGuiCol_PlotHistogram] = ImVec4(0.00f, 1.00f, 1.00f, 1.00f);
    colors[ImGuiCol_PlotHistogramHovered] = ImVec4(0.00f, 1.00f, 1.00f, 1.00f);
    colors[ImGuiCol_TextSelectedBg] = ImVec4(0.00f, 1.00f, 1.00f, 0.22f);
}

bool EngineCore::SDLInit() {
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

    return true;
}

bool EngineCore::ImGuiInit() {
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

    this->StyleColorsEnemyMouse(&style);

    // Setup Platform/Renderer backends
    if (!ImGui_ImplSDL2_InitForSDLRenderer(window_.get(), renderer_.get())){
        SDL_Log("Error initializing ImGui_ImplSDL2_InitForSDLRenderer: %s\n", SDL_GetError());
        return false;
    }
    if (!ImGui_ImplSDLRenderer2_Init(renderer_.get())){
        SDL_Log("Error initializing ImGui_ImplSDLRenderer2_Init: %s\n", SDL_GetError());
        return false;
    }
    return true;
}

