//
// Created by nex on 06.06.23.
//

#include "engine_core.h"
#include "firespin/rendering/imgui/ThemeManager.h"
#include "imgui_internal.h"
#include <utility>

bool EngineCore::Init(int mode, const std::string& config_path){
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

    bool init = true;
    if (mode_ == Mode::GUI || mode_ == Mode::GUI_RL) {
        init = this->SDLInit() && this->ImGuiInit();
        SDL_GetRendererOutputSize(renderer_, &width_, &height_);
        ImVec2 window_size = ImVec2(250, 55);
        ImGui::SetNextWindowSize(window_size);
        ImVec2 appWindowPos = ImVec2((width_ - window_size.x) * 0.5f, (height_ - window_size.y) * 0.5f);
        ImGui::SetNextWindowPos(appWindowPos);
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.45f, 0.6f, 0.85f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.45f, 0.7f, 0.95f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.45f, 0.5f, 0.75f, 1.0f));
        model_ = FireModel::Create(mode_, config_path);
        model_->SetRenderer(renderer_);
        ImGui::PopStyleColor(3);
    } else if (mode_ == Mode::NoGUI || mode_ == Mode::NoGUI_RL) {
        std::cout << "Loading Firemodel without GUI\n";
        model_ = FireModel::Create(mode_, config_path);
        update_simulation_ = true;
        std::cout << "Firemodel loaded\n";

    }
    StartServer();
    return is_running_ = init;
}

EngineCore::~EngineCore() = default;

void EngineCore::Clean() {
    model_.reset();
    // Cleanup GUI stuff
    if (mode_ == Mode::GUI || mode_ == Mode::GUI_RL) {
        ImGui_ImplSDLRenderer2_Shutdown();
        ImGui_ImplSDL2_Shutdown();
        ImGui::DestroyContext();

        SDL_DestroyRenderer(renderer_);
        SDL_DestroyWindow(window_);
        SDL_Quit();
    }
    StopServer();
    std::cout << "[Engine] Cleaned up successfully." << std::endl;
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

        // Dockspace host: panels dock into side rails, the passthru central
        // node stays empty so the SDL fire grid shows through behind it.
        ImGuiID dockspace_id = ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport(),
                                                            ImGuiDockNodeFlags_PassthruCentralNode);
        if (needs_default_layout_) {
            needs_default_layout_ = false;
            ImGui::DockBuilderRemoveNodeChildNodes(dockspace_id);
            ImGuiID main_id = dockspace_id;
            ImGuiID right = ImGui::DockBuilderSplitNode(main_id, ImGuiDir_Right, 0.30f, nullptr, &main_id);
            ImGuiID right_bottom = ImGui::DockBuilderSplitNode(right, ImGuiDir_Down, 0.35f, nullptr, &right);
            ImGui::DockBuilderDockWindow("Control Panel", right);
            ImGui::DockBuilderDockWindow("Simulation Parameters", right_bottom);
            ImGui::DockBuilderFinish(dockspace_id);
        }

        if(model_ != nullptr) {
            model_->ImGuiRendering(update_simulation_, render_simulation_, delay_, io_->Framerate);
        }

        // Rendering
        ImGui::Render();
        SDL_RenderSetScale(renderer_, io_->DisplayFramebufferScale.x, io_->DisplayFramebufferScale.y);
        if (model_ != nullptr && render_simulation_) {
            model_->Render();
        }
        else {
            SDL_Color color = {41, 49, 51, 255};
            SDL_SetRenderDrawColor(renderer_, color.r, color.g, color.b, color.a);
            SDL_RenderClear(renderer_);
        }
        ImGui_ImplSDLRenderer2_RenderDrawData(ImGui::GetDrawData(), renderer_);
        SDL_RenderPresent(renderer_);
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
        if (model_ != nullptr) {
            is_running_ = model_->GetEarlyClosing();
        }
        while (SDL_PollEvent(&event))
        {
            ImGui_ImplSDL2_ProcessEvent(&event);
            if (model_ != nullptr) {
                model_->HandleEvents(event, io_);
            }
            if (event.type == SDL_QUIT) {
                is_running_ = false;
            }
            if (event.type == SDL_WINDOWEVENT && event.window.event == SDL_WINDOWEVENT_CLOSE && event.window.windowID == SDL_GetWindowID(window_)) {
                is_running_ = false;
            }
        }
    }
    if (model_ != nullptr) {
        model_->CheckReset();
    }
}

void EngineCore::InitializeMap() {
    if(model_ != nullptr){
        model_->InitializeMap();
    }
}

void EngineCore::StartServer() {
    int result = std::system("cd ../openstreetmap && npm start");
    if(result == -1){
        std::cerr << "Failed to start OSM-Server. System command Errno: " << errno << std::endl;
    } else {
        int exit_status = WEXITSTATUS(result);
        std::cout << "Started OSM-Server with command status: " << exit_status << std::endl;
    }
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
    std::cout << "OSM-Server-Process with ID " << std::to_string(pid) << " stopped." << std::endl;
    auto c = std::system(kill_command.c_str());
}


pybind11::dict EngineCore::GetBatchedObservations(const std::string& agent_type) {
    return model_->GetBatchedObservations(agent_type);
}

void EngineCore::RefreshObservations(const std::string& agent_type) {
    model_->RefreshObservations(agent_type);
}

bool EngineCore::AgentIsRunning() {
    if(model_ != nullptr && update_simulation_){
        return model_->AgentIsRunning();
    } else {
        return false;
    }
}

StepResult EngineCore::Step(const std::string& agent_type, std::vector<std::shared_ptr<Action>> actions) {
    return model_->Step(agent_type, std::move(actions));
}

void EngineCore::SendDataToModel(const std::string& data) {
    if(model_ != nullptr){
        model_->GetData(data);
    }
}

void EngineCore::SendRLStatusToModel(pybind11::dict status) {
    if(model_ != nullptr){
        model_->SetRLStatus(std::move(status));
    }
}

void EngineCore::UpdateReward() {
    if(model_ != nullptr){
        model_->UpdateReward();
    }
}

pybind11::dict EngineCore::GetRLStatusFromModel() {
    if(model_ != nullptr){
        return model_->GetRLStatus();
    }
    return {};
}

std::string EngineCore::GetUserInput() {
    if(model_ != nullptr){
        return model_->GetUserInput();
    }
    // Return empty string if model is not initialized
    return "";
}

bool EngineCore::InitialModeSelectionDone() {
    // Early closing through Events
    if (!is_running_) {
        return true;
    }
    return model_->InitialModeSelectionDone();
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
    window_ = SDL_CreateWindow("ROSHAN", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, width_, height_, window_flags_);
    if (window_ == nullptr)
    {
        SDL_Log("Error creating SDL_Window: %s\n", SDL_GetError());
        return false;
    }
    SDL_MaximizeWindow(window_);

    renderer_ = SDL_CreateRenderer(window_, -1, SDL_RENDERER_ACCELERATED);
    if (renderer_ == nullptr)
    {
        SDL_Log("Error creating SDL_Renderer: %s\n", SDL_GetError());
        return false;
    }

    SDL_GetRendererOutputSize(renderer_, &width_, &height_);
    SDL_SetRenderDrawBlendMode(renderer_, SDL_BLENDMODE_BLEND);
    return true;
}

bool EngineCore::ImGuiInit() {
    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    io_ = &ImGui::GetIO();
    if (io_ == nullptr) {
        SDL_Log("Error getting ImGuiIO: %s\n", SDL_GetError());
        return false;
    }
    io_->ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    io_->ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls
    io_->ConfigFlags |= ImGuiConfigFlags_DockingEnable;         // No viewports: SDLRenderer2 backend can't
    // Stamp the default dock layout only when no saved layout exists
    needs_default_layout_ = io_->IniFilename == nullptr || !std::filesystem::exists(io_->IniFilename);

    // HiDPI: scale font and widget sizes by the framebuffer/window ratio
    float dpi_scale = 1.0f;
    int win_w = 0, out_w = 0;
    SDL_GetWindowSize(window_, &win_w, nullptr);
    SDL_GetRendererOutputSize(renderer_, &out_w, nullptr);
    if (win_w > 0 && out_w > win_w) {
        dpi_scale = static_cast<float>(out_w) / static_cast<float>(win_w);
    }
    ui::ThemeManager::SetUIScale(dpi_scale);

    // Change Font
    try {
        auto project_path = get_project_path("root_path", {});
        auto path_to_font = project_path / "assets" / "DejaVuSansMono.ttf";
        static const ImWchar ranges_mono[] = {
                0x0020, 0x00FF,   // Basic Latin + Latin-1
                0x2500, 0x257F,  // Box Drawing
                0x2580, 0x259F,  // Block Elements
                0x0370, 0x03FF,   // Greek (includes μ)
                0
        };
        ImFont* font = io_->Fonts->AddFontFromFileTTF(path_to_font.c_str(), 13.0f * dpi_scale, nullptr, ranges_mono);
        if (font == nullptr) {
            SDL_Log("Failed to load font from path: %s", path_to_font.c_str());
        }
        io_->Fonts->Build();
    } catch (const std::exception& e) {
        SDL_Log("Exception while trying to change the font: %s", e.what());
    }

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();
    style.FrameBorderSize = 1.0f; // Set border size to 1.0f
    style.FrameRounding = 6.0f; // Set rounding to 5.0f

    // Setup Platform/Renderer backends
    if (!ImGui_ImplSDL2_InitForSDLRenderer(window_, renderer_)){
        SDL_Log("Error initializing ImGui_ImplSDL2_InitForSDLRenderer: %s\n", SDL_GetError());
        return false;
    }
    if (!ImGui_ImplSDLRenderer2_Init(renderer_)){
        SDL_Log("Error initializing ImGui_ImplSDLRenderer2_Init: %s\n", SDL_GetError());
        return false;
    }
    return true;
}
