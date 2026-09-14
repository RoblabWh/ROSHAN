// Dear ImGui: standalone example application for SDL2 + SDL_Renderer
// (SDL is a cross-platform general purpose library for handling windows, inputs, OpenGL/Vulkan/Metal graphics context creation, etc.)
// If you are new to Dear ImGui, read documentation from the docs/ folder + read the top of imgui.cpp.
// Read online: https://github.com/ocornut/imgui/tree/master/docs

// Important to understand: SDL_Renderer is an _optional_ component of SDL2.
// For a multi-platform app consider using e.g. SDL+DirectX on Windows and SDL+OpenGL on Linux/OSX.

#include "engine_core.h"

int main(int argc, char** argv)
{
    py::scoped_interpreter guard{};  // LIVES until the very end

    Mode mode = Mode::GUI;
    auto engine = std::make_unique<EngineCore>();
    // Assets, config and the OSM server are addressed as "../x" relative to build/, so run
    // from there regardless of the launch cwd (main.py does the same os.chdir at startup).
    std::filesystem::current_path(get_project_path("module_directory", {}));
    engine->Init(static_cast<int>(mode));
    engine->InitializeMap();

    while(engine->IsRunning()) {
        engine->HandleEvents();
        engine->Update();
        engine->Render();
    }

    engine->Clean();
    engine.reset();
    return 0;
}
