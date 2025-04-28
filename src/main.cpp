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
    engine->Init(static_cast<int>(mode));

    while(engine->IsRunning()) {
        engine->HandleEvents();
        engine->Update();
        engine->Render();
    }

    engine->Clean();
    engine.reset();
    return 0;
}
