// Dear ImGui: standalone example application for SDL2 + SDL_Renderer
// (SDL is a cross-platform general purpose library for handling windows, inputs, OpenGL/Vulkan/Metal graphics context creation, etc.)
// If you are new to Dear ImGui, read documentation from the docs/ folder + read the top of imgui.cpp.
// Read online: https://github.com/ocornut/imgui/tree/master/docs

// Important to understand: SDL_Renderer is an _optional_ component of SDL2.
// For a multi-platform app consider using e.g. SDL+DirectX on Windows and SDL+OpenGL on Linux/OSX.

#include "engine_core.h"

int main(int argc, char** argv)
{
    Mode mode = Mode::NoGUI;
    EngineCore::GetInstance()->Init(static_cast<int>(mode));

    while(EngineCore::GetInstance()->IsRunning()) {
        EngineCore::GetInstance()->HandleEvents();
        EngineCore::GetInstance()->Render();
        EngineCore::GetInstance()->Update();
    }

    EngineCore::GetInstance()->Clean();

    return 0;
}
