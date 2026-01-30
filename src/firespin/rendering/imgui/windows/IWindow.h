//
// IWindow.h - Base interface for ImGui windows
//
// All modular window classes implement this interface for consistent management.
//

#ifndef ROSHAN_IWINDOW_H
#define ROSHAN_IWINDOW_H

namespace ui {

class IWindow {
public:
    virtual ~IWindow() = default;

    // Render the window contents (called every frame)
    virtual void Render() = 0;

    // Get window visibility
    virtual bool IsVisible() const = 0;

    // Set window visibility
    virtual void SetVisible(bool visible) = 0;

    // Get window name/title (for debugging/logging)
    virtual const char* GetName() const = 0;
};

} // namespace ui

#endif // ROSHAN_IWINDOW_H
