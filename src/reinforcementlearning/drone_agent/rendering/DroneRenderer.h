//
// Created by nex on 13.07.23.
//

#ifndef ROSHAN_DRONERENDERER_H
#define ROSHAN_DRONERENDERER_H

#include <SDL.h>
#include <memory>

class DroneRenderer {
public:
    DroneRenderer() = default;
    DroneRenderer(std::shared_ptr<SDL_Renderer> renderer);
    ~DroneRenderer() = default;
    void Render(std::pair<int, int> position, int size, int view_range, double angle);
    void init();
private:
    std::shared_ptr<SDL_Texture> drone_texture_;
    std::shared_ptr<SDL_Renderer> renderer_;
};


#endif //ROSHAN_DRONERENDERER_H
