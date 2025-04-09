//
// Created by nex on 28.11.24.
//

#ifndef ROSHAN_GROUNDSTATION_H
#define ROSHAN_GROUNDSTATION_H

#include <utility>
#include <SDL.h>
#include <deque>
#include <memory>
#include "reinforcementlearning/texturerenderer.h"
#include "firespin/model_parameters.h"

class Groundstation {
public:
    explicit Groundstation(std::pair<int, int> point, FireModelParameters &parameters);
    std::pair<int, int> GetGridPosition();
    std::pair<double, double> GetGridPositionDouble();
    void SetRenderer(SDL_Renderer* renderer) { renderer_ = TextureRenderer(renderer, "../assets/groundstation.png"); }
    void Render(std::pair<int, int> position, int size);
private:
    TextureRenderer renderer_;
    std::pair<double, double> position_; // x, y in (m)
    FireModelParameters &parameters_;
};


#endif //ROSHAN_GROUNDSTATION_H
