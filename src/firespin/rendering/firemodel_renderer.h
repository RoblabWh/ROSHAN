//
// Created by nex on 08.06.23.
//

#ifndef ROSHAN_FIREMODEL_RENDERER_H
#define ROSHAN_FIREMODEL_RENDERER_H

#include <SDL.h>
#include "firemodel_camera.h"
#include "firemodel_pixelbuffer.h"
#include "firespin/firemodel_gridmap.h"
#include "firespin/particles/radiation_particle.h"
#include "firespin/particles/virtual_particle.h"
#include "firespin/model_parameters.h"
#include "imgui.h"
#include "firespin/utils.h"
#include <chrono>
#include <SDL_image.h>
#include <memory>
#include <utility>
#include "reinforcementlearning/agents/fly_agent.h"
#include "reinforcementlearning/groundstation.h"

#include "firespin/cell_classes/cell_generic_burned.h"
#include "firespin/cell_classes/cell_generic_unburned.h"
#include "firespin/cell_classes/cell_generic_burning.h"
#include "firespin/cell_classes/cell_lichens_and_mosses.h"
#include "firespin/cell_classes/cell_low_growing_woody_plants.h"
#include "firespin/cell_classes/cell_non_and_sparsley_vegetated.h"
#include "firespin/cell_classes/cell_outside_area.h"
#include "firespin/cell_classes/cell_periodically_herbaceous.h"
#include "firespin/cell_classes/cell_permanent_herbaceous.h"
#include "firespin/cell_classes/cell_sealed.h"
#include "firespin/cell_classes/cell_snow_and_ice.h"
#include "firespin/cell_classes/cell_water.h"
#include "firespin/cell_classes/cell_woody_breadleaved_deciduous_trees.h"
#include "firespin/cell_classes/cell_woody_broadleaved_evergreen_trees.h"
#include "firespin/cell_classes/cell_woody_needle_leaved_trees.h"

class FlyAgent;

enum class Palette {Fire, Trail};

class FireModelRenderer {
public:
    FireModelRenderer(SDL_Renderer* renderer, FireModelParameters& parameters);

    //only one instance of this class is allowed
    static std::shared_ptr<FireModelRenderer> Create(SDL_Renderer* renderer, FireModelParameters& parameters) {
        return std::make_shared<FireModelRenderer>(renderer, parameters);
    }

    void Render(const std::shared_ptr<std::vector<std::shared_ptr<FlyAgent>>>& drones);
    void SetScreenResolution();
    void SetGridMap(std::shared_ptr<GridMap> gridmap) { gridmap_ = std::move(gridmap); SetFullRedraw(); }
    SDL_Renderer* GetRenderer() { return renderer_; }

    // Converter Functions
    std::pair<int, int> ScreenToGridPosition(int x, int y);
    static ImVec4 GetMappedColor(int cell_type);

    // Camera functions
    void ChangeCameraPosition(double x, double y) { camera_.Move(x, y); SetFullRedraw();}
    void ApplyZoom(double z) { camera_.Zoom(z); SetFullRedraw();}

    // Drawing Related
    void SetFullRedraw() { needs_full_redraw_ = true; }
    void SetInitCellNoise() { gridmap_->GenerateNoiseMap(); }
    void ResizeEvent();
    void DrawArrow(double angle);
    void SetPalette(Palette p) {palette_ = p;}

    // Flash Screen
    void SetFlashScreen(bool flash_screen) { flash_screen_ = flash_screen; }
    void ShowGreenFlash() { show_green_flash_ = true; show_red_flash_=false; flash_start_time_ = SDL_GetTicks(); }
    void ShowRedFlash() { show_red_flash_ = true; show_green_flash_=false; flash_start_time_ = SDL_GetTicks(); }

    ~FireModelRenderer();

private:
    void DrawCells();
    void DrawCircle(int x, int y, int min_radius, double intensity);
    void DrawParticles();

    FireModelParameters& parameters_;
    FireModelCamera camera_;
    SDL_Renderer* renderer_;
    SDL_Texture* texture_;
    PixelBuffer* pixel_buffer_;
    SDL_PixelFormat* pixel_format_;
    SDL_Texture* arrow_texture_{};
    SDL_Texture * episode_succeeded_texture_{};
    SDL_Texture * episode_failed_texture_{};

    std::shared_ptr<GridMap> gridmap_;
    int width_{};
    int height_{};

    bool needs_full_redraw_;
    bool needs_init_cell_noise_;

    //Flash Screen
    bool flash_screen_ = parameters_.episode_termination_indicator_;
    bool show_green_flash_ = false;
    bool show_red_flash_ = false;
    Uint32 flash_start_time_ = 0;
    Uint32 flash_duration_ = 400;

    Palette palette_ = Palette::Fire;

    void DrawAllCells(int grid_left, int grid_right, int grid_top, int grid_bottom);
    void DrawChangesCells();
    SDL_Rect DrawCell(int x, int y);

    void ResizePixelBuffer();
    void ResizeTexture();
    void DrawDrones(const std::shared_ptr<std::vector<std::shared_ptr<FlyAgent>>>& drones);
    void DrawGroundstation(const std::shared_ptr<Groundstation>& groundstation);

    void FlashScreen();

    void DrawTrail(const std::deque<std::pair<double, double>>& trail);

    void DrawEpisodeEnd(bool success);

    void LoadSimpleTextures();
};


#endif //ROSHAN_FIREMODEL_RENDERER_H
