//
// Created by nex on 08.06.23.
//

#include <iostream>
#include <utility>
#include "firemodel_renderer.h"

FireModelRenderer::FireModelRenderer(SDL_Renderer* renderer, FireModelParameters& parameters)
    : parameters_(parameters), renderer_(renderer), camera_(FireModelCamera()) {
    SetScreenResolution();
    texture_ = SDL_CreateTexture(renderer_,SDL_PIXELFORMAT_ARGB8888,SDL_TEXTUREACCESS_STREAMING,width_,height_);

    this->LoadSimpleTextures();

    // This whole overhead just to get the format...
    Uint32 format;
    int access, w, h;
    SDL_QueryTexture(texture_, &format, &access, &w, &h);
    pixel_format_ = SDL_AllocFormat(format);

    pixel_buffer_ = new PixelBuffer(width_, height_, parameters_.background_color_, pixel_format_);
    needs_full_redraw_ = true;
    needs_init_cell_noise_ = false;
    if (texture_ == nullptr) {
        SDL_Log("Unable to create texture from surface: %s", SDL_GetError());
        return;
    }
}


void FireModelRenderer::LoadSimpleTextures() {
    // Load the arrow texture
    SDL_Surface *arrow_surface = IMG_Load("../assets/arrow.png");
    if (arrow_surface == nullptr) {
        SDL_Log("Unable to load image: %s", SDL_GetError());
        return;
    }
    SDL_Surface *episode_succeeded_surface = IMG_Load("../assets/episode_succeeded.png");
    if (episode_succeeded_surface == nullptr) {
        SDL_Log("Unable to load image: %s", SDL_GetError());
        return;
    }
    SDL_Surface *episode_failed_surface = IMG_Load("../assets/episode_failed.png");
    if (episode_failed_surface == nullptr) {
        SDL_Log("Unable to load image: %s", SDL_GetError());
        return;
    }

    arrow_texture_ = SDL_CreateTextureFromSurface(renderer_, arrow_surface);
    SDL_FreeSurface(arrow_surface);

    episode_succeeded_texture_ = SDL_CreateTextureFromSurface(renderer_, episode_succeeded_surface);
    SDL_FreeSurface(episode_succeeded_surface);

    episode_failed_texture_ = SDL_CreateTextureFromSurface(renderer_, episode_failed_surface);
    SDL_FreeSurface(episode_failed_surface);

    // Set blend mode to the texture for transparency
    if(SDL_SetTextureBlendMode(arrow_texture_, SDL_BLENDMODE_BLEND) < 0) {
        printf("Set blend mode error: %s\n", SDL_GetError());
    }

    // Set the transparency for the texture
    if(SDL_SetTextureAlphaMod(arrow_texture_, 127) < 0) { // 127 for semi transparency, can be changed according to the needs
        printf("Set texture alpha error: %s\n", SDL_GetError());
        // Handle the error
    }
}

// Cached burn_time for the current frame — set once per Render() call
static double s_cached_burn_time = 0.0;

// Precomputed flicker lookup table (256 entries for fast sin approximation)
static constexpr int FLICKER_LUT_SIZE = 256;
static Uint8 s_flicker_lut[FLICKER_LUT_SIZE];
static bool s_flicker_lut_initialized = false;

static void InitFlickerLUT() {
    if (s_flicker_lut_initialized) return;
    for (int i = 0; i < FLICKER_LUT_SIZE; ++i) {
        double phase = (2.0 * M_PI * i) / FLICKER_LUT_SIZE;
        double flicker = (std::sin(phase) + 1.0) * 0.9; // 0..1.8 range
        s_flicker_lut[i] = static_cast<Uint8>(flicker * 64);
    }
    s_flicker_lut_initialized = true;
}

void FireModelRenderer::Render(const std::shared_ptr<std::vector<std::shared_ptr<FlyAgent>>>& drones) {
    SDL_RenderClear(renderer_);
    if (gridmap_ != nullptr) {
        // Cache frame time once and initialize flicker LUT (first call only)
        InitFlickerLUT();
        s_cached_burn_time = SDL_GetTicks() / 1000.0;

        camera_.Update(width_, height_, gridmap_->GetRows(), gridmap_->GetCols());
        // Smart redraw: only trigger full redraw when camera is actively animating
        if (camera_.IsAnimating()) {
            SetFullRedraw();
        }
        if (this->needs_init_cell_noise_) {
            gridmap_->GenerateNoiseMap();
            this->needs_init_cell_noise_ = false;
        }
        DrawCells();
        DrawGroundstation(gridmap_->GetGroundstation());
        if (parameters_.render_particles_)
            DrawParticles();
        DrawDrones(drones);
        FlashScreen();
    }
}

void FireModelRenderer::ResizePixelBuffer() {
    if (pixel_buffer_ != nullptr)
        pixel_buffer_->Resize(width_, height_);
}

void FireModelRenderer::ResizeTexture() {
    if (texture_ != nullptr)
        SDL_DestroyTexture(texture_);
    texture_ = SDL_CreateTexture(renderer_,SDL_PIXELFORMAT_ARGB8888,SDL_TEXTUREACCESS_STREAMING,width_,height_);
}

void FireModelRenderer::ResizeEvent() {
    SetScreenResolution();
    ResizePixelBuffer();
    ResizeTexture();
    SetFullRedraw();
}

void FireModelRenderer::SetScreenResolution() {
    SDL_GetRendererOutputSize(renderer_, &width_, &height_);
    camera_.SetViewport(width_, height_);
}

void FireModelRenderer::DrawCells() {

    if (needs_full_redraw_) {
        // Convert viewport corners to grid coordinates
        auto [gridLeft, gridTop] = camera_.ScreenToGridPosition(0, 0);
        auto [gridRight, gridBottom] = camera_.ScreenToGridPosition(camera_.GetViewportWidth(), camera_.GetViewportHeight());

        // Ensure grid coordinates are within gridmap bounds
        gridLeft = std::max(gridLeft, 0);
        gridTop = std::max(gridTop, 0);
        gridRight = std::min(gridRight, gridmap_->GetRows() - 1);
        gridBottom = std::min(gridBottom, gridmap_->GetCols() - 1);

        pixel_buffer_->Reset();
        DrawAllCells(gridLeft, gridRight, gridTop, gridBottom);
        needs_full_redraw_ = false;
        SDL_UpdateTexture(texture_, nullptr, pixel_buffer_->GetData(), pixel_buffer_->GetPitch());
        gridmap_->ResetChangedCells();
    } else {
        DrawChangesCells();
    }

    // Render the texture to the screen
    SDL_RenderCopy(renderer_, texture_, nullptr, nullptr);
}

void FireModelRenderer::DrawAllCells(int grid_left, int grid_right, int grid_top, int grid_bottom) {
    // Start drawing cells from the cell at the camera position
    for (int x = grid_left; x <= grid_right; ++x) {
        for (int y = grid_top; y <= grid_bottom; ++y) {
            DrawCell(x, y);
        }
    }
}

void FireModelRenderer::DrawChangesCells() {
    const auto& changed_cells = gridmap_->GetChangedCells();
    if (changed_cells.empty()) {
        gridmap_->ResetChangedCells();
        return;
    }

    // Compute bounding box of all changed cells for a single SDL_UpdateTexture call
    int bbox_min_x = std::numeric_limits<int>::max();
    int bbox_min_y = std::numeric_limits<int>::max();
    int bbox_max_x = std::numeric_limits<int>::min();
    int bbox_max_y = std::numeric_limits<int>::min();

    // Draw all changed cells into the pixel buffer and track the bounding box
    for (const auto& cell : changed_cells) {
        SDL_Rect cell_rect = DrawCell(cell.x_, cell.y_);
        if (cell_rect.x < 0 || cell_rect.y < 0 ||
            cell_rect.x >= pixel_buffer_->GetWidth() ||
            cell_rect.y >= pixel_buffer_->GetHeight()) {
            continue;
        }

        int clamped_w = cell_rect.w;
        int clamped_h = cell_rect.h;
        if (cell_rect.x + clamped_w > pixel_buffer_->GetWidth()) {
            clamped_w = pixel_buffer_->GetWidth() - cell_rect.x;
        }
        if (cell_rect.y + clamped_h > pixel_buffer_->GetHeight()) {
            clamped_h = pixel_buffer_->GetHeight() - cell_rect.y;
        }

        bbox_min_x = std::min(bbox_min_x, cell_rect.x);
        bbox_min_y = std::min(bbox_min_y, cell_rect.y);
        bbox_max_x = std::max(bbox_max_x, cell_rect.x + clamped_w);
        bbox_max_y = std::max(bbox_max_y, cell_rect.y + clamped_h);
    }

    // Single batched texture upload for the entire bounding box
    if (bbox_min_x < bbox_max_x && bbox_min_y < bbox_max_y) {
        SDL_Rect bbox = {bbox_min_x, bbox_min_y, bbox_max_x - bbox_min_x, bbox_max_y - bbox_min_y};
        Uint32* bboxPixelData = &pixel_buffer_->GetData()[bbox_min_y * pixel_buffer_->GetWidth() + bbox_min_x];
        SDL_UpdateTexture(texture_, &bbox, bboxPixelData, pixel_buffer_->GetPitch());
    }

    gridmap_->ResetChangedCells();
}

std::pair<Uint32, int> ModifyColorWithGradient(Uint32 base_color, int x, int y, Uint32 time) {
    Uint8 r = (base_color >> 24) & 0xFF;
    Uint8 g = (base_color >> 16) & 0xFF;
    Uint8 b = (base_color >> 8) & 0xFF;
    Uint8 a = base_color & 0xFF;

    r = std::min(255, r + (x % 32));
    g = std::min(255, g + (y % 32));
    b = std::min(255, b + ((x + y) % 32));

    // Use cached burn_time (set once per frame) and precomputed flicker LUT
    static constexpr double FLICKER_LUT_SCALE = FLICKER_LUT_SIZE / (2.0 * M_PI);
    int lut_index = static_cast<int>((s_cached_burn_time * 5.0 + x + y) * FLICKER_LUT_SCALE);
    lut_index = ((lut_index % FLICKER_LUT_SIZE) + FLICKER_LUT_SIZE) % FLICKER_LUT_SIZE;
    Uint8 offset = s_flicker_lut[lut_index];

    r = std::min(255, static_cast<int>(r) + offset);
    g = std::min(255, static_cast<int>(g) + offset);
    b = std::min(255, static_cast<int>(b) + offset);

    int phase_offset = 0;
    if (time > 0) {
        phase_offset = static_cast<int>((time / 50));
    }

    return { (r << 24) | (g << 16) | (b << 8) | a, phase_offset };
}

SDL_Rect FireModelRenderer::DrawCell(int x, int y) {
    auto [screen_x, screen_y] = camera_.GridToScreenPosition(floor(x), floor(y));
    auto [next_screen_x, next_screen_y] = camera_.GridToScreenPosition(floor(x) + 1, floor(y) + 1);
    int cell_w = next_screen_x - screen_x;
    int cell_h = next_screen_y - screen_y;
    const SDL_Rect cell_rect = { screen_x, screen_y, cell_w, cell_h };

    auto color_for = [&](int gx, int gy) {
        auto& cell = gridmap_->At(gx, gy);
        Uint32 color = cell.GetMappedColor();
        if (parameters_.lingering_ && cell.WasFlooded())
            color = (255 << 24) | (77 << 16) | (187 << 8) | 230;
        CellState st = cell.GetCellState();
        int phase_offset = 0;
        if (st == CellState::GENERIC_BURNING) {
            auto res = ModifyColorWithGradient(color, gx, gy, 0);
            color = res.first;
            phase_offset = res.second;
        }
        return std::pair<Uint32, int>(color, phase_offset);
    };
    auto base = color_for(x, y);
    auto base_color = base.first;
    auto phase_offset = base.second;

    int grid_offset = !parameters_.render_grid_ ? 0 : (camera_.GetCellSize() >= 3.0 ? -1 : 0);
    auto& this_cell = gridmap_->At(x, y);
    bool has_noise = this_cell.HasNoise() && gridmap_->HasNoiseGenerated() && parameters_.has_noise_ && !this->needs_init_cell_noise_;

    if (has_noise) {
        std::vector<std::vector<int>>& noise_map = this_cell.GetNoiseMap();
        if (!noise_map.empty()){
            pixel_buffer_->Draw(cell_rect, base_color, noise_map, grid_offset, phase_offset);
        } else {
            this->needs_full_redraw_ = true;
        }
    }
    else {
        pixel_buffer_->Draw(cell_rect, base_color, grid_offset);
    }

    if (parameters_.render_terrain_transition) {
        auto this_cell_state = this_cell.GetCellState();
        if (gridmap_->IsPointInGrid(x, y - 1) && gridmap_->At(x, y - 1).GetCellState() < this_cell_state){
            auto left_color = color_for(x, y - 1).first;
            if (left_color != base_color)
                pixel_buffer_->DrawBlendedEdge(cell_rect, base_color, left_color, Edge::Left);
        }
        if (gridmap_->IsPointInGrid(x, y + 1) && gridmap_->At(x, y + 1).GetCellState() < this_cell_state) {
            auto right_color = color_for(x, y + 1).first;
            if (right_color != base_color)
                pixel_buffer_->DrawBlendedEdge(cell_rect, base_color, right_color, Edge::Right);
        }
        if (gridmap_->IsPointInGrid(x - 1, y) && gridmap_->At(x - 1, y).GetCellState() < this_cell_state) {
            auto top_color = color_for(x - 1, y).first;
            if (top_color != base_color)
                pixel_buffer_->DrawBlendedEdge(cell_rect, base_color, top_color, Edge::Top);
        }
        if (gridmap_->IsPointInGrid(x + 1, y) && gridmap_->At(x + 1, y).GetCellState() < this_cell_state) {
            auto bottom_color = color_for(x + 1, y).first;
            if (bottom_color != base_color)
                pixel_buffer_->DrawBlendedEdge(cell_rect, base_color, bottom_color, Edge::Bottom);
        }
    }

    return cell_rect;
}

void FireModelRenderer::DrawCircle(int x, int y, int min_radius, double intensity) {
    int max_radius = 3 * min_radius;
//    int radius = min_radius + static_cast<int>((max_radius - min_radius) * ((intensity - 0.2) / 0.8));
    const int radius = min_radius + static_cast<int>((max_radius - min_radius) * intensity);
    SDL_Color color{};
    switch (palette_) {
        case Palette::Fire: {
            Uint8 r, g, b;
            double rel_intense = intensity;
            if (intensity <= 0.3) {
                // Dark red to light red gradient based on intensity (0 -> dark red, 1 -> light red)
                rel_intense = (intensity / 0.3);
                r = 128 + static_cast<Uint8>(127  * rel_intense);
                g = static_cast<Uint8>(100.0 * rel_intense);
                b = 0;
            } else if (intensity <= 1 /**0.7**/) {
                // Light red to yellow gradient based on intensity (0.3 -> light red, 0.6 -> yellow)
                rel_intense = (intensity / 0.7);
                r = 255;
                g = 128 + static_cast<Uint8>(127 * rel_intense);
                b = 0;
            } else {
                // Yellow to white gradient based on intensity (0.6 -> yellow, 1 -> white)
                r = 255;
                g = 255;
                b = static_cast<Uint8>(128 * intensity);
            }
            color = SDL_Color{r, g, b, static_cast<Uint8>(255.0 * (0.6 + 0.4 * rel_intense))};
            break;
        }
        case Palette::Trail: {
            // Light grey to white gradient based on intensity (0 -> light grey, 1 -> white)
            auto grey = static_cast<Uint8>(255 - 55 * intensity); // 200..255
            color = SDL_Color{grey, grey, grey, static_cast<Uint8>(255.0 * (0.5 + 0.5 * intensity))};
            break;
        }
    }

    SDL_SetRenderDrawBlendMode(renderer_, SDL_BLENDMODE_BLEND);
    SDL_SetRenderDrawColor(renderer_, color.r, color.g, color.b, color.a);

    const int r2 = radius * radius;
    // Pre-allocate points buffer (pi*r^2 max points)
    thread_local std::vector<SDL_Point> points;
    points.clear();
    for(int w = 0; w < radius * 2; w++) {
        int dx = radius - w;
        int dx2 = dx * dx;
        for(int h = 0; h < radius * 2; h++) {
            int dy = radius - h;
            if(dx2 + dy * dy <= r2) {
                points.push_back({x + dx, y + dy});
            }
        }
    }
    if (!points.empty()) {
        SDL_RenderDrawPoints(renderer_, points.data(), static_cast<int>(points.size()));
    }
}

void FireModelRenderer::DrawParticles() {

    int circle_radius = static_cast<int>(camera_.GetCellSize() / 6);

    if (circle_radius > 0) {
        // Pre-compute viewport bounds for frustum culling
        const int vp_w = static_cast<int>(camera_.GetViewportWidth());
        const int vp_h = static_cast<int>(camera_.GetViewportHeight());
        // Margin to account for particle radius (max 3x circle_radius)
        const int margin = circle_radius * 3;

        const double inv_cell_size = 1.0 / parameters_.GetCellSize();

        const std::vector<RadiationParticle>& particles = gridmap_->GetRadiationParticles();

        if (!particles.empty()) {
            for (const auto& particle : particles) {
                double x, y;
                particle.GetPosition(x, y);

                x *= inv_cell_size;
                y *= inv_cell_size;
                auto [posx, posy] = camera_.GridToScreenPosition(x, y);

                // Frustum culling: skip particles outside viewport
                if (posx < -margin || posx > vp_w + margin ||
                    posy < -margin || posy > vp_h + margin) {
                    continue;
                }

                DrawCircle(posx, posy, circle_radius, particle.GetIntensity());
            }
        }

        const std::vector<VirtualParticle>& virtual_particles = gridmap_->GetVirtualParticles();

        if (!virtual_particles.empty()) {
            for (const auto& particle : virtual_particles) {
                double x, y;
                particle.GetPosition(x, y);
                x *= inv_cell_size;
                y *= inv_cell_size;
                auto [posx, posy] = camera_.GridToScreenPosition(x, y);

                // Frustum culling: skip particles outside viewport
                if (posx < -margin || posx > vp_w + margin ||
                    posy < -margin || posy > vp_h + margin) {
                    continue;
                }

                DrawCircle(posx, posy, circle_radius, particle.GetIntensity());
            }
        }
    }

}

void FireModelRenderer::DrawEpisodeEnd(bool success) {
    auto width = width_ - 100;
    auto height = 20;

    if (success) {
        SDL_Rect destRect = {width, height, 100, 100}; // x, y, width and height of the arrow
        SDL_RenderCopyEx(renderer_, episode_succeeded_texture_, nullptr, &destRect, 0, nullptr, SDL_FLIP_NONE);
    } else {
        SDL_Rect destRect = {width, height, 100, 100}; // x, y, width and height of the arrow
        SDL_RenderCopyEx(renderer_, episode_failed_texture_, nullptr, &destRect, 0, nullptr, SDL_FLIP_NONE);
    }
}

void FireModelRenderer::DrawArrow(double angle) {
    // Render the arrow
    SDL_Rect destRect = {width_ - 100, height_ - 100, 50, 50}; // x, y, width and height of the arrow
    SDL_RenderCopyEx(renderer_, arrow_texture_, nullptr, &destRect, angle, nullptr, SDL_FLIP_NONE);
}

std::pair<int, int> FireModelRenderer::ScreenToGridPosition(int x, int y) {
    auto [screenX, screenY] = camera_.ScreenToGridPosition(x, y);

    return std::make_pair(screenX, screenY);
}

void FireModelRenderer::ApplyZoom(double z, int mouseX, int mouseY) {
    if (gridmap_) {
        camera_.ZoomToPoint(z, mouseX, mouseY, gridmap_->GetRows(), gridmap_->GetCols());
    } else {
        camera_.Zoom(z);
    }
}

void FireModelRenderer::FocusOnPoint(double gridX, double gridY) {
    if (!gridmap_) return;
    // Center the target so gridX, gridY is at viewport center
    double targetX = static_cast<double>(gridmap_->GetRows()) / 2.0 - gridX;
    double targetY = static_cast<double>(gridmap_->GetCols()) / 2.0 - gridY;
    camera_.SetTarget(targetX, targetY);
}

void FireModelRenderer::FocusOnFire() {
    if (!gridmap_) return;
    auto centroid = gridmap_->GetFireCentroid();
    FocusOnPoint(centroid.first, centroid.second);
}

void FireModelRenderer::FocusOnDrone(int droneIndex, const std::shared_ptr<std::vector<std::shared_ptr<FlyAgent>>>& drones) {
    if (!drones || droneIndex < 0 || droneIndex >= static_cast<int>(drones->size())) return;
    auto pos = (*drones)[droneIndex]->GetGridPositionDouble();
    FocusOnPoint(pos.first, pos.second);
}

FireModelRenderer::~FireModelRenderer() {
    //Destroy Backbuffer
    delete pixel_buffer_;
    SDL_DestroyTexture(arrow_texture_);
    SDL_DestroyTexture(texture_);
    SDL_FreeFormat(pixel_format_);
}

void FireModelRenderer::DrawDrones(const std::shared_ptr<std::vector<std::shared_ptr<FlyAgent>>>& drones) {
    if (drones->empty()) {
        return;
    }
    for (auto &agent : *drones) {
        auto trail = agent->GetCameraTrail(camera_);
        this->DrawTrail(trail);
        agent->Render(camera_);
    }
}

void FireModelRenderer::DrawTrail(const std::deque<std::pair<double, double>>& trail) {
    this->SetPalette(Palette::Trail);
    const int count = static_cast<int>(trail.size());
    const int base_thickness = 1; // can change it to a larger value for thicker trails

    for (int i = 1; i < count; ++i) {
        // Intensity used for fading here: [0, 1]
        const double intensity = std::min((static_cast<double>(i) / count), 0.8);

        int x1 = static_cast<int>(trail[i - 1].first);
        int y1 = static_cast<int>(trail[i - 1].second);
        int x2 = static_cast<int>(trail[i].first);
        int y2 = static_cast<int>(trail[i].second);

        const double dx = x2 - x1;
        const double dy = y2 - y1;
        const double dist = std::max(1.0, std::hypot(dx, dy));
        const double step = 2; // smaller = denser = smoother (costlier)
        const int samples = static_cast<int>(dist / step);

        for (int s = 0; s <= samples; ++s) {
            const double t = samples ? (static_cast<double>(s) / samples) : 0.0;
            const int xs = static_cast<int>(x1 + dx * t);
            const int ys = static_cast<int>(y1 + dy * t);
            DrawCircle(xs, ys, base_thickness, intensity);
        }
    }

    this->SetPalette(Palette::Fire);
}

void FireModelRenderer::DrawGroundstation(const std::shared_ptr<Groundstation>& groundstation) {
    if (groundstation == nullptr) {
        return;
    }

    int size = static_cast<int>(camera_.GetCellSize());

    std::pair<double, double> station_position = groundstation->GetGridPositionDouble();

    std::pair<int, int> screen_position = camera_.GridToScreenPosition(station_position.first -0.5,
                                                                       station_position.second - 0.5);
    groundstation->Render(screen_position, static_cast<int>(size * 1));
}

ImVec4 FireModelRenderer::GetMappedColor(int cell_type) {
    SDL_Color color;
    // Create switch statement for each cell type
    switch (static_cast<CellState>(cell_type)) {
        case CellState::GENERIC_UNBURNED:
            color = {50, 190, 75, 255}; break;
        case CellState::SEALED:
            color = {100, 100, 100, 255}; break;
        case CellState::WOODY_NEEDLE_LEAVED_TREES:
            color = {0, 230, 0, 255}; break;
        case CellState::WOODY_BROADLEAVED_DECIDUOUS_TREES:
            color = {0, 150, 0, 255}; break;
        case CellState::WOODY_BROADLEAVED_EVERGREEN_TREES:
            color = {0, 255, 0, 255}; break;
        case CellState::LOW_GROWING_WOODY_PLANTS:
            color = {105, 76, 51, 255}; break;
        case CellState::PERMANENT_HERBACEOUS:
            color = {250, 218, 94, 255}; break;
        case CellState::PERIODICALLY_HERBACEOUS:
            color = {240, 230, 140, 255}; break;
        case CellState::LICHENS_AND_MOSSES:
            color = {255, 153, 204, 255}; break;
        case CellState::NON_AND_SPARSLEY_VEGETATED:
            color = {194, 178, 128, 255}; break;
        case CellState::WATER:
            color = {0, 0, 255, 255}; break;
        case CellState::SNOW_AND_ICE:
            color = {0, 255, 255, 255}; break;
        case CellState::OUTSIDE_AREA:
            color = {25, 25, 25, 255}; break;
        case CellState::GENERIC_BURNING:
            color = {255, 0, 0, 255}; break;
        case CellState::GENERIC_BURNED:
            color = { 42, 42, 42, 255 }; break;
        case CellState::GENERIC_FLOODED:
            color = {77, 187, 230, 255}; break;
        default:
            color = { 80, 80, 80, 255 }; break;
    }
    return {color.r / 255.0f, color.g / 255.0f, color.b / 255.0f, 1.0f};
}

void FireModelRenderer::FlashScreen() {
    if ((show_green_flash_ || show_red_flash_) && flash_screen_){
        Uint32 current_time = SDL_GetTicks();
        Uint32 elapsed_time = current_time - flash_start_time_;
        if (elapsed_time < flash_duration_){
            this->DrawEpisodeEnd(show_green_flash_); // Draw the episode end texture
//            if (show_green_flash_) {
//                SDL_SetRenderDrawColor(renderer_, 0, 255, 0, 128);
//            } else {
//                SDL_SetRenderDrawColor(renderer_, 255, 0, 0, 128);
//            }
//            auto [gridLeft, gridTop] = camera_.GridToScreenPosition(0, 0);
//            SDL_Rect fullscreen_rect{gridLeft, gridTop, static_cast<int>(camera_.GetCellSize()) * gridmap_->GetCols(), static_cast<int>(camera_.GetCellSize()) * gridmap_->GetRows()};
//            SDL_RenderFillRect(renderer_, &fullscreen_rect);
        } else {
            show_green_flash_ = false;
            show_red_flash_ = false;
        }
    }
}

