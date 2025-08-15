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

std::vector<std::vector<int>> ScaleNoiseMap(const std::vector<std::vector<int>>& noise_map, int new_size) {
    int original_size = static_cast<int>(noise_map.size());
    std::vector<std::vector<int>> scaled_noise_map(new_size, std::vector<int>(new_size));

    for (int y = 0; y < new_size; ++y) {
        for (int x = 0; x < new_size; ++x) {
            int orig_x = x * original_size / new_size;
            int orig_y = y * original_size / new_size;
            scaled_noise_map[y][x] = noise_map[orig_y][orig_x];
        }
    }
    return scaled_noise_map;
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

void FireModelRenderer::Render(const std::shared_ptr<std::vector<std::shared_ptr<FlyAgent>>>& drones) {
    SDL_RenderClear(renderer_);
    if (gridmap_ != nullptr) {
        camera_.Update(width_, height_, gridmap_->GetRows(), gridmap_->GetCols());
        if (this->needs_init_cell_noise_) {
            gridmap_->GenerateNoiseMap();
            this->needs_init_cell_noise_ = false;
        }
        DrawCells();
        DrawGroundstation(gridmap_->GetGroundstation());
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
    // Start drawing cells from the cell at the camera position
    for (const auto& cell : gridmap_->GetChangedCells()) {
        SDL_Rect cell_rect = DrawCell(cell.x_, cell.y_);
        if (cell_rect.x < 0 || cell_rect.y < 0 ||
            cell_rect.x >= pixel_buffer_->GetWidth() ||
            cell_rect.y >= pixel_buffer_->GetHeight()) {
            continue;
        }

        if (cell_rect.x + cell_rect.w > pixel_buffer_->GetWidth()) {
            cell_rect.w = pixel_buffer_->GetWidth() - cell_rect.x;
        }
        if (cell_rect.y + cell_rect.h > pixel_buffer_->GetHeight()) {
            cell_rect.h = pixel_buffer_->GetHeight() - cell_rect.y;
        }
        // Get a pointer to the pixel data for the cell
        Uint32* cellPixelData = &pixel_buffer_->GetData()[cell_rect.y * pixel_buffer_->GetWidth() + cell_rect.x];

        // Update the portion of the texture that corresponds to the cell
        SDL_UpdateTexture(texture_, &cell_rect, cellPixelData, pixel_buffer_->GetPitch());
    }
    gridmap_->ResetChangedCells();
}

Uint32 ModifyColorWithGradient(Uint32 base_color, int x, int y) {
    Uint8 r = (base_color >> 24) & 0xFF;
    Uint8 g = (base_color >> 16) & 0xFF;
    Uint8 b = (base_color >> 8) & 0xFF;
    Uint8 a = base_color & 0xFF;

    r = std::min(255, r + (x % 32));
    g = std::min(255, g + (y % 32));
    b = std::min(255, b + ((x + y) % 32));

    // Introduce a subtle flicker based on the elapsed time to animate burning cells.
    double time = SDL_GetTicks() / 1000.0; // Convert milliseconds to seconds
    double flicker = (std::sin(time * 5.0 + x + y) + 1.0) * 0.9; // 0..1 range
    auto offset = static_cast<Uint8>(flicker * 64); // Scale to a small color offset

    r = std::min(255, static_cast<int>(r) + offset);
    g = std::min(255, static_cast<int>(g) + offset);
    b = std::min(255, static_cast<int>(b) + offset);

    return (r << 24) | (g << 16) | (b << 8) | a;
}

SDL_Rect FireModelRenderer::DrawCell(int x, int y) {
    auto [screen_x, screen_y] = camera_.GridToScreenPosition(floor(x), floor(y));
    const SDL_Rect cell_rect = {
            screen_x,
            screen_y,
            static_cast<int>(camera_.GetCellSize()),
            static_cast<int>(camera_.GetCellSize())
    };
    Uint32 base_color = gridmap_->At(x, y).GetMappedColor();
    if(parameters_.lingering_){
        if (gridmap_->At(x, y).WasFlooded())
            base_color = (255 << 24) | (77 << 16) | (187 << 8) | 230;
    }
    int grid_offset = !parameters_.render_grid_ ? 0 : (camera_.GetCellSize() >= 3.0 ? -1 : 0);
    if (gridmap_->At(x,y).HasNoise() && gridmap_->HasNoiseGenerated() && parameters_.has_noise_ && !this->needs_init_cell_noise_) {
        std::vector<std::vector<int>>& noise_map = gridmap_->At(x, y).GetNoiseMap();
        if (!noise_map.empty()){
            std::vector<std::vector<int>> scaled_noise_map = ScaleNoiseMap(noise_map, camera_.GetCellSize());
            CellState state = gridmap_->At(x, y).GetCellState();
            Uint32 gradient = base_color;
            if (state == CellState::GENERIC_BURNING || state == CellState::WATER)
                gradient = ModifyColorWithGradient(base_color, x, y);
            pixel_buffer_->Draw(cell_rect, gradient, scaled_noise_map, grid_offset);
        } else {
            this->needs_full_redraw_ = true;
        }
    }
    else {
        pixel_buffer_->Draw(cell_rect, base_color, grid_offset);
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
            if (intensity <= 0.3) {
                // Dark red to light red gradient based on intensity (0 -> dark red, 1 -> light red)
                auto rel_intense = (intensity / 0.3);
                r = 128 + static_cast<Uint8>(127  * rel_intense);
                g = static_cast<Uint8>(100.0 * rel_intense);
                b = 0;
            } else if (intensity <= 0.7) {
                // Light red to yellow gradient based on intensity (0.3 -> light red, 0.6 -> yellow)
                auto rel_intense = (intensity / 0.7);
                r = 255;
                g = 128 + static_cast<Uint8>(127 * rel_intense);
                b = 0;
            } else {
                // Yellow to white gradient based on intensity (0.6 -> yellow, 1 -> white)
                r = 255;
                g = 255;
                b = static_cast<Uint8>(128 * intensity);
            }
            color = SDL_Color{r, g, b, static_cast<Uint8>(255.0 * (0.6 + 0.4 * intensity))};
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

    for(int w = 0; w < radius * 2; w++) {
        for(int h = 0; h < radius * 2; h++) {
            int dx = radius - w; // horizontal offset
            int dy = radius - h; // vertical offset
            if((dx*dx + dy*dy) <= (radius * radius)) {
                SDL_RenderDrawPoint(renderer_, x + dx, y + dy);
            }
        }
    }
}

void FireModelRenderer::DrawParticles() {

    int circle_radius = static_cast<int>(camera_.GetCellSize() / 6);

    if (circle_radius > 0) {
        const std::vector<RadiationParticle>& particles = gridmap_->GetRadiationParticles();

        if (!particles.empty()) {
            for (const auto& particle : particles) {
                double x, y;
                particle.GetPosition(x, y);

                x = x / parameters_.GetCellSize();
                y = y / parameters_.GetCellSize();
                auto [posx, posy] = camera_.GridToScreenPosition(x, y);

                DrawCircle(posx, posy, circle_radius, particle.GetIntensity());  // Scaling with zoom
            }
        }

        const std::vector<VirtualParticle>& virtual_particles = gridmap_->GetVirtualParticles();

        if (!virtual_particles.empty()) {
            for (const auto& particle : virtual_particles) {
                double x, y;
                particle.GetPosition(x, y);
                x = x / parameters_.GetCellSize();
                y = y / parameters_.GetCellSize();
                auto [posx, posy] = camera_.GridToScreenPosition(x, y);

                DrawCircle(posx, posy, circle_radius, particle.GetIntensity());  // Scaling with zoom
            }
        }
    }

}

void FireModelRenderer::DrawEpisodeEnd(bool success) {
    auto width = width_ - 300;
    auto height = 0 + 100;

    if (success) {
        SDL_Rect destRect = {width, height, 200, 200}; // x, y, width and height of the arrow
        SDL_RenderCopyEx(renderer_, episode_succeeded_texture_, nullptr, &destRect, 0, nullptr, SDL_FLIP_NONE);
    } else {
        SDL_Rect destRect = {width, height, 200, 200}; // x, y, width and height of the arrow
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

        // Optional , extra smoot
//        DrawCircle(x1, y1, base_thickness, intensity);
//        DrawCircle(x2, y2, base_thickness, intensity);
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

