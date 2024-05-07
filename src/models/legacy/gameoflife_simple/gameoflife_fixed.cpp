//
// Created by nex on 07.06.23.
//

#include "gameoflife_fixed.h"

std::shared_ptr<GameOfLifeFixed> GameOfLifeFixed::instance_ = nullptr;

GameOfLifeFixed::GameOfLifeFixed(std::shared_ptr<SDL_Renderer> renderer) {
    model_renderer_ = GameOfLifeFixedRenderer::GetInstance(renderer);
    SDL_GetRendererOutputSize(renderer.get(), &width_, &height_);
    Initialize();
}

void GameOfLifeFixed::Initialize() {
    rows_ = std::ceil(static_cast<float>(width_) / cell_size_);
    cols_ = std::ceil(static_cast<float>(height_) / cell_size_);
    state_[0] = std::vector<std::vector<bool>>(rows_, std::vector<bool>(cols_, false));
    state_[1] = std::vector<std::vector<bool>>(rows_, std::vector<bool>(cols_, false));
    current_state_ = 0;
}

void GameOfLifeFixed::Update() {
    std::vector<std::vector<bool>>& current_state = state_[current_state_];
    std::vector<std::vector<bool>>& new_state = state_[1 - current_state_];

    for (int i = 0; i < rows_; ++i) {
        for (int j = 0; j < cols_; ++j) {
            int live_neighbors = 0;
            for (int x = -1; x <= 1; ++x) {
                for (int y = -1; y <= 1; ++y) {
                    if (x == 0 && y == 0) continue;

                    int nx = i + x;
                    int ny = j + y;

                    if (nx >= 0 && nx < rows_ && ny >= 0 && ny < cols_ && current_state[nx][ny]) {
                        ++live_neighbors;
                    }
                }
            }

            if (current_state[i][j] && (live_neighbors < 2 || live_neighbors > 3)) {
                new_state[i][j] = false;
            } else if (!current_state[i][j] && live_neighbors == 3) {
                new_state[i][j] = true;
            } else {
                new_state[i][j] = current_state[i][j];
            }
        }
    }

    current_state_ = 1 - current_state_;
}

std::tuple<std::vector<std::deque<std::shared_ptr<State>>>, std::vector<double>, std::vector<bool>, std::pair<bool, bool>> GameOfLifeFixed::Step(std::vector<std::shared_ptr<Action>> actions) {

    return {};
}

void GameOfLifeFixed::Reset() {
    state_[0] = std::vector<std::vector<bool>>(rows_, std::vector<bool>(cols_, false));
    state_[1] = std::vector<std::vector<bool>>(rows_, std::vector<bool>(cols_, false));
    current_state_ = 0;
}

void GameOfLifeFixed::Config() {
    if (ImGui::Button("Randomize Cells"))
        RandomizeCells();
    if (ImGui::Button("Reset Cells"))
        Reset();
    if (ImGui::Button("Init new Grid"))
        Initialize();
    ImGui::SliderInt("Cell Size", &cell_size_, 2, 100);

    ImGui::ColorEdit3("Background Colour", (float*)&model_renderer_->background_color_); // Edit 3 floats representing a color
}

void GameOfLifeFixed::HandleEvents(SDL_Event event, ImGuiIO* io) {
    if (event.type == SDL_MOUSEBUTTONDOWN && !io->WantCaptureMouse) {
        int x, y;
        SDL_GetMouseState(&x, &y);
        if ( (x / cell_size_) < rows_ && (y / cell_size_) < cols_)
            state_[current_state_][x / cell_size_][ y / cell_size_] = !state_[current_state_][x / cell_size_][ y / cell_size_];
    }
}

void GameOfLifeFixed::Render() {
    model_renderer_->Render(state_[current_state_], cell_size_, rows_, cols_);
}

void GameOfLifeFixed::RandomizeCells() {
    // Get the current time as an integer
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

    // Create a random number generator seeded with the current time
    std::default_random_engine generator(seed);
    std::uniform_int_distribution<int> distribution(0,1);

    for (int i = 0; i < width_ / cell_size_; ++i) {
        for (int j = 0; j < height_ / cell_size_; ++j)
            state_[current_state_][i][j] = distribution(generator);
    }
}

void GameOfLifeFixed::SetWidthHeight(int width, int height) {
    width_ = width;
    height_ = height;
}

void GameOfLifeFixed::ShowPopups() {

}

void GameOfLifeFixed::ImGuiSimulationSpeed() {

}

void GameOfLifeFixed::ImGuiModelMenu() {

}

void GameOfLifeFixed::ShowControls(std::function<void(bool &, bool &, int &)> controls, bool &update_simulation, bool &render_simulation, int &delay) {
    controls(update_simulation, render_simulation, delay);
}

std::vector<std::deque<std::shared_ptr<State>>> GameOfLifeFixed::GetObservations() {
    return std::vector<std::deque<std::shared_ptr<State>>>();
}
