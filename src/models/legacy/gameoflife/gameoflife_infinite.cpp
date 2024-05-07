//
// Created by nex on 07.06.23.
//

#include "gameoflife_infinite.h"

std::shared_ptr<GameOfLifeInfinite> GameOfLifeInfinite::instance_ = nullptr;

GameOfLifeInfinite::GameOfLifeInfinite(std::shared_ptr<SDL_Renderer> renderer) {
    model_renderer_ = GameOfLifeInfiniteRenderer::GetInstance(renderer);
    Initialize();
}

void GameOfLifeInfinite::Initialize() {
    // Initialize the entire grid with dead cells
    for (int x = 0; x < 100; ++x) {
        for (int y = 0; y < 100; ++y) {
            state_[{x, y}] = false;
        }
    }
}

void GameOfLifeInfinite::Update() {
    CellStateGOF new_state;

    // Find all cells that need to be updated
    for (const auto& [cell, is_alive] : state_) {
        if (is_alive) {
            Neighbors neighbors = GetNeighbors(cell);
            for (const auto& neighbor : neighbors) {
                int live_neighbors = CountLiveNeighbors(neighbor);
                bool current_state = state_.count(neighbor) && state_[neighbor];

                // Copy state for cells that will survive or be born
                if ((current_state && (live_neighbors == 2 || live_neighbors == 3)) || (!current_state && live_neighbors == 3)) {
                    new_state[neighbor] = true;
                }
            }
        }
    }

    state_ = std::move(new_state);

}

std::tuple<std::vector<std::deque<std::shared_ptr<State>>>, std::vector<double>, std::vector<bool>, std::pair<bool, bool>> GameOfLifeInfinite::Step(std::vector<std::shared_ptr<Action>> actions) {
    return {};
}

void GameOfLifeInfinite::Reset() {
    state_.clear();
}

void GameOfLifeInfinite::Config() {
    if (ImGui::Button("Randomize Cells"))
        RandomizeCells();
    if (ImGui::Button("Reset Cells"))
        Reset();
    ImGui::SliderInt("Cell Size", &cell_size_, 2, 100);

    ImGui::ColorEdit3("Background Colour", (float*)&model_renderer_->background_color_); // Edit 3 floats representing a color
}

void GameOfLifeInfinite::HandleEvents(SDL_Event event, ImGuiIO* io) {
    if (event.type == SDL_MOUSEBUTTONDOWN && !io->WantCaptureMouse) {
        int x, y;
        SDL_GetMouseState(&x, &y);
        state_[{x / cell_size_, y / cell_size_}] = !state_[{x / cell_size_, y / cell_size_}];
    }
}

void GameOfLifeInfinite::Render() {
    model_renderer_->Render(state_, cell_size_);
}

void GameOfLifeInfinite::RandomizeCells() {
    // Get the current time as an integer
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

    // Create a random number generator seeded with the current time
    std::default_random_engine generator(seed);
    std::uniform_int_distribution<int> distribution(0,1);

    for (int i = 0; i < width_ / cell_size_; ++i) {
        for (int j = 0; j < height_ / cell_size_; ++j)
            state_[{i, j}] = distribution(generator);
    }
}

Neighbors GameOfLifeInfinite::GetNeighbors(const Cell &cell) const {
    Neighbors neighbors;
    for (int x = -1; x <= 1; ++x) {
        for (int y = -1; y <= 1; ++y) {
            if (x != 0 || y != 0)
                neighbors.push_back({cell.first + x, cell.second + y});
        }
    }
    return neighbors;
}

int GameOfLifeInfinite::CountLiveNeighbors(const Cell &cell) const {
    int live_neighbors = 0;
    for (const auto& neighbor : GetNeighbors(cell)) {
        if (state_.find(neighbor) != state_.end() && state_.at(neighbor))
            ++live_neighbors;
    }
    return live_neighbors;
}

void GameOfLifeInfinite::SetWidthHeight(int width, int height) {
    width_ = width;
    height_ = height;
}

void GameOfLifeInfinite::ShowPopups() {

}

void GameOfLifeInfinite::ImGuiSimulationSpeed() {

}

void GameOfLifeInfinite::ImGuiModelMenu() {

}

void GameOfLifeInfinite::ShowControls(std::function<void(bool&, bool&, int&)> controls, bool &update_simulation, bool &render_simulation, int &delay) {
    controls(update_simulation, render_simulation, delay);
}

std::vector<std::deque<std::shared_ptr<State>>> GameOfLifeInfinite::GetObservations() {
    return std::vector<std::deque<std::shared_ptr<State>>>();
}
