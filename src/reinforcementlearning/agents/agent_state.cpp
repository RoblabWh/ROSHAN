//
// Created by nex on 15.07.23.
//

#include "agent_state.h"

std::vector<std::vector<std::vector<int>>> AgentState::GetDroneViewNorm() {

    std::vector<std::vector<std::vector<int>>> drone_view_norm(2,
                                                               std::vector<std::vector<int>>((*drone_view_)[0].size(),
                                                               std::vector<int>((*drone_view_)[0][0].size())));

    for (size_t i = 0; i < (*drone_view_)[0].size(); ++i) {
        for (size_t j = 0; j < (*drone_view_)[0][i].size(); ++j) {
            drone_view_norm[0][i][j] = static_cast<int>((*drone_view_)[0][i][j]) / int(static_cast<CellState>(CELL_STATE_COUNT - 1));
            drone_view_norm[1][i][j] = (*drone_view_)[1][i][j];
        }
    }

    return drone_view_norm;
}

int AgentState::CountOutsideArea() {
    int outside_area = 0;
    for (auto & i : (*drone_view_)[0]) {
        for (int j : i) {
            if (j == OUTSIDE_GRID) {
                outside_area++;
            }
        }
    }
    return outside_area;
}

std::pair<double, double> AgentState::GetPositionNormAroundCenter() const {
    double x = (2.0 * position_.first / (map_dimensions_.first * cell_size_)) - 1;
    double y = (2.0 * position_.second / (map_dimensions_.second * cell_size_)) - 1;
    return std::make_pair(x, y);
}

std::pair<double, double> AgentState::GetGoalPositionNorm() const {
    double x = goal_position_.first / map_dimensions_.first;
    double y = goal_position_.second / map_dimensions_.second;
    return std::make_pair(x, y);
}

std::pair<double, double> AgentState::GetDeltaGoal() const {
    auto position = GetGridPositionDouble();
    auto largest_side = std::max(map_dimensions_.first, map_dimensions_.second);
//    auto diagonal = sqrt(map_dimensions_.first * map_dimensions_.first + map_dimensions_.second * map_dimensions_.second);
    double x = ((position.first - goal_position_.first) / largest_side);
    double y = ((position.second - goal_position_.second) / largest_side);
    return std::make_pair(x, y);
}

[[nodiscard]] std::pair<double, double> AgentState::GetVelocityNorm() const {
    auto largest_side = std::max(map_dimensions_.first, map_dimensions_.second);
    // Multiply by 100 to make the feature more significant for the neural network
    return std::make_pair((velocity_.first / (largest_side * cell_size_)), (velocity_.second / (largest_side * cell_size_)));
    //{ return std::make_pair(velocity_.first / max_speed_.first, velocity_.second / max_speed_.second); }
}

std::pair<double, double> AgentState::GetOrientationToGoal() const {
    double x = goal_position_.first - position_.first;
    double y = goal_position_.second - position_.second;
    double magnitude = sqrt(x * x + y * y);
    return std::make_pair(x / magnitude, y / magnitude);
}

std::pair<double, double> AgentState::GetGridPositionDoubleNorm() const {
    auto grid_position = GetGridPositionDouble();
    double x = (2 * grid_position.first / map_dimensions_.first) - 1;
    double y = (2 * grid_position.second / map_dimensions_.second) - 1;
    return std::make_pair(x, y);
}

std::pair<double, double> AgentState::GetPositionInExplorationMap() const {
    auto grid_double_norm = GetGridPositionDoubleNorm();
    // Get Dim from exploration map
    auto dimension = (*exploration_map_).size();
    double x = grid_double_norm.first * dimension;
    double y = grid_double_norm.second * dimension;
    return std::make_pair(x, y);
}

std::pair<double, double> AgentState::GetGridPositionDouble() const {
    double x, y;
    x = position_.first / cell_size_;
    y = position_.second / cell_size_;
    return std::make_pair(x, y);
}

double AgentState::GetDistanceToNearestBoundaryNorm() const {
    auto largest_side = std::max(map_dimensions_.first, map_dimensions_.second);
    auto position = GetGridPositionDouble();
    double x = position.first;
    double y = position.second;

    if (x < 0) {
        x = std::abs(x);
    }
    if (y < 0) {
        y = std::abs(y);
    }
    auto distance_x = map_dimensions_.first - x;
    auto distance_y = map_dimensions_.second - y;
    auto distance = std::min({x, y, distance_x, distance_y});

    return distance / largest_side;
}

std::vector<std::vector<double>> AgentState::GetExplorationMapNorm() const {
    std::vector<std::vector<double>> exploration_map_norm((*exploration_map_).size(), std::vector<double>((*exploration_map_)[0].size()));
    double max_value = 255;//map_dimensions_.first * map_dimensions_.second;
    for (size_t i = 0; i < (*exploration_map_).size(); ++i) {
        for (size_t j = 0; j < (*exploration_map_)[i].size(); ++j) {
            exploration_map_norm[i][j] = static_cast<double>((*exploration_map_)[i][j]) * max_value;
        }
    }
    return exploration_map_norm;
}

double AgentState::GetExplorationMapScalar() const {
    double scalar = 0;
    double max_value = static_cast<double>((*exploration_map_).size()) * static_cast<double>((*exploration_map_)[0].size());
    for (const auto & row : *exploration_map_) {
        for (int value : row) {
            scalar += static_cast<double>(value);
        }
    }
    return scalar / max_value;
}

std::shared_ptr<std::vector<std::pair<double, double>>> AgentState::GetFirePositionsFromFireMap() {
    std::vector<std::pair<double, double>> fire_positions;
    // Append Wait Token for the Network at (-1, -1)
    fire_positions.emplace_back(-1, -1);
        for (size_t i = 0; i < (*fire_map_).size(); ++i) {
            for (size_t j = 0; j < (*fire_map_)[i].size(); ++j) {
                if ((*fire_map_)[i][j] > 0) {
                    fire_positions.emplace_back(i, j);
                }
            }
        }
    return std::make_shared<std::vector<std::pair<double, double>>>(fire_positions);
}


