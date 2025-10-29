//
// Created by nex on 15.07.23.
//

#include "agent_state.h"

std::vector<std::vector<std::vector<double>>> AgentState::GetDroneViewNorm() const {

    std::vector<std::vector<std::vector<double>>> drone_view_norm(2,
                                                                  std::vector<std::vector<double>>((*drone_view_)[0].size(),
                                                            std::vector<double>((*drone_view_)[0][0].size())));

    for (size_t i = 0; i < (*drone_view_)[0].size(); ++i) {
        for (size_t j = 0; j < (*drone_view_)[0][i].size(); ++j) {
            drone_view_norm[0][i][j] = static_cast<double>((*drone_view_)[0][i][j]) /
                                       static_cast<double>(static_cast<int>(CELL_STATE_COUNT) - 1);
            drone_view_norm[1][i][j] = static_cast<double>((*drone_view_)[1][i][j]);
        }
    }

    return drone_view_norm;
}

int AgentState::CountOutsideArea() const {
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
    double x = (2.0 * position_.first / (norm_scale_ * cell_size_)) - 1;
    double y = (2.0 * position_.second / (norm_scale_ * cell_size_)) - 1;
    return std::make_pair(x, y);
}

std::pair<double, double> AgentState::GetGoalPositionNorm() const {
    double x = goal_position_.first / norm_scale_;
    double y = goal_position_.second / norm_scale_;
    return std::make_pair(x, y);
}

std::pair<double, double> AgentState::GetDeltaGoal() const {
    auto position = GetGridPositionDouble();
    double dx = (goal_position_.first - position.first) / norm_scale_;
    double dy = (goal_position_.second - position.second) / norm_scale_;
    dx = std::clamp(dx, -1.0, 1.0);
    dy = std::clamp(dy, -1.0, 1.0);
    return std::make_pair(dx, dy);
}

std::pair<double, double> AgentState::GetCosSinToGoal() const {
    // goal direction ĝ
    auto position = GetGridPositionDouble();
    double gx = goal_position_.first - position.first;
    double gy = goal_position_.second - position.second;
    double gnorm = std::hypot(gx, gy);

    // velocity direction v̂
    double vx = velocity_.first;
    double vy = velocity_.second;
    double vnorm = std::hypot(vx, vy);

    if (gnorm < 1e-8) return {1.0, 0.0}; // at goal => aligned

    gx /= gnorm; gy /= gnorm;


    if (vnorm < 1e-8) return {0.0, 0.0}; // no movement => undefined; use neutral

    vx /= vnorm; vy /= vnorm;

    // cos and sin of the signed angle from ĝ to v̂
    const double cos_th = vx*gx + vy*gy;
    const double sin_th = vx*(-gy) + vy*(gx); // dot with perp(ĝ) = (-gy, gx)
    return { std::clamp(cos_th, -1.0, 1.0), std::clamp(sin_th, -1.0, 1.0) };
}

std::pair<double, double> AgentState::GetVelocityNorm() const {
    return {
        std::clamp(velocity_.first  / max_speed_.first / norm_scale_,  -1.0, 1.0),
        std::clamp(velocity_.second / max_speed_.second / norm_scale_, -1.0, 1.0)
    };
}

double AgentState::GetSpeed() const {
//    return std::min(std::hypot(velocity_.first, velocity_.second), 1.0);
    return std::min(std::hypot(velocity_.first / max_speed_.first,velocity_.second / max_speed_.second), 1.0);
}

double AgentState::GetDistanceToGoal() const {
    auto position = GetGridPositionDouble();
    double dx = (goal_position_.first - position.first) / norm_scale_;
    double dy = (goal_position_.second - position.second) / norm_scale_;
    return std::min(std::hypot(dx, dy), 1.0);
}

std::pair<double, double> AgentState::GetOrientationToGoal() const {
    auto position = GetGridPositionDouble();
    double x = goal_position_.first - position.first;
    double y = goal_position_.second - position.second;
    double magnitude = sqrt(x * x + y * y);
    // Check if we are at the goal position
    if (magnitude < std::numeric_limits<double>::epsilon()) {
        return std::make_pair(0.0, 0.0);
    }
    return std::make_pair(x / magnitude, y / magnitude);
}

std::pair<double, double> AgentState::GetGridPositionDoubleNorm() const {
    auto grid_position = GetGridPositionDouble();
    double x = (2 * grid_position.first / norm_scale_) - 1;
    double y = (2 * grid_position.second / norm_scale_) - 1;
    return std::make_pair(x, y);
}

// TODO: turn on debugging in GUI (its out commented)
// Function for Debugging and potential use in training(later)
std::pair<double, double> AgentState::GetPositionInExplorationMap() const {
    auto grid_double_norm = GetGridPositionDoubleNorm();
    // Get Dim from exploration map
    auto dimension = exploration_map_->size();
    double x = grid_double_norm.first * static_cast<double>(dimension);
    double y = grid_double_norm.second * static_cast<double>(dimension);
    return std::make_pair(x, y);
}

std::pair<double, double> AgentState::GetGridPositionDouble() const {
    double x = position_.first / cell_size_;
    double y = position_.second / cell_size_;
    return std::make_pair(x, y);
}

double AgentState::GetDistanceToNearestBoundaryNorm() const {
    auto first_distance_point = this->GetDistancesToOtherAgents()[0];
    return std::sqrt(first_distance_point[0] * first_distance_point[0] +
                     first_distance_point[1] * first_distance_point[1]);
}

std::vector<std::vector<double>> AgentState::GetExplorationMapNorm() const {
    // "Normalizing" is probably not the right term here, but it is used to scale the values
    std::vector exploration_map_norm(exploration_map_->size(), std::vector<double>((*exploration_map_)[0].size()));
    for (size_t i = 0; i < exploration_map_->size(); ++i) {
        for (size_t j = 0; j < (*exploration_map_)[i].size(); ++j) {
            exploration_map_norm[i][j] = static_cast<double>((*exploration_map_)[i][j]) * 255;
        }
    }
    return exploration_map_norm;
}

double AgentState::GetExplorationMapScalar() const {
    double scalar = 0;
    for (const auto & row : *exploration_map_) {
        for (int value : row) {
            scalar += static_cast<double>(value);
        }
    }
    // divide scalar by max_value
    return scalar / static_cast<double>(exploration_map_->size()) * static_cast<double>((*exploration_map_)[0].size());
}

std::shared_ptr<std::vector<std::pair<double, double>>> AgentState::GetFirePositionsFromFireMap() const {
    std::vector<std::pair<double, double>> fire_positions;
    // Append Wait Token for the Network at (-1, -1)
    fire_positions.emplace_back(-1.0, -1.0);
        for (size_t i = 0; i < fire_map_->size(); ++i) {
            for (size_t j = 0; j < (*fire_map_)[i].size(); ++j) {
                if ((*fire_map_)[i][j] > 0) {
                    fire_positions.emplace_back(static_cast<double>(i) + 0.5, static_cast<double>(j) + 0.5);
                }
            }
        }
    return std::make_shared<std::vector<std::pair<double, double>>>(fire_positions);
}


