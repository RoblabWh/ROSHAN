//
// Created by nex on 15.07.23.
//

#ifndef ROSHAN_AGENT_STATE_H
#define ROSHAN_AGENT_STATE_H

#include <utility>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include "state.h"

struct AgentState : public State {
    std::pair<double, double> velocity{};
    int norm_scale{};
    std::pair<double, double> max_speed{};
    int id{};
    double cell_size{};
    std::pair<double, double> position{};
    std::pair<double, double> goal_position{};
    std::shared_ptr<std::vector<std::vector<double>>> distances_to_other_agents;
    std::shared_ptr<std::vector<bool>> distances_mask;
    std::shared_ptr<std::vector<std::pair<double, double>>> drone_positions =
        std::make_shared<std::vector<std::pair<double, double>>>();
    std::shared_ptr<std::vector<std::pair<double, double>>> fire_positions =
        std::make_shared<std::vector<std::pair<double, double>>>();
    std::shared_ptr<std::vector<std::pair<double, double>>> goal_positions;
    double fire_count{0.0};                             // num_fires / (rows*cols)
    std::pair<double, double> fire_centroid{0.0, 0.0};  // normalized to same space as fire_positions
};

// Computed feature helpers operating on AgentState
namespace state_features {

inline std::pair<double, double> GridPositionDouble(const AgentState& s) {
    return {s.position.first / s.cell_size, s.position.second / s.cell_size};
}

inline std::pair<double, double> GridPositionDoubleNorm(const AgentState& s) {
    auto gp = GridPositionDouble(s);
    return {(2.0 * gp.first / s.norm_scale) - 1.0, (2.0 * gp.second / s.norm_scale) - 1.0};
}

inline std::pair<double, double> PositionNormAroundCenter(const AgentState& s) {
    return {(2.0 * s.position.first / (s.norm_scale * s.cell_size)) - 1.0,
            (2.0 * s.position.second / (s.norm_scale * s.cell_size)) - 1.0};
}

inline std::pair<double, double> GoalPositionNorm(const AgentState& s) {
    return {(2.0 * s.goal_position.first / s.norm_scale) - 1.0,
            (2.0 * s.goal_position.second / s.norm_scale) - 1.0};
}

inline std::pair<double, double> VelocityNorm(const AgentState& s) {
    return {
        std::clamp(s.velocity.first  / s.max_speed.first / s.norm_scale,  -1.0, 1.0),
        std::clamp(s.velocity.second / s.max_speed.second / s.norm_scale, -1.0, 1.0)
    };
}

inline double Speed(const AgentState& s) {
    return std::min(std::hypot(s.velocity.first / s.max_speed.first,
                               s.velocity.second / s.max_speed.second), 1.0);
}

inline std::pair<double, double> DeltaGoal(const AgentState& s) {
    auto pos = GridPositionDouble(s);
    double dx = std::clamp((s.goal_position.first - pos.first) / s.norm_scale, -1.0, 1.0);
    double dy = std::clamp((s.goal_position.second - pos.second) / s.norm_scale, -1.0, 1.0);
    return {dx, dy};
}

inline double DistanceToGoal(const AgentState& s) {
    auto pos = GridPositionDouble(s);
    double dx = (s.goal_position.first - pos.first) / s.norm_scale;
    double dy = (s.goal_position.second - pos.second) / s.norm_scale;
    return std::min(std::hypot(dx, dy), 1.0);
}

inline std::pair<double, double> CosSinToGoal(const AgentState& s) {
    auto pos = GridPositionDouble(s);
    double gx = s.goal_position.first - pos.first;
    double gy = s.goal_position.second - pos.second;
    double gnorm = std::hypot(gx, gy);

    double vx = s.velocity.first;
    double vy = s.velocity.second;
    double vnorm = std::hypot(vx, vy);

    if (gnorm < 1e-8) return {1.0, 0.0};
    gx /= gnorm; gy /= gnorm;

    if (vnorm < 1e-8) return {0.0, 0.0};
    vx /= vnorm; vy /= vnorm;

    const double cos_th = vx*gx + vy*gy;
    const double sin_th = vx*(-gy) + vy*(gx);
    return {std::clamp(cos_th, -1.0, 1.0), std::clamp(sin_th, -1.0, 1.0)};
}

inline std::pair<double, double> OrientationToGoal(const AgentState& s) {
    auto pos = GridPositionDouble(s);
    double x = s.goal_position.first - pos.first;
    double y = s.goal_position.second - pos.second;
    double magnitude = std::sqrt(x * x + y * y);
    if (magnitude < std::numeric_limits<double>::epsilon()) {
        return {0.0, 0.0};
    }
    return {x / magnitude, y / magnitude};
}

inline double DistanceToNearestBoundaryNorm(const AgentState& s) {
    const auto& first = (*s.distances_to_other_agents)[0];
    return std::sqrt(first[0] * first[0] + first[1] * first[1]);
}

} // namespace state_features

#endif //ROSHAN_AGENT_STATE_H
