//
// Created by nex on 04.06.24.
//

#ifndef ROSHAN_REINFORCEMENTLEARNING_HANDLER_H
#define ROSHAN_REINFORCEMENTLEARNING_HANDLER_H

#include <shared_mutex>
#include <array>
#include <deque>
#include <utility>
#include <vector>
#include <unordered_map>
#include "externals/pybind11/include/pybind11/embed.h"
#include "externals/pybind11/include/pybind11/stl.h"
#include "externals/pybind11/include/pybind11/complex.h"
#include "state.h"
#include "firespin/rendering/firemodel_renderer.h"
#include "agents/agent_factory.h"
#include "src/reinforcementlearning/agents/agent_state.h"
#include "reinforcementlearning/actions/fly_action.h"
#include "src/utils.h"
#include "src/reinforcementlearning/start_goal_strategy.h"
#include "src/reinforcementlearning/feature_schema.h"
#include "src/reinforcementlearning/feature_definitions.h"

namespace py = pybind11;

// Squared distance (no sqrt)
inline double dist2(const std::pair<double,double>& a,
                    const std::pair<double,double>& b) {
    const double dx = a.first  - b.first;
    const double dy = a.second - b.second;
    return dx*dx + dy*dy;
}

// True collision (discs touching/overlapping)
inline bool circlesCollide(const double d1, double r1, double r2) {
    const double R = r1 + r2;
    return d1 <= R*R;
}

// Returns pairs of indices that are actually colliding
inline void findCollisions(const std::vector<std::shared_ptr<FlyAgent>>& agents, const std::shared_ptr<GridMap>& gridmap, const int view_range_ = -1, const bool init = false) {
    std::vector<std::pair<int,int>> hits;
    const int n = static_cast<int>(agents.size());
    auto view_range = view_range_ == -1 ? agents[0]->GetViewRange() : view_range_;
    auto view_range_h = view_range / 2.0;
    // first loop through all agents and clear their distance records
    for (auto & agent : agents) {
        agent->ClearDistances();
        agent->ClearMask();
        std::vector<double> bvec_norm;
        auto mask = agent->GetDistanceToNearestBoundaryNorm(gridmap->GetRows(), gridmap->GetCols(), view_range, bvec_norm);
        agent->AppendDistance(bvec_norm);  // e.g., store alongside neighbor features
        agent->AppendMask(mask);
    }
    // Cache agent positions to avoid repeated virtual calls
    std::vector<std::pair<double,double>> positions(n);
    for (int i = 0; i < n; ++i) {
        positions[i] = agents[i]->GetGridPositionDouble();
    }

    for (int i = 0; i < n; ++i) {
        for (int j = i+1; j < n; ++j) {
            const double dx = positions[j].first  - positions[i].first;
            const double dy = positions[j].second - positions[i].second;
            if (std::fabs(dx) < view_range_h && std::fabs(dy) < view_range_h) {
                std::pair<double, double> a_j_dvx;
                std::pair<double, double> a_i_dvx;
                // If this is a reset or init we don't have set the speed to normalize yet (BAD but works for now)
                if (!init) {
                    a_j_dvx = agents[j]->GetVelocityVecNorm();
                    a_i_dvx = agents[i]->GetVelocityVecNorm();
                } else {
                    a_j_dvx = {0.0, 0.0};
                    a_i_dvx = {0.0, 0.0};
                }

                agents[i]->AppendDistance({dx / view_range, dy / view_range, a_j_dvx.first, a_j_dvx.second});
                agents[j]->AppendDistance({-dx / view_range, -dy / view_range, a_i_dvx.first, a_i_dvx.second});
                agents[i]->AppendMask(true);
                agents[j]->AppendMask(true);
                // Reuse already-computed dx/dy for collision check instead of calling dist2 again
                const double d_sq = dx * dx + dy * dy;
                if (circlesCollide(d_sq, agents[i]->GetDroneSize(), agents[j]->GetDroneSize())) {
                    hits.emplace_back(i, j);
                }
            } else {
                agents[i]->AppendDistance({0.0, 0.0, 0.0, 0.0});
                agents[j]->AppendDistance({0.0, 0.0, 0.0, 0.0});
                agents[i]->AppendMask(false);
                agents[j]->AppendMask(false);
            }
        }
    }
    for (auto collision : hits) {
        agents[collision.first]->SetCollision(true);
        agents[collision.second]->SetCollision(true);
    }
}

class __attribute__((visibility("default"))) ReinforcementLearningHandler {

public:
    explicit ReinforcementLearningHandler(FireModelParameters &parameters);
    //only one instance of this class can be created

    static std::shared_ptr<ReinforcementLearningHandler> Create(FireModelParameters &parameters) {
        return std::make_shared<ReinforcementLearningHandler>(parameters);
    }

    ~ReinforcementLearningHandler(){
        if (rl_status_) {
            rl_status_.attr("clear")();
        }
    }

    // Schema-driven generic batch observation API: returns py::dict keyed by group name
    py::dict GetBatchedObservations(const std::string& agent_type);
    void StepDroneManual(int drone_idx, double speed_x, double speed_y, int water_dispense);
    void ResetEnvironment(Mode mode);
    StepResult Step(const std::string& agent_type, std::vector<std::shared_ptr<Action>> actions);
    void SetModelRenderer(std::shared_ptr<FireModelRenderer> model_renderer) { model_renderer_ = std::move(model_renderer); }
    void SetGridMap(std::shared_ptr<GridMap> gridmap) { gridmap_ = std::move(gridmap); }
    void SetRLStatus(py::dict status);
    void UpdateReward();
    bool AgentIsRunning() const { return rl_status_["agent_is_running"].cast<bool>(); }
    py::dict GetRLStatus() { return rl_status_; }
    std::string GetRLMode() const { return rl_status_["rl_mode"].cast<std::string>(); }
    std::shared_ptr<std::vector<std::shared_ptr<FlyAgent>>> GetDrones() {
        auto fly_agents = std::make_shared<std::vector<std::shared_ptr<FlyAgent>>>();

        // Use CastAgents helper to avoid repeated dynamic_pointer_cast loops
        static const std::array<const char*, 3> bucket_keys = {"fly_agent", "ExploreFlyAgent", "PlannerFlyAgent"};
        for (const auto* key : bucket_keys) {
            auto it = agents_by_type_.find(key);
            if (it != agents_by_type_.end() && !it->second.empty()) {
                auto casted = CastAgents<FlyAgent>(it->second);
                fly_agents->insert(fly_agents->end(),
                                   std::make_move_iterator(casted.begin()),
                                   std::make_move_iterator(casted.end()));
            }
        }
        return fly_agents;
    }

    // Read-only access to the schema for a given agent_type. Used by the GUI
    // to render the same feature pipeline the network consumes — keeps the
    // displayed "Network Input" identical to what GetBatchedObservations() emits.
    const FeatureSchema* GetSchema(const std::string& agent_type) const {
        auto it = schemas_.find(agent_type);
        return it == schemas_.end() ? nullptr : &it->second;
    }

    // Returns the planner instance (or nullptr when the active hierarchy has none).
    // Multiple planners aren't currently a thing, so we expose just the first.
    std::shared_ptr<PlannerAgent> GetPlannerAgent() const {
        auto it = agents_by_type_.find("planner_agent");
        if (it == agents_by_type_.end() || it->second.empty()) return nullptr;
        return std::dynamic_pointer_cast<PlannerAgent>(it->second.front());
    }

    std::function<void()> onUpdateRLStatus;
private:
    std::shared_ptr<GridMap> gridmap_;
    std::shared_ptr<FireModelRenderer> model_renderer_;
    FireModelParameters& parameters_;
    std::unordered_map<std::string, std::vector<std::shared_ptr<Agent>>> agents_by_type_;
    // Feature schemas for schema-driven observation extraction
    std::unordered_map<std::string, FeatureSchema> schemas_;

    //Flags
    bool eval_mode_ = false;
    int frame_ctrl_ = 0;

    // Rewards Collection for Debugging!
    int total_env_steps_;

    pybind11::dict rl_status_; // Status of the current episode
};


#endif //ROSHAN_REINFORCEMENTLEARNING_HANDLER_H
