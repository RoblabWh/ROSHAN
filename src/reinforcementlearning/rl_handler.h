//
// Created by nex on 04.06.24.
//

#ifndef ROSHAN_REINFORCEMENTLEARNING_HANDLER_H
#define ROSHAN_REINFORCEMENTLEARNING_HANDLER_H

#include <shared_mutex>
#include <deque>
#include <utility>
#include <vector>
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

namespace py = pybind11;

// Squared distance (no sqrt)
inline double dist2(const std::pair<double,double>& a,
                    const std::pair<double,double>& b) {
    const double dx = a.first  - b.first;
    const double dy = a.second - b.second;
    return dx*dx + dy*dy;
}

// X,Y distance
inline std::pair<double,double> dist_vec(const std::pair<double,double>& a,
                   const std::pair<double,double>& b) {
    return {(b.first - a.first),
            (b.second - a.second)};
}

// True collision (discs touching/overlapping)
inline bool circlesCollide(const double d1, double r1, double r2) {
    const double R = r1 + r2;
    return d1 <= R*R;
}

// “Almost” collision (early warning / safety buffer)
inline bool almostCollide(const std::pair<double,double>& p1, double r1,
                          const std::pair<double,double>& p2, double r2,
                          double safety_margin = 0.25) {
    const double R = r1 + r2 + safety_margin;
    return dist2(p1, p2) <= R*R;
}


// Same, but for early warnings
inline std::vector<std::pair<int,int>> findAlmostCollisions(const std::vector<std::shared_ptr<FlyAgent>>& agents, double safety_margin = 0.25) {
    std::vector<std::pair<int,int>> hits;
    const int n = static_cast<int>(agents.size());
    for (int i = 0; i < n; ++i) {
        for (int j = i+1; j < n; ++j) {
            if (almostCollide(agents[i]->GetGridPositionDouble(), agents[i]->GetDroneSize(),
                               agents[j]->GetGridPositionDouble(), agents[j]->GetDroneSize(),
                               safety_margin))  {
                hits.emplace_back(i, j);
                              }
        }
    }
    return hits;
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
    for (int i = 0; i < n; ++i) {
        for (int j = i+1; j < n; ++j) {
            auto vec_dist = dist_vec(agents[i]->GetGridPositionDouble(), agents[j]->GetGridPositionDouble());
            if (std::fabs(vec_dist.first) < view_range_h && std::fabs(vec_dist.second) < view_range_h) {
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

                agents[i]->AppendDistance({vec_dist.first / view_range, vec_dist.second / view_range, a_j_dvx.first, a_j_dvx.second});
                agents[j]->AppendDistance({-vec_dist.first / view_range, -vec_dist.second / view_range, a_i_dvx.first, a_i_dvx.second});
                agents[i]->AppendMask(true);
                agents[j]->AppendMask(true);
                if (circlesCollide(dist2(agents[i]->GetGridPositionDouble(), agents[j]->GetGridPositionDouble()), agents[i]->GetDroneSize(), agents[j]->GetDroneSize())) {
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

inline void handleCollisions(const std::vector<std::pair<int,int>>& collisions, const std::vector<std::shared_ptr<FlyAgent>>& agents) {
    for (auto collision : collisions) {
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

    std::unordered_map<std::string, std::vector<std::deque<std::shared_ptr<State>>>> GetObservations();
    void StepDroneManual(int drone_idx, double speed_x, double speed_y, int water_dispense);
    void ResetEnvironment(Mode mode);
//    std::tuple<std::unordered_map<std::string, std::vector<std::deque<std::shared_ptr<State>>>>,
//            std::vector<double>,
//            std::vector<bool>,
//            std::unordered_map<std::string, bool>,
//            double> Step(const std::string& agent_type, std::vector<std::shared_ptr<Action>> actions);
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

        bool fly_agents_exist = agents_by_type_.find("fly_agent") != agents_by_type_.end();
        bool explore_fly_agents_exist = agents_by_type_.find("ExploreFlyAgent") != agents_by_type_.end();
        bool planner_fly_agents_exist = agents_by_type_.find("PlannerFlyAgent") != agents_by_type_.end();
        if (!fly_agents_exist && !explore_fly_agents_exist && !planner_fly_agents_exist) {
            return std::make_shared<std::vector<std::shared_ptr<FlyAgent>>>();
        }
        if (fly_agents_exist) {
            for(const auto& agent : agents_by_type_["fly_agent"]) {
                auto fly_agent = std::dynamic_pointer_cast<FlyAgent>(agent);
                if (fly_agent){
                    fly_agents->push_back(std::shared_ptr<FlyAgent>(fly_agent));
                } else {
                    std::cerr << "Non-fly_agent is not a fly_agent!\n";
                }
            }
        }
        if (explore_fly_agents_exist) {
            for(const auto& agent : agents_by_type_["ExploreFlyAgent"]) {
                auto fly_agent = std::dynamic_pointer_cast<FlyAgent>(agent);
                if (fly_agent){
                    fly_agents->push_back(std::shared_ptr<FlyAgent>(fly_agent));
                } else {
                    std::cerr << "Non-FlyAgent is not a FlyAgent!\n";
                }
            }
        }
        if (planner_fly_agents_exist) {
            for(const auto& agent : agents_by_type_["PlannerFlyAgent"]) {
                auto fly_agent = std::dynamic_pointer_cast<FlyAgent>(agent);
                if (fly_agent){
                    fly_agents->push_back(std::shared_ptr<FlyAgent>(fly_agent));
                } else {
                    std::cerr << "Non-FlyAgent is not a FlyAgent!\n";
                }
            }
        }
        return fly_agents;
    }
    std::function<void()> onUpdateRLStatus;
private:
    std::shared_ptr<GridMap> gridmap_;
    std::shared_ptr<FireModelRenderer> model_renderer_;
    FireModelParameters& parameters_;
    std::unordered_map<std::string, std::vector<std::shared_ptr<Agent>>> agents_by_type_;

    //Flags
    bool eval_mode_ = false;
    int frame_ctrl_ = 0;

    // Rewards Collection for Debugging!
    int total_env_steps_;

    pybind11::dict rl_status_; // Status of the current episode
};


#endif //ROSHAN_REINFORCEMENTLEARNING_HANDLER_H
