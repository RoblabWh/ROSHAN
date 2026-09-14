//
// Created by nex on 21.06.25.
//

#ifndef ROSHAN_PLANNER_AGENT_H
#define ROSHAN_PLANNER_AGENT_H

#include "agent.h"
#include "reinforcementlearning/actions/plan_action.h"
#include "reinforcementlearning/agents/explore_agent.h"


class GridMap;
class FireModelRenderer;

class PlannerAgent : public Agent {
public:
    explicit PlannerAgent(FireModelParameters &parameters, int total_id, int id, int time_steps);

    void ExecuteAction(std::shared_ptr<Action> action, std::string hierarchy_type, std::shared_ptr<GridMap> gridMap) override {
        action->ExecuteOn(shared_from_this(), hierarchy_type, gridMap);
    }

    // Deferred goal assignment: runs after CalculateReward so the reward reflects the goals
    // that governed the just-completed window (not the freshly-decided ones). Dispatches to
    // CommitPlan for PlanAction; ignores other action types.
    void CommitAction(std::shared_ptr<Action> action, std::string hierarchy_type, std::shared_ptr<GridMap> gridMap) override {
        (void)hierarchy_type;
        if (auto plan = std::dynamic_pointer_cast<PlanAction>(action)) {
            CommitPlan(plan.get(), gridMap);
        }
    }

    void Initialize(std::shared_ptr<ExploreAgent> explore_agent,
                    std::vector<std::shared_ptr<FlyAgent>> fly_agents,
                    const std::shared_ptr<GridMap> &grid_map);

    void Reset(Mode mode,
               const std::shared_ptr<GridMap>& grid_map,
               const std::shared_ptr<FireModelRenderer>& model_renderer) override;

    void PerformPlan(PlanAction* action, const std::string& hierarchy_type, const std::shared_ptr<GridMap>& gridMap);
    // Assigns the newly-decided goals to the fly agents and re-baselines the per-drone
    // DistanceProgress buffer against those goals. Called via CommitAction after the reward
    // for the completed window has been computed.
    void CommitPlan(PlanAction* action, const std::shared_ptr<GridMap>& gridMap);
    bool GetPerformedHierarchyAction() const override { return did_hierarchy_step; }
    double CalculateReward(const std::shared_ptr<GridMap>& grid_map) override;
    void StepReset() override {
        did_hierarchy_step = false;
    }

    AgentTerminal GetTerminalStates(bool eval_mode, const std::shared_ptr<GridMap>& grid_map, int total_env_steps) override;
    void SetEvalMode(bool eval_mode) { eval_mode_ = eval_mode; }
    const std::vector<std::shared_ptr<FlyAgent>>& GetFlyAgents() const { return fly_agents_; }
private:
    void InitializePlannerAgentStates(const std::shared_ptr<GridMap> &grid_map);
    // Computes Φ(s) = w_f·Φ_fire + w_w·Φ_water + w_d·Φ_dist for the current grid state.
    // Caches Φ_fire/Φ_water/Φ_dist in phi_*_last_ for tensorboard logging.
    double ComputePotential(const std::shared_ptr<GridMap>& grid_map);

    // Agents
    std::shared_ptr<ExploreAgent> explore_agent_;
    std::vector<std::shared_ptr<FlyAgent>> fly_agents_;

    //Gridmap
    std::shared_ptr<GridMap> gridmap_;

    bool did_hierarchy_step = false;
    std::vector<std::deque<std::pair<double, double>>> perfect_goals_;
    int goal_idx_ = 0;
    int revisited_cells_{};
    int extinguished_fires_ = 0;

    bool extinguished_last_fire_ = false;
    bool eval_mode_ = false;
    // Per-drone normalized distances from the previous planner step. -1.0 sentinel
    // means the drone was ineligible last step (no baseline to compare against), so
    // joining/leaving the eligible set doesn't fabricate a spurious progress signal.
    // Indexed by fly_agents_ position; resized lazily.
    std::vector<double> prev_drone_distances_;
    double prev_num_burning_ = -1.0;
    // Per-drone water levels from the previous planner step, used to compute the
    // WaterRefill dense reward. Empty vector is the "first call" sentinel.
    std::vector<double> prev_water_levels_;
    // Per-drone planner-assigned goals from the previous planner step. Used to compute
    // the GoalCommit reward (sticky-assignment bonus that counters autoregressive
    // Categorical sampling noise). Empty vector is the "first call" sentinel.
    std::vector<std::pair<double, double>> prev_planner_goals_;

    // PBRS state. B_init_ is the burning-cell count captured once at episode reset;
    // prev_phi_ stores Φ(s_{t-1}) so F = γΦ(s_t) − Φ(s_{t-1}) can be emitted.
    // phi_initialized_ false → first planner step of episode → emit F = 0.
    int    B_init_{0};
    double prev_phi_{0.0};
    bool   phi_initialized_{false};
    // Component caches populated by ComputePotential, exposed via reward_components for
    // tensorboard logging only (do not affect the reward).
    double phi_fire_last_{0.0};
    double phi_water_last_{0.0};
    double phi_dist_last_{0.0};

    std::shared_ptr<AgentState> BuildAgentState(const std::shared_ptr<GridMap> &grid_map) override;
};

#endif //ROSHAN_PLANNER_AGENT_H
