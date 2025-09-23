//
// Created by nex on 25.07.23.
//

#ifndef ROSHAN_AGENT_H
#define ROSHAN_AGENT_H

#define DEBUG_REWARD_NO

#include <string>
#include <memory>
#include <vector>
#include <deque>
#include <unordered_map>
#include "firespin/model_parameters.h"
#include "firespin/utils.h"
#include "src/utils.h"
#include "agent_state.h"

class Action;
class State;
class GridMap;
class FireModelRenderer;

enum AgentType {
    FLY_AGENT,
    EXPLORE_AGENT,
    PLANNER_AGENT
};

class Agent : public std::enable_shared_from_this<Agent> {
public:
    explicit Agent(FireModelParameters &parameters, int buffer_size = 300) :
    rewards_(buffer_size),
    parameters_(parameters) {}
    virtual ~Agent() = default;

    virtual void
    ExecuteAction(std::shared_ptr<Action> action, std::string hierarchy_type, std::shared_ptr<GridMap> gridMap) = 0;

    virtual AgentTerminal GetTerminalStates(bool eval_mode, const std::shared_ptr<GridMap> &grid_map, int total_env_steps) = 0;
    
    virtual bool GetPerformedHierarchyAction() const { return did_hierarchy_step; };
    virtual double CalculateReward() = 0;
    virtual void StepReset() = 0;
    virtual void Reset(Mode mode,
                       const std::shared_ptr<GridMap>& grid_map,
                       const std::shared_ptr<FireModelRenderer>& model_renderer) = 0;
    int GetFrameSkips() const { return frame_skips_; }
    int GetFrameCtrl() const { return frame_ctrl_; }
    void SetFrameControl(int frame_ctrl) { frame_ctrl_ = frame_ctrl; }
    AgentType GetAgentType() const { return agent_type_; }
    std::deque<std::shared_ptr<State>> GetObservations() const;
    void UpdateStates(const std::shared_ptr<GridMap>& grid_map);

    int GetId() const { return id_; }

    void ModifyReward(double reward) {
        reward_components_["Intrinsic Reward"] = reward;
        rewards_.ModifyLast(static_cast<float>(reward)); }
    CircularBuffer<float> GetEpisodeRewards() { return rewards_; }

    std::unordered_map<std::string, double> GetRewardComponents() { return reward_components_; }

    AgentState GetLastState() const { return *agent_states_[0]; }
//    std::deque<AgentState> GetStates() { return agent_states_; }
protected:
    AgentType agent_type_{};
    FireModelParameters& parameters_;
    int id_{};
    std::deque<std::shared_ptr<AgentState>> agent_states_;
    static double ComputeTotalReward(const std::unordered_map<std::string, double> &rewards);
    virtual std::shared_ptr<AgentState> BuildAgentState(const std::shared_ptr<GridMap> &grid_map) = 0;
    void LogRewards(const std::unordered_map<std::string, double> &rewards);

    void SetReward(double reward) { rewards_.put(static_cast<float>(reward));}
    CircularBuffer<float> rewards_;

    bool objective_reached_ = false; // This is the Agents Objective, must be set to true when the agent reaches its goal,
                                     // this COULD be a desired reached positional goal, or another objective
    bool agent_terminal_state_ = false;
    std::unordered_map<std::string, bool> agent_terminal_states_;
    int frame_skips_{};
    int frame_ctrl_{};

    int env_steps_remaining_ = 0;
    bool did_hierarchy_step = false; // Was the last action a hierarchy step? Determines if reward needs to be calculated
    std::unordered_map<std::string, double> reward_components_;
    std::string agent_sub_type_;
    int time_steps_{};
};

#endif //ROSHAN_AGENT_H
