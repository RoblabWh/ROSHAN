//
// Feature schema definitions for each agent type.
// Adding a new observation feature = one line here.
//

#ifndef ROSHAN_FEATURE_DEFINITIONS_H
#define ROSHAN_FEATURE_DEFINITIONS_H

#include "feature_schema.h"

inline FeatureSchema CreateFlyAgentSchema() {
    FeatureSchema schema;

    auto& agent = schema.AddGroup("agent", FeatureGroupType::FIXED);
    agent.Add("id", 1, [](const AgentState& s, float* o) {
        o[0] = static_cast<float>(s.GetID());
    });
    agent.Add("velocity", 2, [](const AgentState& s, float* o) {
        auto v = s.GetVelocityNorm();
        o[0] = static_cast<float>(v.first);
        o[1] = static_cast<float>(v.second);
    });
    agent.Add("delta_goal", 2, [](const AgentState& s, float* o) {
        auto d = s.GetDeltaGoal();
        o[0] = static_cast<float>(d.first);
        o[1] = static_cast<float>(d.second);
    });
    agent.Add("cos_sin", 2, [](const AgentState& s, float* o) {
        auto c = s.GetCosSinToGoal();
        o[0] = static_cast<float>(c.first);
        o[1] = static_cast<float>(c.second);
    });
    agent.Add("speed", 1, [](const AgentState& s, float* o) {
        o[0] = static_cast<float>(s.GetSpeed());
    });
    agent.Add("distance_to_goal", 1, [](const AgentState& s, float* o) {
        o[0] = static_cast<float>(s.GetDistanceToGoal());
    });

    // --- Adding a new feature is just one more line: ---
    // agent.Add("water_level", 1, [](const AgentState& s, float* o) {
    //     o[0] = static_cast<float>(s.GetWaterLevel());
    // });

    auto& neighbors = schema.AddGroup("neighbors", FeatureGroupType::RELATIONAL);
    neighbors.bulk_dims = 4;  // [dx, dy, dvx, dvy]
    neighbors.extract_bulk = [](const AgentState& s, float* data, bool* mask, int K, int D) {
        const auto& dists = s.GetDistancesToOtherAgentsRef();
        const auto& m = s.GetDistancesMaskRef();
        for (int k = 0; k < K; k++) {
            mask[k] = (k < static_cast<int>(m.size())) && m[k];
            if (k < static_cast<int>(dists.size())) {
                const auto& row = dists[k];
                for (int f = 0; f < D; f++) {
                    data[k * D + f] = (f < static_cast<int>(row.size()))
                        ? static_cast<float>(row[f]) : 0.0f;
                }
            } else {
                for (int f = 0; f < D; f++) {
                    data[k * D + f] = 0.0f;
                }
            }
        }
    };

    return schema;
}

inline FeatureSchema CreateExploreAgentSchema() {
    FeatureSchema schema;

    // Minimal placeholder schema — ExploreAgent is heuristic (no_algo)
    // and does not consume observations for training.
    auto& agent = schema.AddGroup("agent", FeatureGroupType::FIXED);
    agent.Add("id", 1, [](const AgentState& s, float* o) {
        o[0] = static_cast<float>(s.GetID());
    });

    return schema;
}

inline FeatureSchema CreatePlannerAgentSchema() {
    FeatureSchema schema;

    auto& drones = schema.AddGroup("drone_positions", FeatureGroupType::SET);
    drones.bulk_dims = 2;
    drones.extract_bulk = [](const AgentState& s, float* data, bool* mask, int M, int D) {
        const auto& dp = s.GetDronePositionsRef();
        for (int i = 0; i < M; i++) {
            mask[i] = i < static_cast<int>(dp.size());
            if (mask[i]) {
                data[i * D] = static_cast<float>(dp[i].first);
                data[i * D + 1] = static_cast<float>(dp[i].second);
            } else {
                data[i * D] = 0.0f;
                data[i * D + 1] = 0.0f;
            }
        }
    };

    auto& goals = schema.AddGroup("goal_positions", FeatureGroupType::SET);
    goals.bulk_dims = 2;
    goals.extract_bulk = [](const AgentState& s, float* data, bool* mask, int M, int D) {
        const auto& gp = s.GetGoalPositionsRef();
        for (int i = 0; i < M; i++) {
            mask[i] = i < static_cast<int>(gp.size());
            if (mask[i]) {
                data[i * D] = static_cast<float>(gp[i].first);
                data[i * D + 1] = static_cast<float>(gp[i].second);
            } else {
                data[i * D] = 0.0f;
                data[i * D + 1] = 0.0f;
            }
        }
    };

    auto& fires = schema.AddGroup("fire_positions", FeatureGroupType::SET);
    fires.bulk_dims = 2;
    fires.extract_bulk = [](const AgentState& s, float* data, bool* mask, int M, int D) {
        const auto& fp = s.GetFirePositionsRef();
        for (int i = 0; i < M; i++) {
            mask[i] = i < static_cast<int>(fp.size());
            if (mask[i]) {
                data[i * D] = static_cast<float>(fp[i].first);
                data[i * D + 1] = static_cast<float>(fp[i].second);
            } else {
                data[i * D] = 0.0f;
                data[i * D + 1] = 0.0f;
            }
        }
    };

    return schema;
}

#endif //ROSHAN_FEATURE_DEFINITIONS_H
