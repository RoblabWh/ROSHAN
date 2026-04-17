//
// Feature schema definitions for each agent type.
//
// Each CreateXxxSchema() registers feature groups with extraction lambdas.
// Adding a new observation feature = one Add() call or one new group here;
// batching, Python exposure, and schema metadata are handled automatically
// by rl_handler.cpp and python_bindings.cpp.
//
// Group type choice guide:
//   FIXED       — agent's own scalar/vector state (columns with named extractors)
//   RELATIONAL  — per-agent neighbors whose validity varies per timestep (mask exposed)
//   SET         — shared scene collections where zero-padding suffices (no mask exposed)
//

#ifndef ROSHAN_FEATURE_DEFINITIONS_H
#define ROSHAN_FEATURE_DEFINITIONS_H

#include "feature_schema.h"

// Helper: bulk extractor for vector<pair<float,float>> fields (e.g., positions).
// Extracts into (M, 2) with mask + zero-padding for absent slots.
using PairVec = std::vector<std::pair<double, double>>;
inline FeatureGroup::BulkExtractFn MakePairBulkExtractor(
        std::function<const PairVec&(const AgentState&)> get_vec) {
    return [get_vec](const AgentState& s, float* data, bool* mask, int M, int D) {
        const auto& vec = get_vec(s);
        const int n = static_cast<int>(vec.size());
        for (int i = 0; i < M; i++) {
            mask[i] = i < n;
            if (mask[i]) {
                data[i * D]     = static_cast<float>(vec[i].first);
                data[i * D + 1] = static_cast<float>(vec[i].second);
            } else {
                data[i * D]     = 0.0f;
                data[i * D + 1] = 0.0f;
            }
        }
    };
}

inline FeatureSchema CreateFlyAgentSchema() {
    FeatureSchema schema;

    auto& agent = schema.AddGroup("agent", FeatureGroupType::FIXED);
    agent.Add("id", 1, [](const AgentState& s, float* o) {
        o[0] = static_cast<float>(s.id);
    });
    agent.Add("velocity", 2, [](const AgentState& s, float* o) {
        auto v = state_features::VelocityNorm(s);
        o[0] = static_cast<float>(v.first);
        o[1] = static_cast<float>(v.second);
    });
    agent.Add("delta_goal", 2, [](const AgentState& s, float* o) {
        auto d = state_features::DeltaGoal(s);
        o[0] = static_cast<float>(d.first);
        o[1] = static_cast<float>(d.second);
    });
    agent.Add("cos_sin", 2, [](const AgentState& s, float* o) {
        auto c = state_features::CosSinToGoal(s);
        o[0] = static_cast<float>(c.first);
        o[1] = static_cast<float>(c.second);
    });
    agent.Add("speed", 1, [](const AgentState& s, float* o) {
        o[0] = static_cast<float>(state_features::Speed(s));
    });
    agent.Add("distance_to_goal", 1, [](const AgentState& s, float* o) {
        o[0] = static_cast<float>(state_features::DistanceToGoal(s));
    });

    // --- Adding a new feature is just one more line: ---
    // agent.Add("water_level", 1, [](const AgentState& s, float* o) {
    //     o[0] = static_cast<float>(s.water_level);
    // });

    // RELATIONAL: neighbor count and validity vary per agent per timestep,
    // so the mask is exposed to Python for use in masked attention/pooling.
    auto& neighbors = schema.AddGroup("neighbors", FeatureGroupType::RELATIONAL);
    neighbors.bulk_dims = 4;  // [dx, dy, dvx, dvy]
    neighbors.entity_count = [](const AgentState& s) {
        return static_cast<int>(s.distances_to_other_agents->size());
    };
    neighbors.extract_bulk = [](const AgentState& s, float* data, bool* mask, int K, int D) {
        const auto& dists = *s.distances_to_other_agents;
        const auto& m = *s.distances_mask;
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
        o[0] = static_cast<float>(s.id);
    });

    return schema;
}

inline FeatureSchema CreatePlannerAgentSchema() {
    FeatureSchema schema;

    // FIXED: global fire summary (fire density + centroid + wind vector)
    auto& fire_globals = schema.AddGroup("fire_globals", FeatureGroupType::FIXED);
    fire_globals.Add("fire_count", 1, [](const AgentState& s, float* o) {
        o[0] = static_cast<float>(s.fire_count);
    });
    fire_globals.Add("fire_centroid", 2, [](const AgentState& s, float* o) {
        o[0] = static_cast<float>(s.fire_centroid.first);
        o[1] = static_cast<float>(s.fire_centroid.second);
    });
    fire_globals.Add("wind", 2, [](const AgentState& s, float* o) {
        o[0] = static_cast<float>(s.wind_vector.first);
        o[1] = static_cast<float>(s.wind_vector.second);
    });

    // SET groups: drone/goal/fire positions are global scene collections shared
    // across all agents. Zero-padding is sufficient for absent slots — no mask
    // is exposed to Python (unlike RELATIONAL neighbors).
    auto& drones = schema.AddGroup("drone_positions", FeatureGroupType::SET);
    drones.bulk_dims = 2;
    drones.entity_count = [](const AgentState& s) {
        return static_cast<int>(s.drone_positions->size());
    };
    drones.extract_bulk = MakePairBulkExtractor(
        [](const AgentState& s) -> const PairVec& { return *s.drone_positions; });

    auto& goals = schema.AddGroup("goal_positions", FeatureGroupType::SET);
    goals.bulk_dims = 2;
    goals.entity_count = [](const AgentState& s) {
        return s.goal_positions ? static_cast<int>(s.goal_positions->size()) : 0;
    };
    goals.extract_bulk = MakePairBulkExtractor(
        [](const AgentState& s) -> const PairVec& { return *s.goal_positions; });

    // Fires are RELATIONAL so the validity mask reaches Python — the planner's
    // pointer-network and cross-attention must distinguish real fires from
    // zero-padded slots, otherwise phantom fires at the map centre attract
    // attention mass and can be sampled as goals.
    auto& fires = schema.AddGroup("fire_positions", FeatureGroupType::RELATIONAL);
    fires.bulk_dims = 2;
    fires.entity_count = [](const AgentState& s) {
        return static_cast<int>(s.fire_positions->size());
    };
    fires.extract_bulk = MakePairBulkExtractor(
        [](const AgentState& s) -> const PairVec& { return *s.fire_positions; });

    return schema;
}

#endif //ROSHAN_FEATURE_DEFINITIONS_H
