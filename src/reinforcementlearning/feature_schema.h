//
// Declarative observation pipeline: define features once in C++,
// extraction/batching/Python access flow automatically.
//
// Three group types control how observations are shaped and delivered to Python:
//
//   FIXED       — Named columns with per-column extractors.
//                  Output: (N, T, D) where D = sum of all column dims.
//                  Use for an agent's own state (velocity, goal delta, ...).
//
//   RELATIONAL  — Variable-count entities with a bulk extractor.
//                  Output: (N, T, K, D) data  +  (N, T, K) boolean mask.
//                  The mask is exposed to Python so the network can distinguish
//                  valid entities from padding (e.g., masked attention over neighbors).
//                  Use for per-agent relationships that vary per timestep.
//
//   SET         — Variable-count entities with a bulk extractor.
//                  Output: (N, T, M, D) data only — no mask exposed.
//                  Invalid slots are zero-padded; the network treats zeros as absent.
//                  Use for global/shared collections (fire positions, goal positions).
//
// RELATIONAL and SET share the same bulk extraction interface (extract_bulk,
// entity_count, max_entities, bulk_dims). They are kept as separate types because
// they encode different semantic contracts about how missing data is communicated
// to the network, and may diverge further in the future (e.g., edge features for
// RELATIONAL, ordering guarantees for SET).
//

#ifndef ROSHAN_FEATURE_SCHEMA_H
#define ROSHAN_FEATURE_SCHEMA_H

#include <string>
#include <vector>
#include <functional>
#include "src/reinforcementlearning/agents/agent_state.h"

enum class FeatureGroupType {
    FIXED,       // Per-column extractors → (N, T, D), no mask
    RELATIONAL,  // Bulk extractor → (N, T, K, D) + (N, T, K) mask exposed to Python
    SET          // Bulk extractor → (N, T, M, D), mask internal only (zero-padded)
};

struct FeatureColumn {
    std::string name;
    int dims;
    std::function<void(const AgentState&, float*)> extract;
};

class FeatureGroup {
public:
    std::string name;
    FeatureGroupType type;

    // For FIXED groups: per-column extractors
    std::vector<FeatureColumn> columns;

    void Add(std::string col_name, int dims,
             std::function<void(const AgentState&, float*)> extract_fn) {
        columns.push_back({std::move(col_name), dims, std::move(extract_fn)});
    }

    [[nodiscard]] int TotalDims() const {
        int total = 0;
        for (const auto& col : columns) total += col.dims;
        return total;
    }

    // --- Bulk extraction (RELATIONAL and SET groups) ---
    // Both types use the same interface; the difference is whether the mask
    // reaches Python (RELATIONAL) or stays internal (SET). See rl_handler.cpp
    // GetBatchedObservations() for the branching logic.
    //
    // BulkExtractFn signature: (state, data_out, mask_out, max_entities, dims_per_entity)
    //   data_out: contiguous (max_entities * dims_per_entity) floats
    //   mask_out: contiguous (max_entities) bools — true = valid entity
    using BulkExtractFn = std::function<void(const AgentState&, float*, bool*, int, int)>;
    using EntityCountFn = std::function<int(const AgentState&)>;
    BulkExtractFn extract_bulk = nullptr;
    EntityCountFn entity_count = nullptr;  // actual entity count for dynamic sizing
    int max_entities = 0;   // capacity (0 = determine dynamically from entity_count)
    int bulk_dims = 0;      // feature dimensions per entity

    // Schema metadata for Python side
    [[nodiscard]] std::vector<std::pair<std::string, int>> GetColumnInfo() const {
        std::vector<std::pair<std::string, int>> info;
        if (type == FeatureGroupType::FIXED) {
            for (const auto& col : columns)
                info.emplace_back(col.name, col.dims);
        } else {
            // For bulk groups, report as single entry
            info.emplace_back(name, bulk_dims);
        }
        return info;
    }
};

class FeatureSchema {
public:
    std::vector<FeatureGroup> groups;

    FeatureGroup& AddGroup(const std::string& group_name, FeatureGroupType type) {
        groups.push_back({group_name, type, {}, nullptr, nullptr, 0, 0});
        return groups.back();
    }

    [[nodiscard]] const FeatureGroup* GetGroup(const std::string& group_name) const {
        for (const auto& g : groups) {
            if (g.name == group_name) return &g;
        }
        return nullptr;
    }
};

#endif //ROSHAN_FEATURE_SCHEMA_H
