//
// Declarative observation pipeline: define features once in C++,
// extraction/batching/Python access flow automatically.
//

#ifndef ROSHAN_FEATURE_SCHEMA_H
#define ROSHAN_FEATURE_SCHEMA_H

#include <string>
#include <vector>
#include <functional>
#include <stdexcept>
#include "src/reinforcementlearning/agents/agent_state.h"

enum class FeatureGroupType { FIXED, RELATIONAL, SET };

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

    // For RELATIONAL/SET groups: bulk extractor fills entire slice at once
    using BulkExtractFn = std::function<void(const AgentState&, float*, bool*, int, int)>;
    BulkExtractFn extract_bulk = nullptr;
    int max_entities = 0;   // max neighbors, max fires, etc. (0 = determine dynamically)
    int bulk_dims = 0;      // features per entity

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
        groups.push_back({group_name, type, {}, nullptr, 0, 0});
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
