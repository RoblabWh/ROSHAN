//
// SchemaTable.h - Schema-driven feature renderer for ImGui
//
// Walks a FeatureSchema and invokes the same extract / extract_bulk lambdas
// the observation batcher uses (rl_handler.cpp::GetBatchedObservations), so
// the table is truth-by-construction: whatever the network sees, the GUI shows.
//

#ifndef ROSHAN_SCHEMATABLE_H
#define ROSHAN_SCHEMATABLE_H

#include "imgui.h"
#include "../UITypes.h"
#include "src/reinforcementlearning/feature_schema.h"
#include "reinforcementlearning/agents/agent_state.h"
#include <algorithm>
#include <vector>
#include <string>
#include <cstdio>

namespace ui {

class SchemaTable {
public:
    static void Draw(const FeatureSchema* schema, const AgentState& state) {
        if (!schema) {
            ImGui::TextDisabled("No schema available");
            return;
        }
        for (const auto& g : schema->groups) {
            ImGui::PushID(g.name.c_str());
            const std::string header = g.name + " [" + GroupTypeLabel(g.type) + "]";
            if (ImGui::CollapsingHeader(header.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
                if (g.type == FeatureGroupType::FIXED) {
                    DrawFixed(g, state);
                } else {
                    DrawBulk(g, state, g.type == FeatureGroupType::RELATIONAL);
                }
            }
            ImGui::PopID();
        }
    }

private:
    static const char* GroupTypeLabel(FeatureGroupType t) {
        switch (t) {
            case FeatureGroupType::FIXED:      return "FIXED";
            case FeatureGroupType::RELATIONAL: return "RELATIONAL";
            case FeatureGroupType::SET:        return "SET";
        }
        return "?";
    }

    static void DrawFixed(const FeatureGroup& g, const AgentState& state) {
        if (g.columns.empty()) {
            ImGui::TextDisabled("(no columns)");
            return;
        }
        if (!ImGui::BeginTable("FixedTable", 2,
                               ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) return;
        ImGui::TableSetupColumn("Column", ImGuiTableColumnFlags_WidthFixed, 200.0f);
        ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableHeadersRow();

        std::vector<float> buf;
        for (const auto& col : g.columns) {
            buf.assign(col.dims, 0.0f);
            if (col.extract) col.extract(state, buf.data());
            ImGui::TableNextRow();
            ImGui::TableNextColumn(); ImGui::TextUnformatted(col.name.c_str());
            ImGui::TableNextColumn(); ImGui::TextUnformatted(FormatRow(buf, col.dims).c_str());
        }
        ImGui::EndTable();
    }

    static void DrawBulk(const FeatureGroup& g, const AgentState& state, bool show_mask) {
        const int K = g.entity_count ? std::max(0, g.entity_count(state)) : 0;
        const int D = g.bulk_dims;

        if (K == 0 || D == 0 || !g.extract_bulk) {
            ImGui::TextDisabled("(empty)");
            return;
        }

        std::vector<float> data(static_cast<size_t>(K) * D, 0.0f);
        // std::vector<bool> doesn't expose a writable contiguous bool*; use char as a stand-in.
        std::vector<char> mask(K, 0);
        g.extract_bulk(state, data.data(),
                       reinterpret_cast<bool*>(mask.data()), K, D);

        const int n_cols = show_mask ? 3 : 2;
        if (!ImGui::BeginTable("BulkTable", n_cols,
                               ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) return;
        ImGui::TableSetupColumn("Slot", ImGuiTableColumnFlags_WidthFixed, 200.0f);
        ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);
        if (show_mask) ImGui::TableSetupColumn("Mask", ImGuiTableColumnFlags_WidthFixed, 60.0f);
        ImGui::TableHeadersRow();

        std::vector<float> row(D, 0.0f);
        for (int i = 0; i < K; ++i) {
            for (int d = 0; d < D; ++d) row[d] = data[i * D + d];
            ImGui::TableNextRow();
            ImGui::TableNextColumn(); ImGui::Text("%s_%d", g.name.c_str(), i);
            ImGui::TableNextColumn(); ImGui::TextUnformatted(FormatRow(row, D).c_str());
            if (show_mask) {
                ImGui::TableNextColumn();
                const bool m = mask[i] != 0;
                ImGui::TextColored(m ? colors::kStatusActive : colors::kStatusInactive,
                                   "%s", m ? "true" : "false");
            }
        }
        ImGui::EndTable();
    }

    static std::string FormatRow(const std::vector<float>& v, int dims) {
        char buf[256];
        if (dims == 1) {
            std::snprintf(buf, sizeof(buf), "%.6f", v[0]);
            return buf;
        }
        if (dims == 2) {
            std::snprintf(buf, sizeof(buf), "(%.6f, %.6f)", v[0], v[1]);
            return buf;
        }
        std::string s = "[";
        for (int i = 0; i < dims; ++i) {
            std::snprintf(buf, sizeof(buf), "%s%.6f", i == 0 ? "" : ", ", v[i]);
            s += buf;
        }
        s += "]";
        return s;
    }
};

} // namespace ui

#endif // ROSHAN_SCHEMATABLE_H
