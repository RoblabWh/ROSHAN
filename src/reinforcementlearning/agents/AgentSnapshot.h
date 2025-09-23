//
// Created by nex on 21.09.25.
//

#ifndef ROSHAN_AGENTSNAPSHOT_H
#define ROSHAN_AGENTSNAPSHOT_H
#include <cmath>
#include <functional>
#include <memory>
#include <string>
#include <vector>

// Kinematics + Goal
struct Kin {
    double px, py;          // position in world units
    double vx, vy;          // velocity in world units/sec
    double max_speed;       // scalar
};

struct Goal {
    double gx, gy;          // world units
};

// Local perception / maps
struct Perception {
    // Consider switching heavy containers to views/spans if possible.
    std::shared_ptr<const std::vector<std::vector<std::vector<int>>>> drone_view; // [C, H, W]
    std::shared_ptr<const std::vector<std::vector<double>>> total_drone_view;     // [H, W]
    std::shared_ptr<const std::vector<std::vector<int>>> exploration_map;         // [H, W]
    std::shared_ptr<const std::vector<std::vector<double>>> fire_map;             // [H, W]
};

// Neighborhood (K neighbors, feature layout TBD)
struct Neighborhood {
    // [[dx,dy,dvx,dvy,dist]
    std::shared_ptr<const std::vector<std::vector<double>>> feats; // [K, F]
    std::shared_ptr<const std::vector<bool>> mask;                  // [K]
};

// World/meta
struct WorldMeta {
    double cell_size;
    int rows, cols;
    double norm_scale; // vision range / normalization scale
};

// Immutable snapshot assembled each step
struct AgentSnapshot {
    Kin kin;
    Goal goal;
    // Perception perc;
    Neighborhood nbr;
    WorldMeta meta;
    int water_dispense; // small scalars ok to keep here

    // Convenience derived values (computed once)
    double speed()   const { return std::hypot(kin.vx, kin.vy); }
    double dx_goal() const { return (goal.gx - kin.px); }
    double dy_goal() const { return (goal.gy - kin.py); }
    double dist_goal()const { return std::hypot(dx_goal(), dy_goal()); }
    std::string print()const {
        return "p=(" + std::to_string(kin.px) + "," + std::to_string(kin.py) + ") v=(" +
               std::to_string(kin.vx) + "," + std::to_string(kin.vy) + ") g=(" +
               std::to_string(goal.gx) + "," + std::to_string(goal.gy) + ") d=" +
               std::to_string(dist_goal());
    }
};

struct FeatureColumn {
    std::string name;             // e.g., "velocity_norm"
    int dim;                      // how many scalars this feature adds (e.g., 2 for (x,y), 1 for scalar)
    std::function<void(const AgentSnapshot&, float* out)> fill;
    // writes 'dim' values into out[offset..offset+dim)
};

struct FeatureSpec {
    std::vector<std::string> names; // ordered list of features you want
};

struct FeaturePack {
    std::vector<float> data;  // flattened (T, F) or (F) depending on your use
    int F{};                    // total feature dimension
    std::unordered_map<std::string, std::pair<int,int>> columns; // name -> [start, end)
};

class FeatureRegistry {
public:
    // Register built-in columns once at startup
    FeatureRegistry(double clip01_ttc_cap) : ttc_cap_(clip01_ttc_cap) { register_default(); }

    void register_default() {
        // velocity normalized by max_speed (no norm_scale mixing!)
        add({"velocity_norm", 2,
            [](const AgentSnapshot& s, float* out){
                const double ms = std::max(s.kin.max_speed, 1e-8);
                out[0] = static_cast<float>(std::clamp(s.kin.vx / ms, -1.0, 1.0));
                out[1] = static_cast<float>(std::clamp(s.kin.vy / ms, -1.0, 1.0));
            }
        });
        // delta goal normalized by norm_scale
        add({"delta_goal", 2,
            [](const AgentSnapshot& s, float* out){
                const double ns = std::max(s.meta.norm_scale, 1e-8);
                out[0] = static_cast<float>(std::clamp((s.goal.gx - s.kin.px)/ns, -1.0, 1.0));
                out[1] = static_cast<float>(std::clamp((s.goal.gy - s.kin.py)/ns, -1.0, 1.0));
            }
        });

        // speed (0..1)
        add({"speed", 1,
            [](const AgentSnapshot& s, float* out){
                const double ms = std::max(s.kin.max_speed, 1e-8);
                out[0] = static_cast<float>(std::min(std::hypot(s.kin.vx, s.kin.vy)/ms, 1.0));
            }
        });

        // distance_to_goal (0..1)
        add({"distance_to_goal", 1,
            [](const AgentSnapshot& s, float* out){
                const double ns = std::max(s.meta.norm_scale, 1e-8);
                out[0] = static_cast<float>(std::min(std::hypot(s.goal.gx - s.kin.px, s.goal.gy - s.kin.py)/ns, 1.0));
            }
        });

        // Example: neighbor pooled features already computed elsewhere (e.g., your NeighborEncoder)
        // Here we only reserve a slot to copy precomputed (D) neighbor embedding if you have it.
        // add_dynamic(...) pattern shown below.
    }

    void add(FeatureColumn col) { cols_.push_back(std::move(col)); }

    // Build a single snapshot’s feature vector
    FeaturePack build(const AgentSnapshot &s, const FeatureSpec& spec) const {
        // compute total F
        int F = 0;
        std::vector<const FeatureColumn*> used;
        used.reserve(spec.names.size());
        for (const auto& name : spec.names) {
            const auto* c = find(name);
            if (!c) throw std::runtime_error("Unknown feature: " + name);
            used.push_back(c);
            F += c->dim;
        }

        FeaturePack pack;
        pack.data.resize(F);
        pack.F = F;

        int off = 0;
        for (const auto* c : used) {
            pack.columns[c->name] = {off, off + c->dim};
            c->fill(s, pack.data.data() + off);   // NOTE: float* out
            off += c->dim;
        }
        return pack;
    }

private:
    const FeatureColumn* find(const std::string& name) const {
        for (const auto& c : cols_) if (c.name == name) return &c;
        return nullptr;
    }
    std::vector<FeatureColumn> cols_;
    double ttc_cap_;
};

#endif //ROSHAN_AGENTSNAPSHOT_H