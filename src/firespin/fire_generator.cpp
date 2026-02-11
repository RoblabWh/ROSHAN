//
// Created by nex on 24.08.25.
//

#include "fire_generator.h"

FireGenerator::FireGenerator(std::shared_ptr<GridMap> gridmap, FireModelParameters& parameters)
        : gridmap_(std::move(gridmap)), parameters_(parameters) {}

void FireGenerator::StartFires() {
    auto cells = static_cast<float>(gridmap_->GetNumCells());
    int fires = static_cast<int>(cells * parameters_.GetFirePercentage());
    if (fires <= 0) return;

    // Resolve Random pattern by sampling from {Cluster, Scattered, Ring}
    FirePattern pattern = parameters_.GetFirePattern();
    if (pattern == FirePattern::Random) {
        std::uniform_int_distribution<int> patternDist(0, 2);
        pattern = static_cast<FirePattern>(patternDist(parameters_.gen_));
    }

    switch (pattern) {
        case FirePattern::Scattered:
            IgniteScattered(fires);
            break;
        case FirePattern::Ring:
            IgniteRing(fires);
            break;
        case FirePattern::Cluster:
        default: {
            auto clusters = std::max(1, std::min(parameters_.num_fire_clusters_, fires));
            auto fires_per_cluster = fires / clusters;
            auto remainder = fires % clusters;
            std::set<std::pair<int, int>> used;
            for (int c = 0; c < clusters; c++) {
                int cluster_size = fires_per_cluster + (c + 1 == clusters ? remainder : 0);
                std::pair<int, int> start_point;
                int attempts = 0;
                do {
                    start_point = gridmap_->GetRandomPointInGrid();
                    attempts++;
                } while ((used.count(start_point) || !gridmap_->CellCanIgnite(start_point.first, start_point.second)) && attempts < 100000);
                IgniteFireCluster(cluster_size, start_point, used);
            }
            break;
        }
    }
}

void FireGenerator::IgniteFireCluster(int fires, std::pair<int, int> start_point, std::set<std::pair<int, int>>& used) {
    std::set<std::pair<int, int>> visited;
    std::queue<std::pair<int, int>> to_visit;
    std::vector<std::pair<int, int>> frontier; // ignited cells for re-seeding fallback
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    to_visit.push(start_point);
    visited.insert(start_point);

    int ignited = 0;
    int reseed_failures = 0;
    constexpr int kMaxReseedFailures = 200;

    while (!to_visit.empty() && ignited < fires) {
        auto current = to_visit.front();
        to_visit.pop();

        if (gridmap_->CellCanIgnite(current.first, current.second)) {
            gridmap_->IgniteCell(current.first, current.second);
            ignited++;
            frontier.push_back(current);
        }
        used.insert(current);

        auto neighbors = gridmap_->GetMooreNeighborhood(current.first, current.second);
        std::vector<std::pair<int, int>> ignitable_neighbors;

        for (auto& neighbor : neighbors) {
            if (!visited.count(neighbor) && !used.count(neighbor) &&
                gridmap_->CellCanIgnite(neighbor.first, neighbor.second)) {
                ignitable_neighbors.push_back(neighbor);
            }
        }

        // Original single-cell fallback: when queue is about to die, shuffle
        // neighbors and push just ONE to keep growing in a random tendril direction
        if (to_visit.empty() && ignited < fires && !ignitable_neighbors.empty()) {
            std::shuffle(ignitable_neighbors.begin(), ignitable_neighbors.end(), parameters_.gen_);
            to_visit.push(ignitable_neighbors.front());
            visited.insert(ignitable_neighbors.front());
            used.insert(ignitable_neighbors.front());
            continue;
        }

        // Normal probabilistic expansion — the organic shape comes from here
        for (auto& neighbor : ignitable_neighbors) {
            double randomValue = dist(parameters_.gen_);
            double fireProbability = parameters_.GetFireSpreadProb() + (dist(parameters_.gen_) - 0.5) * parameters_.GetFireNoise();
            if (randomValue < fireProbability) {
                to_visit.push(neighbor);
                visited.insert(neighbor);
                used.insert(neighbor);
            }
        }

        // Frontier re-seed: when queue is empty AND the single-cell fallback above
        // didn't fire (no ignitable neighbors on the current cell), pick a random
        // already-ignited cell and try to grow from there.
        if (to_visit.empty() && ignited < fires && reseed_failures < kMaxReseedFailures) {
            bool reseeded = false;
            if (!frontier.empty()) {
                for (int r = 0; r < 30 && !reseeded; r++) {
                    std::uniform_int_distribution<int> fIdx(0, static_cast<int>(frontier.size()) - 1);
                    auto& seed = frontier[fIdx(parameters_.gen_)];
                    auto seed_neighbors = gridmap_->GetMooreNeighborhood(seed.first, seed.second);
                    std::shuffle(seed_neighbors.begin(), seed_neighbors.end(), parameters_.gen_);
                    for (auto& n : seed_neighbors) {
                        if (!visited.count(n) && !used.count(n) && gridmap_->CellCanIgnite(n.first, n.second)) {
                            to_visit.push(n);
                            visited.insert(n);
                            reseeded = true;
                            break;
                        }
                    }
                }
            }
            if (!reseeded) {
                reseed_failures++;
                // Last resort: random map point to escape dead zones
                for (int r = 0; r < 20; r++) {
                    auto pt = gridmap_->GetRandomPointInGrid();
                    if (!visited.count(pt) && !used.count(pt) && gridmap_->CellCanIgnite(pt.first, pt.second)) {
                        to_visit.push(pt);
                        visited.insert(pt);
                        break;
                    }
                }
            }
        }
    }
}

void FireGenerator::IgniteScattered(int fires) {
    if (!gridmap_->CanStartFires(fires)) {
        std::cout << "Map is incapable of burning that much. Please choose a lower percentage." << std::endl;
        return;
    }
    int ignited = 0;
    int attempts = 0;
    constexpr int kMaxAttempts = 1000000;
    while (ignited < fires && attempts < kMaxAttempts) {
        auto point = gridmap_->GetRandomPointInGrid();
        if (gridmap_->CellCanIgnite(point.first, point.second)) {
            gridmap_->IgniteCell(point.first, point.second);
            ignited++;
        }
        attempts++;
    }
}

void FireGenerator::IgniteRing(int fires) {
    int rows = gridmap_->GetRows();
    int cols = gridmap_->GetCols();
    if (rows <= 0 || cols <= 0 || fires <= 0) return;

    int clusters = std::max(1, std::min(parameters_.num_fire_clusters_, fires));
    int fires_per_ring = fires / clusters;
    int remainder = fires % clusters;
    int total_ignited = 0;

    for (int c = 0; c < clusters; c++) {
        int ring_fires = fires_per_ring + (c + 1 == clusters ? remainder : 0);
        if (ring_fires <= 0) continue;

        // Pick a random center
        auto center = gridmap_->GetRandomPointInGrid();
        int cx = center.first;
        int cy = center.second;

        // Compute ring geometry
        double radius = std::sqrt(static_cast<double>(ring_fires) / M_PI);
        radius = std::max(radius, 2.0);
        double thickness = static_cast<double>(ring_fires) / (2.0 * M_PI * radius);
        thickness = std::max(thickness, 1.0);

        double inner_r = radius - thickness / 2.0;
        double outer_r = radius + thickness / 2.0;
        if (inner_r < 0) inner_r = 0;

        double inner_r2 = inner_r * inner_r;
        double outer_r2 = outer_r * outer_r;

        // Collect candidate cells in the annulus bounding box
        int min_x = std::max(0, cx - static_cast<int>(std::ceil(outer_r)));
        int max_x = std::min(rows - 1, cx + static_cast<int>(std::ceil(outer_r)));
        int min_y = std::max(0, cy - static_cast<int>(std::ceil(outer_r)));
        int max_y = std::min(cols - 1, cy + static_cast<int>(std::ceil(outer_r)));

        std::vector<std::pair<int, int>> candidates;
        for (int x = min_x; x <= max_x; x++) {
            for (int y = min_y; y <= max_y; y++) {
                double dx = static_cast<double>(x - cx);
                double dy = static_cast<double>(y - cy);
                double dist2 = dx * dx + dy * dy;
                if (dist2 >= inner_r2 && dist2 <= outer_r2 && gridmap_->CellCanIgnite(x, y)) {
                    candidates.emplace_back(x, y);
                }
            }
        }

        std::shuffle(candidates.begin(), candidates.end(), parameters_.gen_);
        int ignited = 0;
        for (auto& pt : candidates) {
            if (ignited >= ring_fires) break;
            gridmap_->IgniteCell(pt.first, pt.second);
            ignited++;
        }
        total_ignited += ignited;

        // Fallback to scattered for any remainder clipped by map edge
        int leftover = ring_fires - ignited;
        if (leftover > 0) {
            int attempts = 0;
            while (leftover > 0 && attempts < 100000) {
                auto point = gridmap_->GetRandomPointInGrid();
                if (gridmap_->CellCanIgnite(point.first, point.second)) {
                    gridmap_->IgniteCell(point.first, point.second);
                    leftover--;
                    total_ignited++;
                }
                attempts++;
            }
        }
    }
}
