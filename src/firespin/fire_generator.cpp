//
// Created by nex on 24.08.25.
//

#include "fire_generator.h"

FireGenerator::FireGenerator(std::shared_ptr<GridMap> gridmap, FireModelParameters& parameters)
        : gridmap_(std::move(gridmap)), parameters_(parameters) {}

void FireGenerator::StartFires() {
    auto cells = static_cast<float>(gridmap_->GetNumCells());
    int fires = static_cast<int>(cells * parameters_.GetFirePercentage());
    if (parameters_.num_fire_clusters_ > 0) {
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
            } while ((used.find(start_point) != used.end() || !gridmap_->CellCanIgnite(start_point.first, start_point.second)) && attempts < 1000);
            IgniteFireCluster(cluster_size, start_point, used);
        }
    }
    // TODO this is probably not needed anymore because we can simulate a similar effect with high number of clusters and low fire percentage
    else {
        if (!gridmap_->CanStartFires(fires)){
            std::cout << "Map is incapable of burning that much. Please choose a lower percentage." << std::endl;
            return;
        }
        for(int i = 0; i < fires;) {
            std::pair<int, int> point = gridmap_->GetRandomPointInGrid();
            if (gridmap_->CellCanIgnite(point.first, point.second)) {
                gridmap_->IgniteCell(point.first, point.second);
                i++;
            }
        }
    }
}

void FireGenerator::IgniteFireCluster(int fires, std::pair<int, int> start_point, std::set<std::pair<int, int>>& used) {
    std::set<std::pair<int, int>> visited;
    std::queue<std::pair<int, int>> to_visit;
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    to_visit.push(start_point);
    visited.insert(start_point);

    int ignited = 0;

    while (!to_visit.empty() && ignited < fires) {
        auto current = to_visit.front();
        to_visit.pop();

        if (gridmap_->CellCanIgnite(current.first, current.second)) {
            gridmap_->IgniteCell(current.first, current.second);
            ignited++;
        }
        used.insert(current);

        auto neighbors = gridmap_->GetMooreNeighborhood(current.first, current.second);
        std::vector<std::pair<int, int>> ignitable_neighbors;

        for (auto& neighbor : neighbors) {
            if (visited.find(neighbor) == visited.end() && used.find(neighbor) == used.end() &&
                gridmap_->CellCanIgnite(neighbor.first, neighbor.second)) {
                ignitable_neighbors.push_back(neighbor);
            }
        }

        if (to_visit.empty() && ignited < fires && !ignitable_neighbors.empty()) {
            std::shuffle(ignitable_neighbors.begin(), ignitable_neighbors.end(), parameters_.gen_);
            to_visit.push(ignitable_neighbors.front());
            visited.insert(ignitable_neighbors.front());
            used.insert(ignitable_neighbors.front());
            continue;
        }

        for (auto& neighbor : ignitable_neighbors) {
            double randomValue = dist(parameters_.gen_);
            double fireProbability = parameters_.GetFireSpreadProb() + (dist(parameters_.gen_) - 0.5) * parameters_.GetFireNoise();
            if (randomValue < fireProbability) {
                to_visit.push(neighbor);
                visited.insert(neighbor);
                used.insert(neighbor);
            }
        }
    }
}
