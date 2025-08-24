//
// Created by nex on 24.08.25.
//

#ifndef ROSHAN_FIRE_GENERATOR_H
#define ROSHAN_FIRE_GENERATOR_H

#include <memory>
#include <queue>
#include <set>
#include <vector>
#include <random>
#include <algorithm>
#include <iostream>
#include "firemodel_gridmap.h"
#include "model_parameters.h"

class FireGenerator {
public:
    FireGenerator(std::shared_ptr<GridMap> gridmap, FireModelParameters& parameters);

    void StartFires();

private:
    void IgniteFireCluster(int fires, std::pair<int, int> start_point, std::set<std::pair<int, int>>& used);

    std::shared_ptr<GridMap> gridmap_;
    FireModelParameters& parameters_;
};

#endif // ROSHAN_FIRE_GENERATOR_H