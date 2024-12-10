//
// Created by nex on 10.12.24.
//

#include "utils.h"

double DiscretizeOutput(double netout, double bin_size) {
    double clamped = std::clamp(netout, -1.0, 1.0);
    double discrete = std::round(clamped / bin_size) * bin_size;

    return discrete;
}