//
// Created by nex on 10.12.24.
//

#ifndef ROSHAN_UTILS3_H
#define ROSHAN_UTILS3_H

#include <algorithm>
#include <cmath>

// * Discretize the output of the neural network
// * @param netout The output of the neural network
// * @param bin_size The size of the bins
// * @return The discretized output
double DiscretizeOutput(double netout, double bin_size);

#endif //ROSHAN_UTILS3_H
