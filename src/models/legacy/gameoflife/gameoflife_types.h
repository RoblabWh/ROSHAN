//
// Created by nex on 08.06.23.
//

#ifndef ROSHAN_GAMEOFLIFE_TYPES_H
#define ROSHAN_GAMEOFLIFE_TYPES_H

#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace std {
    template <>
    struct hash<std::pair<int, int>> {
        size_t operator()(const std::pair<int, int>& pair) const {
            return std::hash<int>()(pair.first) ^ (std::hash<int>()(pair.second) << 1);
        }
    };
}

using Cell = std::pair<int, int>;
using CellStateGOF = std::unordered_map<Cell, bool, std::hash<Cell>>;
using Neighbors = std::vector<Cell>;

#endif //ROSHAN_GAMEOFLIFE_TYPES_H
