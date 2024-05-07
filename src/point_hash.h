//
// Created by nex on 11.06.23.
//

#ifndef ROSHAN_POINT_HASH_H
#define ROSHAN_POINT_HASH_H

#include "point.h"
#include <functional>

namespace std {
    template <>
    struct hash<Point> {
        size_t operator()(const Point& p) const {
            return p.x_ ^ (p.y_ << 1);
        }
    };
}

#endif //ROSHAN_POINT_HASH_H
