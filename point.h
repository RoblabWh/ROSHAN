//
// Created by nex on 11.06.23.
//

#ifndef ROSHAN_POINT_H
#define ROSHAN_POINT_H

#include <unordered_set>

class Point {

public:
    Point(int x, int y) {
        x_ = x;
        y_ = y;
    }

    bool operator==(const Point& other) const {
        return x_ == other.x_ && y_ == other.y_;
    }

    int x_;
    int y_;
};


#endif //ROSHAN_POINT_H
