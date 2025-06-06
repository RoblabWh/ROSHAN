//
// Created by nex on 10.06.24.
//

#ifndef ROSHAN_UTILS2_H
#define ROSHAN_UTILS2_H

#include <chrono>
#include <unordered_set>
#include <functional>
#include <memory>

enum Mode {
    GUI_RL = 0,
    GUI = 1,
    NoGUI_RL = 2,
    NoGUI = 3
};

class Timer {
public:
    void Start() {
        start_ = std::chrono::high_resolution_clock::now();
    }

    void Stop() {
        end_ = std::chrono::high_resolution_clock::now();
        time_steps_ -= 1;
    }

    [[nodiscard]] double GetDurationInMilliseconds() const {
        std::chrono::duration<double, std::milli> duration = end_ - start_;
        return duration.count();
    }
    [[nodiscard]] int GetTimeSteps() const {
        return time_steps_;
    }
    void AppendDuration(double duration) {
        durations_.push_back(duration);
    }
    [[nodiscard]] std::vector<double> GetDurations() const {
        return durations_;
    }
    [[nodiscard]] double GetAverageDuration() const {
        double sum = 0;
        for (double duration : durations_) {
            sum += duration;
        }
        return sum / durations_.size();
    }

private:
    int time_steps_ = 10;
    std::vector<double> durations_;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
    std::chrono::time_point<std::chrono::high_resolution_clock> end_;
};

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

template <typename Derived, typename Base>
std::vector<std::shared_ptr<Derived>> CastAgents(const std::vector<std::shared_ptr<Base>>& base_agents) {
    std::vector<std::shared_ptr<Derived>> result;
    for (const auto& base : base_agents) {
        if (auto derived = std::dynamic_pointer_cast<Derived>(base)) {
            result.push_back(derived);
        }
    }
    return result;
}

namespace std {
    template <>
    struct hash<Point> {
        size_t operator()(const Point& p) const {
            return p.x_ ^ (p.y_ << 1);
        }
    };
}

#endif //ROSHAN_UTILS2_H
