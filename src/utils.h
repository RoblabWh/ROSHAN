//
// Created by nex on 10.06.24.
//

#ifndef ROSHAN_UTILS2_H
#define ROSHAN_UTILS2_H

#include <chrono>
#include <unordered_set>
#include <functional>
#include <memory>
#include <deque>
#include <utility>
#include "state.h"

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

// --- Enums & helpers ---

enum Mode { GUI_RL, GUI, NoGUI_RL, NoGUI };
enum class TerminationKind : uint8_t { None, Failed, Succeeded };

// Maybe allow multi failure reasons in the future, but for now we keep it simple
enum class FailureReason : uint8_t { None, Timeout, BoundaryExit, Burnout, Collision, Stuck, NoProgress, Other };

// --- Per-agent terminal payload ---
struct AgentTerminal {
    bool is_terminal{false};
    TerminationKind kind{TerminationKind::None};
    FailureReason reason{FailureReason::None};   // meaningful iff kind==Failed
};

// --- Episode summary ---
struct EpisodeSummary {
    bool env_reset{false};
    bool any_failed{false};
    bool any_succeeded{false};
    FailureReason reason{FailureReason::None};
    // These two are no longer needed, but this would be the place to add them when you would change the behavior
    // from ONE agent fails/succeeds to ALL agents must fail/succeed -> but this requires huge changes in alot of
    // places. Python-side we deal with irregular tensor-shapes which is a hassle and I don't want to deal with that
//    bool all_failed{true};
//    bool all_succeeded{true};
};

// --- Whole step result ---
using Observations =
        std::unordered_map<std::string, std::vector<std::deque<std::shared_ptr<State>>>>;

struct StepResult {
    Observations observations;
    std::vector<double> rewards;        // aligned with agents vector
    std::vector<AgentTerminal> terminals; // aligned with agents vector
    EpisodeSummary summary;
    double percent_burned{0.0};
};
#endif //ROSHAN_UTILS2_H
