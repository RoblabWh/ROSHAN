//
// Created by nex on 10.06.24.
//

#ifndef ROSHAN_UTILS2_H
#define ROSHAN_UTILS2_H

#include <chrono>

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

    double GetDurationInMilliseconds() const {
        std::chrono::duration<double, std::milli> duration = end_ - start_;
        return duration.count();
    }
    int GetTimeSteps() const {
        return time_steps_;
    }
    void AppendDuration(double duration) {
        durations_.push_back(duration);
    }
    std::vector<double> GetDurations() const {
        return durations_;
    }
    double GetAverageDuration() const {
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

#endif //ROSHAN_UTILS2_H
