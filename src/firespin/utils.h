//
// Created by nex on 12.06.23.
//

#ifndef ROSHAN_UTILS_H
#define ROSHAN_UTILS_H

#include <string>
#include <deque>
#include <filesystem>
#include <optional>
#include <utility>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <random>
#include <cmath>
#include <limits>
#include <cmath>
#include <fstream>
#include <regex>
#include "json.hpp"

std::string formatTime(int seconds);

enum CellState { OUTSIDE_GRID = 0,
                 SEALED = 1,
                 WOODY_NEEDLE_LEAVED_TREES = 2,
                 WOODY_BROADLEAVED_DECIDUOUS_TREES = 3,
                 WOODY_BROADLEAVED_EVERGREEN_TREES = 4,
                 LOW_GROWING_WOODY_PLANTS = 5,
                 PERMANENT_HERBACEOUS = 6,
                 PERIODICALLY_HERBACEOUS = 7,
                 LICHENS_AND_MOSSES = 8,
                 NON_AND_SPARSLEY_VEGETATED = 9,
                 WATER = 10,
                 SNOW_AND_ICE = 11,
                 GENERIC_BURNING = 12,
                 GENERIC_BURNED = 13,
                 GENERIC_FLOODED = 14,
                 OUTSIDE_AREA = 15,
                 GENERIC_UNBURNED = 16,
                 CELL_STATE_COUNT};

std::string CellStateToString(CellState cell_state);
std::optional<std::filesystem::path> find_project_root(const std::filesystem::path& start);

[[maybe_unused]] std::vector<std::vector<int>> PoolingResize(const std::vector<std::vector<int>>& input_map, int new_width, int new_height);
std::vector<std::vector<int>> InterpolationResize(const std::vector<std::vector<int>>& input_map, int new_width, int new_height);

[[maybe_unused]] std::vector<std::vector<int>> ResizeFire(const std::vector<std::vector<int>>& input_map, int new_width, int new_height);
std::vector<std::vector<double>> BilinearInterpolation(const std::vector<std::vector<int>>& input_map, int new_width, int new_height);

std::deque<std::pair<double, double>> GetPerfectTrajectory(int height, int width, std::pair<double, double> start, int view_range, bool horizontal);
std::deque<std::pair<double, double>> GenerateLawnmowerPath(
        double x_start, double x_end,
        double y_start, double y_end,
        double step_x, double step_y,
        bool horizontal_sweep,
        bool invert_major,
        bool invert_minor
) ;
std::vector<std::deque<std::pair<double, double>>>
GeneratePaths(
        int width, int height,
        int num_drones,
        std::pair<double, double> start,
        int view_range
);

std::filesystem::path get_project_path(const std::string& config_key, const std::vector<std::string>& extensions);

template <typename T>
class CircularBuffer {
public:
    [[maybe_unused]] explicit CircularBuffer(size_t size)
            : buffer(size), max_size(size), head(0), tail(0), full(false) {}

    void put(T item) {
        buffer[head] = item;
        if (full) {
            if (++tail == max_size) {
                tail = 0;
            }
        }
        // use instead of: head = (head + 1) % max_size;
        if (++head == max_size) {
            head = 0;
        }
        full = head == tail;
    }

    void ModifyLast(T newValue) {
        size_t lastIndex;
        if (isEmpty()) {
            lastIndex = head;
        } else {
            lastIndex = (head == 0 ? max_size : head) - 1;
        }
        buffer[lastIndex] += newValue;
    }

    T get() {
        if (isEmpty()) {
            throw std::runtime_error("Buffer is empty");
        }

        auto val = buffer[tail];
        full = false;
        tail = (tail + 1) % max_size;
        return val;
    }

    void reset() {
        head = tail;
        full = false;
    }

    [[nodiscard]] bool isEmpty() const {
        return (!full && (head == tail));
    }

    [[nodiscard]] size_t getHead() const {
        return head;
    }

    [[maybe_unused]] [[nodiscard]] size_t getTail() const {
        return tail;
    }

    [[nodiscard]] size_t size() const {
        size_t size = max_size;

        if (!full) {
            if (head >= tail) {
                size = head - tail;
            } else {
                size = max_size + head - tail;
            }
        }

        return size;
    }

    std::vector<T> getBuffer() const {
        std::vector<T> data(size());
        if (isEmpty()) {
            return data;
        }

        if (full) {
            std::copy(buffer.begin(), buffer.end(), data.begin());
        } else {
            if (head >= tail) {
                std::copy(buffer.begin() + tail, buffer.begin() + head, data.begin());
            } else {
                std::copy(buffer.begin() + tail, buffer.end(), data.begin());
                std::copy(buffer.begin(), buffer.begin() + head, data.begin() + (max_size - tail));
            }
        }

        return data;
    }

private:
    std::vector<T> buffer;
    size_t head;
    size_t tail;
    const size_t max_size;
    bool full;
};

class RandomBuffer {
public:
    explicit RandomBuffer(size_t size) : index_(0) {
        buffer_.resize(size);

        fillBuffer();
    }

    void fillBuffer(){
        // TODO PCG is faster, but not by very much
        //std::minstd_rand rng(std::random_device{}());
        std::mt19937 rng(std::random_device{}());
//        pcg32 rng;
//        pcg_extras::seed_seq_from<std::random_device> seed_source;
//        rng.seed(seed_source);
        std::normal_distribution<double> dist(0.0, 1.0);
        for (auto &num : buffer_) {
            num = dist(rng);
        }
    }

    // Check if there are enough numbers from the current index to the end of the buffer
    [[maybe_unused]] bool hasEnough(size_t n) {
        return buffer_.size() - index_ > n;
    }

    double getNext() {
        // Shuffle buffer if end is reached
//        if (index_ >= buffer_.size()) {
//            std::shuffle(buffer_.begin(), buffer_.end(), std::mt19937(std::random_device()()));
//            index_ = 0;
//        }
//        // Wrap around if buffer end is reached
        if (index_ >= buffer_.size()) index_ = 0;
        return buffer_[index_++];
    }

private:
    std::vector<double> buffer_;
    size_t index_;
};

class LogReader {
public:
    LogReader() = default;
    explicit LogReader(std::string path = "") : log_path(std::move(path)), last_pos(0) {}

    std::vector<std::string> readNewLines() {
        std::regex log_entry_start(R"(^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} -)");
        std::vector<std::string> lines;
        std::ifstream file(log_path, std::ios::in);
        if (!file.is_open()) return lines;

        file.seekg(last_pos);
        std::streampos last_good_pos = file.tellg();  // before reading
        std::string line;

        while (std::getline(file, line)) {
            // Cut of Date and Miliseconds
            if (std::regex_search(line, log_entry_start)) {
                // If the line starts with a timestamp, we can process it
                size_t space_pos = line.find(' ');
                if (space_pos != std::string::npos) {
                    line = line.substr(space_pos + 1);  // remove date
                }

                size_t comma_pos = line.find(',');
                if (comma_pos != std::string::npos) {
                    line = line.substr(0, comma_pos) + line.substr(line.find(' ', comma_pos));  // Keep time up to comma, skip to first "-"
                }
            }


            lines.push_back(line);
            last_good_pos = file.tellg();  // update last good position
        }

        if (!lines.empty() && last_good_pos != std::streampos(-1)){
            last_pos = last_good_pos;  // update last position to the end of the read lines
        }

        return lines;
    }

    void set_model_path(std::string& path) {
        if(path != log_path) {
            log_path = path;
            reset();
        }
    }

    void reset() {
        last_pos = 0;
    }

private:
    std::string log_path;
    std::streampos last_pos;
    std::streampos last_pos_read_ = 0;
};

#endif //ROSHAN_UTILS_H
