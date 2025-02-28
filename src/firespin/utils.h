//
// Created by nex on 12.06.23.
//

#ifndef ROSHAN_UTILS_H
#define ROSHAN_UTILS_H

#include <string>
#include <filesystem>
#include <optional>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <random>
#include <cmath>
#include <limits>

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
std::vector<std::vector<int>> PoolingResize(const std::vector<std::vector<int>>& input_map, int new_width, int new_height);
std::vector<std::vector<int>> InterpolationResize(const std::vector<std::vector<int>>& input_map, int new_width, int new_height);
std::vector<std::vector<int>> ResizeFire(const std::vector<std::vector<int>>& input_map, int new_width, int new_height);
std::vector<std::vector<double>> BilinearInterpolation(const std::vector<std::vector<int>>& input_map, int new_width, int new_height);

template <typename T>
class CircularBuffer {
public:
    explicit CircularBuffer(size_t size)
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
        if (isEmpty()) {
            throw std::runtime_error("Buffer is empty, no last element to modify");
        }
        size_t lastIndex = (head == 0 ? max_size : head) - 1;
        buffer[lastIndex] = newValue;
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

    [[nodiscard]] size_t getTail() const {
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

    const std::vector<T> getBuffer() const {
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
    RandomBuffer(size_t size) : index_(0) {
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
    bool hasEnough(size_t n) {
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

#endif //ROSHAN_UTILS_H
