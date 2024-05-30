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


std::string formatTime(int seconds);

enum CellState { GENERIC_UNBURNED = 0,
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
                 OUTSIDE_GRID = 16,
                 CELL_STATE_COUNT};

std::string CellStateToString(CellState cell_state);
std::optional<std::filesystem::path> find_project_root(const std::filesystem::path& start);

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

    bool isEmpty() const {
        return (!full && (head == tail));
    }

    size_t getHead() const {
        return head;
    }

    size_t getTail() const {
        return tail;
    }

    size_t size() const {
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
#endif //ROSHAN_UTILS_H
