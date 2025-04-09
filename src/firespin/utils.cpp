#include "utils.h"

std::string CellStateToString(CellState cell_state) {
    switch (cell_state) {
        case GENERIC_UNBURNED:
            return "GENERIC_UNBURNED";
        case SEALED:
            return "SEALED";
        case WOODY_NEEDLE_LEAVED_TREES:
            return "WOODY_NEEDLE_LEAVED_TREES";
        case WOODY_BROADLEAVED_DECIDUOUS_TREES:
            return "WOODY_BROADLEAVED_DECIDUOUS_TREES";
        case WOODY_BROADLEAVED_EVERGREEN_TREES:
            return "WOODY_BROADLEAVED_EVERGREEN_TREES";
        case LOW_GROWING_WOODY_PLANTS:
            return "LOW_GROWING_WOODY_PLANTS";
        case PERMANENT_HERBACEOUS:
            return "PERMANENT_HERBACEOUS";
        case PERIODICALLY_HERBACEOUS:
            return "PERIODICALLY_HERBACEOUS";
        case LICHENS_AND_MOSSES:
            return "LICHENS_AND_MOSSES";
        case NON_AND_SPARSLEY_VEGETATED:
            return "NON_AND_SPARSLEY_VEGETATED";
        case WATER:
            return "WATER";
        case SNOW_AND_ICE:
            return "SNOW_AND_ICE";
        case OUTSIDE_AREA:
            return "OUTSIDE_AREA";
        case GENERIC_BURNING:
            return "GENERIC_BURNING";
        case GENERIC_BURNED:
            return "GENERIC_BURNED";
        case GENERIC_FLOODED:
            return "GENERIC_FLOODED";
        default:
            return "UNKNOWN";
    }
}

std::string formatTime(int total_seconds) {
    const int seconds_per_minute = 60;
    const int minutes_per_hour = 60;
    const int hours_per_day = 24;

    int days = total_seconds / (hours_per_day * minutes_per_hour * seconds_per_minute);
    int hours = (total_seconds / (minutes_per_hour * seconds_per_minute)) % hours_per_day;
    int minutes = (total_seconds / seconds_per_minute) % minutes_per_hour;
//    int seconds = total_seconds % seconds_per_minute;

    std::string formatted_time;

    if (days > 0) {
        formatted_time += std::to_string(days) + " day(s) ";
    }
    if (hours > 0 || days > 0) {
        formatted_time += std::to_string(hours) + " hour(s) ";
    }
    if (minutes > 0 || hours > 0 || days > 0) {
        formatted_time += std::to_string(minutes) + " minute(s) ";
    }
    //formatted_time += std::to_string(seconds) + " second(s)";

    return formatted_time;
}

std::optional<std::filesystem::path> find_project_root(const std::filesystem::path& start) {
    std::filesystem::path current = start;

    // Search for .git directory, which is a good indicator of a project root
    while (current.has_relative_path()) {
        if (std::filesystem::exists(current / ".git")) {
            return current;
        }
        if (current.parent_path() == current) {
            break;
        }
        current = current.parent_path();
    }
    return std::nullopt;
}

[[maybe_unused]] std::vector<std::vector<int>> PoolingResize(const std::vector<std::vector<int>>& input_map, int new_width, int new_height) {
    int old_width = input_map.size();
    int old_height = input_map[0].size();

    std::vector <std::vector<int>> resized_map(new_width, std::vector<int>(new_height, 0));

    float x_ratio = static_cast<float>(old_width) / new_width;
    float y_ratio = static_cast<float>(old_height) / new_height;

    for (int y = 0; y < new_height; ++y) {
        for (int x = 0; x < new_width; ++x) {
            int x_start = static_cast<int>(x * x_ratio);
            int x_end = static_cast<int>((x + 1) * x_ratio);
            int y_start = static_cast<int>(y * y_ratio);
            int y_end = static_cast<int>((y + 1) * y_ratio);

            int sum = 0;
            int count = 0;

            for (int yy = y_start; yy < y_end; ++yy) {
                for (int xx = x_start; xx < x_end; ++xx) {
                    if (xx < old_width && yy < old_height) {
                        sum += input_map[xx][yy];
                        count++;
                    }
                }
            }

            resized_map[x][y] = (count > 0) ? (sum / count) : 0;
        }
    }

    return resized_map;
}

std::vector<std::vector<int>> InterpolationResize(const std::vector<std::vector<int>>& input_map, int new_width, int new_height) {
    auto old_width = input_map.size();
    auto old_height = input_map[0].size();

    std::vector<std::vector<int>> resized_map(new_width, std::vector<int>(new_height, 0));

    float x_ratio = static_cast<float>(old_width) / static_cast<float>(new_width);
    float y_ratio = static_cast<float>(old_height) / static_cast<float>(new_height);

    for (int y = 0; y < new_height; ++y) {
        for (int x = 0; x < new_width; ++x) {
            float gx = static_cast<float>(x) * x_ratio;
            float gy = static_cast<float>(y) * y_ratio;
            int gxi = static_cast<int>(gx);
            int gyi = static_cast<int>(gy);
            float x_diff = gx - static_cast<float>(gxi);
            float y_diff = gy - static_cast<float>(gyi);

            // Bounds checking
            if (gxi >= old_width - 1 || gyi >= old_height - 1) {
                resized_map[x][y] = input_map[gxi][gyi];
                continue;
            }

            // Bilinear interpolation
            auto top_left = static_cast<float>(input_map[gxi][gyi]);
            auto top_right = static_cast<float>(input_map[gxi + 1][gyi]);
            auto bottom_left = static_cast<float>(input_map[gxi][gyi + 1]);
            auto bottom_right = static_cast<float>(input_map[gxi + 1][gyi + 1]);

            float top = top_left + x_diff * (top_right - top_left);
            float bottom = bottom_left + x_diff * (bottom_right - bottom_left);
            resized_map[x][y] = static_cast<int>(top + y_diff * (bottom - top));
        }
    }

    return resized_map;
}

[[maybe_unused]] std::vector<std::vector<int>> ResizeFire(const std::vector<std::vector<int>>& input_map, int new_width, int new_height) {
    int old_width = input_map.size();
    int old_height = input_map[0].size();

    std::vector<std::vector<int>> resized_map(new_width, std::vector<int>(new_height, 0));

    float x_ratio = static_cast<float>(old_width) / new_width;
    float y_ratio = static_cast<float>(old_height) / new_height;

    for (int y = 0; y < new_height; ++y) {
        for (int x = 0; x < new_width; ++x) {
            int x_orig = static_cast<int>(x * x_ratio);
            int y_orig = static_cast<int>(y * y_ratio);

            // Ensure indices are within bounds
            if (x_orig >= old_width) x_orig = old_width - 1;
            if (y_orig >= old_height) y_orig = old_height - 1;

            resized_map[x][y] = input_map[x_orig][y_orig];
        }
        }

    return resized_map;
}

std::vector<std::vector<double>> BilinearInterpolation(const std::vector<std::vector<int>>& input_map, int new_width, int new_height) {
    int inputRows = input_map.size();
    int inputCols = input_map[0].size();

    std::vector<std::vector<double>> output_map(new_width, std::vector<double>(new_height, 0.0));

    double x_scale = (inputCols - 1) / static_cast<double>(new_height - 1);
    double y_scale = (inputRows - 1) / static_cast<double>(new_width - 1);

    // Perform bilinear interpolation
    for (int i = 0; i < new_width; ++i) {
        for (int j = 0; j < new_height; ++j) {
            double y_in = i * y_scale;
            double x_in = j * x_scale;

            int x0 = static_cast<int>(std::floor(x_in));
            int x1 = static_cast<int>(std::ceil(x_in));
            int y0 = static_cast<int>(std::floor(y_in));
            int y1 = static_cast<int>(std::ceil(y_in));

            x0 = std::max(0, std::min(x0, inputCols - 1));
            x1 = std::max(0, std::min(x1, inputCols - 1));
            y0 = std::max(0, std::min(y0, inputRows - 1));
            y1 = std::max(0, std::min(y1, inputRows - 1));

            double dx = x_in - x0;
            double dy = y_in - y0;

            double val00 = input_map[y0][x0];
            double val10 = input_map[y0][x1];
            double val01 = input_map[y1][x0];
            double val11 = input_map[y1][x1];

            // Compute the interpolated value
            double value = (1 - dx) * (1 - dy) * val00
                           + dx * (1 - dy) * val10
                           + (1 - dx) * dy * val01
                           + dx * dy * val11;

            output_map[i][j] = value;
        }
    }
    return output_map;
}