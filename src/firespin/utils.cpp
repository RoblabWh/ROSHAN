#include "utils.h"
#include <numeric>

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
    int old_width = static_cast<int>(input_map.size());
    int old_height = static_cast<int>(input_map[0].size());

    std::vector<std::vector<int>> resized_map(new_width, std::vector<int>(new_height, 0));

    // Flatten the input map for cache-friendly access
    std::vector<int> flat(old_width * old_height);
    for (int x = 0; x < old_width; ++x) {
        std::copy(input_map[x].begin(), input_map[x].end(), flat.begin() + x * old_height);
    }

    float x_ratio = static_cast<float>(old_width) / static_cast<float>(new_width);
    float y_ratio = static_cast<float>(old_height) / static_cast<float>(new_height);

    // Precompute region boundaries
    std::vector<int> x_bounds(new_width + 1);
    std::vector<int> y_bounds(new_height + 1);
    for (int i = 0; i <= new_width; ++i) {
        x_bounds[i] = static_cast<int>(i * x_ratio);
    }
    for (int i = 0; i <= new_height; ++i) {
        y_bounds[i] = static_cast<int>(i * y_ratio);
    }

    for (int x = 0; x < new_width; ++x) {
        int xs = x_bounds[x];
        int xe = x_bounds[x + 1];
        for (int y = 0; y < new_height; ++y) {
            int ys = y_bounds[y];
            int ye = y_bounds[y + 1];

            int sum = 0;
            int elements = (xe - xs) * (ye - ys);

            if (elements > 0) {
                int *x_ptr = flat.data() + xs * old_height;
                for (int xx = xs; xx < xe; ++xx, x_ptr += old_height) {
                    sum += std::accumulate(x_ptr + ys, x_ptr + ye, 0);
                }
                resized_map[x][y] = sum / elements;
            } else {
                resized_map[x][y] = 0;
            }
        }
    }

    return resized_map;
}

std::vector<std::vector<int>> InterpolationResize(const std::vector<std::vector<int>>& input_map, int new_width, int new_height) {
    int old_width = static_cast<int>(input_map.size());
    int old_height = static_cast<int>(input_map[0].size());

    std::vector<std::vector<int>> resized_map(new_width, std::vector<int>(new_height, 0));

    // Flatten the input for pointer arithmetic
    std::vector<int> flat(old_width * old_height);
    for (int x = 0; x < old_width; ++x) {
        std::copy(input_map[x].begin(), input_map[x].end(), flat.begin() + x * old_height);
    }

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

            int idx = gxi * old_height + gyi;

            // Bounds checking
            if (gxi >= old_width - 1 || gyi >= old_height - 1) {
                resized_map[x][y] = flat[idx];
                continue;
            }

            // Bilinear interpolation using pointer offsets
            auto top_left = static_cast<float>(flat[idx]);
            auto top_right = static_cast<float>(flat[idx + old_height]);
            auto bottom_left = static_cast<float>(flat[idx + 1]);
            auto bottom_right = static_cast<float>(flat[idx + old_height + 1]);

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

std::vector<std::deque<std::pair<double, double>>>
GeneratePaths(
        int width, int height,
        int num_drones,
        std::pair<double, double> start,
        int view_range
) {
    std::vector<std::deque<std::pair<double, double>>> paths(num_drones);

    bool split_vertical = (width >= height);

    bool x_inc = (start.first < static_cast<double>(width) / 2);   // Sweep right if true
    bool y_inc = (start.second < static_cast<double>(height) / 2); // Sweep down if true

    if (split_vertical) {
        // Vertical stripes
        int stripe_w = width / num_drones;
        int x0 = 0;

        for (int i = 0; i < num_drones; ++i) {
            int x1 = x0 + stripe_w;
            if (x1 > width) x1 = width;

            int x_step = std::max(1, stripe_w / static_cast<int>(std::ceil((float)stripe_w / (static_cast<float>(view_range) - 1))));
            int y_step = std::max(1, height / static_cast<int>(std::ceil((float)height / (static_cast<float>(view_range) - 1))));

            int start_y = y_inc ? 0 : height;
            int end_y = y_inc ? height : 0;
            int y_step_dir = y_inc ? y_step : -y_step;

            bool forward = x_inc;
            for (int y = start_y; (y_inc ? y <= end_y : y >= end_y); y += y_step_dir) {
                if (forward) {
                    for (int x = x0; x <= x1; x += x_step) {
                        paths[i].emplace_back(x, y);
                    }
                    if ((x1 - x0) % x_step != 0 && x1 - 1 >= x0) {
                        paths[i].emplace_back(x1 - 1, y);  // Right edge
                    }
                } else {
                    for (int x = x1; x >= x0; x -= x_step) {
                        paths[i].emplace_back(x, y);
                    }
                    if ((x1 - x0) % x_step != 0 && x0 <= x1 - 1) {
                        paths[i].emplace_back(x0, y);  // Left edge
                    }
                }
                forward = !forward;
            }

            x0 = x1;
        }
    }
    else {
        // Horizontal stripes
        int stripe_h = height / num_drones;
        int y0 = 0;

        for (int i = 0; i < num_drones; ++i) {
            int y1 = y0 + stripe_h;
            if (y1 > height) y1 = height;

            int y_step = std::max(1, stripe_h / static_cast<int>(std::ceil((float)stripe_h / (static_cast<float>(view_range) - 1))));
            int x_step = std::max(1, width / static_cast<int>(std::ceil((float)width / (static_cast<float>(view_range) - 1))));

            int start_x = x_inc ? 0 : width;
            int end_x = x_inc ? width : 0;
            int x_step_dir = x_inc ? x_step : -x_step;

            bool forward = y_inc;
            for (int x = start_x; (x_inc ? x <= end_x : x >= end_x); x += x_step_dir) {
                if (forward) {
                    for (int y = y0; y <= y1; y += y_step) {
                        paths[i].emplace_back(x, y);
                    }
                    if ((y1 - y0) % y_step != 0 && y1 - 1 >= y0) {
                        paths[i].emplace_back(x, y1 - 1);  // Bottom edge
                    }
                } else {
                    for (int y = y1; y >= y0; y -= y_step) {
                        paths[i].emplace_back(x, y);
                    }
                    if ((y1 - y0) % y_step != 0 && y0 <= y1 - 1) {
                        paths[i].emplace_back(x, y0);  // Top edge
                    }
                }
                forward = !forward;
            }

            y0 = y1;
        }
    }

    return paths;
}

std::filesystem::path get_project_path(const std::string &config_key, const std::vector<std::string> &extensions) {
    auto start_path = std::filesystem::current_path();
    auto project_root = find_project_root(start_path);
    if (!project_root) {
        std::cerr << "Project root not found. OOPSI! This should generally not happen." << std::endl;
        return "";
    }
    std::filesystem::path config_path = *project_root / "project_paths.json";
    if (!std::filesystem::exists(config_path)) {
        std::cerr << "project_paths.json not found at: " << config_path << std::endl;
        return "";
    }
    std::ifstream config_file(config_path);
    nlohmann::json config;
    config_file >> config;

    // Check if "config_key" is present in the config
    if (!config.contains(config_key)) {
        std::cerr << "Config key '" << config_key << "' not found in project_paths.json." << std::endl;
        return "";
    }
    std::filesystem::path project_path = config[config_key];
    if (!project_path.is_absolute()) {
        project_path = *project_root / project_path;
    }
    std::string extension = project_path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
    if (!extensions.empty() &&
        std::find(extensions.begin(), extensions.end(), extension) == extensions.end()) {
        std::cerr << "Invalid file extension '" << extension << "'. Expected one of: ";
        for (const auto &ext : extensions) {
            std::cerr << ext << " ";
        }
        std::cerr << std::endl;
        return "";
    }

    return project_path;
}

void RandomBuffer::fillBuffer(){
    std::normal_distribution<double> dist(0.0, 1.0);
    for (auto &num : buffer_) {
        num = dist(parameters_.gen_);
    }
}
