//
// GridVisualizer.h - Grid rendering component for ImGui
//
// Renders 2D grid data as colored rectangles with various color mapping modes.
// Extracted from firemodel_imgui.cpp DrawGrid template method.
//

#ifndef ROSHAN_GRIDVISUALIZER_H
#define ROSHAN_GRIDVISUALIZER_H

#include "imgui.h"
#include "../UITypes.h"
#include <vector>
#include <functional>
#include <algorithm>

namespace ui {

class GridVisualizer {
public:
    // Color mapping function type
    using ColorMapper = std::function<ImVec4(double)>;

    // Default cell size in pixels
    static constexpr float kDefaultCellSize = 5.0f;

    // Draw a grid with a custom color mapper
    template<typename T>
    static void Draw(const std::vector<std::vector<T>>& grid,
                     const ColorMapper& colorMapper,
                     float cellSize = kDefaultCellSize) {
        if (grid.empty()) return;

        ImVec2 cursorPos = ImGui::GetCursorScreenPos();
        ImDrawList* drawList = ImGui::GetWindowDrawList();

        for (size_t y = 0; y < grid.size(); ++y) {
            for (size_t x = 0; x < grid[y].size(); ++x) {
                ImVec4 color = colorMapper(static_cast<double>(grid[y][x]));
                ImVec2 pMin(cursorPos.x + x * cellSize, cursorPos.y + y * cellSize);
                ImVec2 pMax(cursorPos.x + (x + 1) * cellSize, cursorPos.y + (y + 1) * cellSize);

                drawList->AddRectFilled(pMin, pMax,
                    IM_COL32(static_cast<int>(color.x * 255),
                             static_cast<int>(color.y * 255),
                             static_cast<int>(color.z * 255),
                             static_cast<int>(color.w * 255)));
            }
        }
    }

    // Draw using a predefined color mode
    template<typename T>
    static void Draw(const std::vector<std::vector<T>>& grid,
                     GridColorMode mode,
                     float maxExplorationTime = 100.0f,
                     float cellSize = kDefaultCellSize) {
        ColorMapper mapper = GetColorMapper(mode, maxExplorationTime);
        Draw(grid, mapper, cellSize);
    }

    // Get a color mapper for a specific mode
    static ColorMapper GetColorMapper(GridColorMode mode, float maxExplorationTime = 100.0f) {
        switch (mode) {
            case GridColorMode::ExplorationInterpolated:
                return [maxExplorationTime](double value) -> ImVec4 {
                    float normalized = std::clamp(static_cast<float>(value) / maxExplorationTime, 0.0f, 0.6f);
                    return {0.6f - normalized, 0.6f - normalized, 0.3f, 1.0f};
                };

            case GridColorMode::FireInterpolated:
                return [](double value) -> ImVec4 {
                    return {0.0f + static_cast<float>(value), 1.0f - static_cast<float>(value), 0.0f, 1.0f};
                };

            case GridColorMode::ExplorationInterpolated2:
                return [](double value) -> ImVec4 {
                    return {0.2f, 0.0f + static_cast<float>(value), 0.2f, 1.0f};
                };

            case GridColorMode::TotalView:
                return [](double value) -> ImVec4 {
                    float normalized = std::clamp(static_cast<float>(value), 0.0f, 1.0f);
                    return {0.6f - normalized, 0.6f - normalized, 0.3f, 1.0f};
                };

            case GridColorMode::Fire:
                return [](double value) -> ImVec4 {
                    return value > 0 ? ImVec4(1.0f, 0.0f, 0.0f, 1.0f) : ImVec4(0.0f, 1.0f, 0.0f, 1.0f);
                };

            case GridColorMode::Terrain:
            default:
                // For terrain, caller should provide GetMappedColor from renderer
                return [](double /*value*/) -> ImVec4 {
                    return {1.0f, 1.0f, 1.0f, 1.0f};
                };
        }
    }

    // Draw with terrain colors using external color mapper (e.g., from FireModelRenderer)
    template<typename T>
    static void DrawTerrain(const std::vector<std::vector<T>>& grid,
                            std::function<ImVec4(int)> getMappedColor,
                            float cellSize = kDefaultCellSize) {
        Draw(grid, [&getMappedColor](double value) -> ImVec4 {
            return getMappedColor(static_cast<int>(value));
        }, cellSize);
    }
};

} // namespace ui

#endif // ROSHAN_GRIDVISUALIZER_H
