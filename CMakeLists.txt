cmake_minimum_required(VERSION 3.25)
project(ROSHAN)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set the build path
set(MY_CPP_BUILD_PATH "${CMAKE_BINARY_DIR}")

# Enable optimizations
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -flto -funroll-loops -ffast-math")
# Enable faster-build optimizations
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
# Development flags
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")


# Generate a JSON configuration file
file(WRITE "config.json" "{\n\"module_directory\": \"${MY_CPP_BUILD_PATH}\"\n}\n")

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/externals/pcg-cpp-0.98/include)
include_directories(${CMAKE_SOURCE_DIR}/externals/imgui)
include_directories(${CMAKE_SOURCE_DIR}/externals/imgui/backends)
include_directories(${CMAKE_SOURCE_DIR}/externals/nlohmann)
include_directories(/usr/local/include/SDL2)
include_directories(/usr/include/python3.9)

# Set CMake module path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

# Add subdirectory for pybind11
add_subdirectory(externals/pybind11)
set(PYTHON_EXECUTABLE "$ENV{CONDA_PREFIX}/bin/python")

# Find and link against OpenGL
find_package(OpenGL REQUIRED)

# Add the imgui library
file(GLOB IMGUI_SOURCES ${CMAKE_SOURCE_DIR}/externals/imgui/*.cpp)
add_library(imgui_external ${IMGUI_SOURCES}
        externals/imgui/backends/imgui_impl_sdl2.cpp
        externals/imgui/backends/imgui_impl_opengl3.cpp
        externals/imgui/backends/imgui_impl_opengl3.h
        externals/imgui/backends/imgui_impl_sdl2.h
        externals/imgui/backends/imgui_impl_sdlrenderer2.cpp)
target_include_directories(imgui_external PUBLIC ${CMAKE_SOURCE_DIR}/externals/imgui ${CMAKE_SOURCE_DIR}/externals/imgui/backends/)

# GDAL
find_package(GDAL REQUIRED)
include_directories(${GDAL_INCLUDE_DIRS})

# SDL2 libraries
list(APPEND CMAKE_PREFIX_PATH "/usr/local")
set(SDL2_LIBRARIES /usr/local/lib/libSDL2.so)
set(SDL2_IMAGE_LIBRARIES "/usr/local/lib/libSDL2_image.so")

# Core sources
set(CORE_SOURCES
        src/main.cpp
        src/engine_core.cpp src/engine_core.h
        src/corine/dataset_handler.cpp src/corine/dataset_handler.h
        externals/ImGuiFileDialog/ImGuiFileDialog.cpp externals/ImGuiFileDialog/ImGuiFileDialog.h
        src/state.h
        src/action.h
        src/model_interface.h
        src/point.h
        src/point_hash.h
        src/utils.h)

# Agent sources
set(AGENT_SOURCES
        src/reinforcementlearning/drone_agent/drone.cpp src/reinforcementlearning/drone_agent/drone.h
        src/reinforcementlearning/drone_agent/rendering/DroneRenderer.cpp src/reinforcementlearning/drone_agent/rendering/DroneRenderer.h
        src/reinforcementlearning/drone_agent/drone_state.cpp src/reinforcementlearning/drone_agent/drone_state.h
        src/reinforcementlearning/drone_agent/drone_action.h
        src/agent.h
        src/reinforcementlearning/reinforcementlearning_handler.cpp
        src/reinforcementlearning/reinforcementlearning_handler.h)

# Fire model sources
set(FIREMODEL_SOURCES
        src/models/firespin/firemodel.cpp src/models/firespin/firemodel.h
        src/models/firespin/firemodel_gridmap.cpp src/models/firespin/firemodel_gridmap.h
        src/models/firespin/rendering/firemodel_renderer.cpp src/models/firespin/rendering/firemodel_renderer.h
        src/models/firespin/rendering/firemodel_camera.cpp src/models/firespin/rendering/firemodel_camera.h
        src/models/firespin/firemodel_firecell.cpp src/models/firespin/firemodel_firecell.h
        src/models/firespin/particles/virtual_particle.cpp src/models/firespin/particles/virtual_particle.h
        src/models/firespin/particles/radiation_particle.cpp src/models/firespin/particles/radiation_particle.h
        src/models/firespin/wind.cpp src/models/firespin/wind.h
        src/models/firespin/utils.h
        src/models/firespin/model_parameters.h
        src/models/firespin/firemodel_cell_interface.h
        src/models/firespin/cell_classes/cell_generic_flooded.cpp src/models/firespin/cell_classes/cell_generic_flooded.h
        src/models/firespin/cell_classes/cell_generic_unburned.cpp src/models/firespin/cell_classes/cell_generic_unburned.h
        src/models/firespin/cell_classes/cell_sealed.cpp src/models/firespin/cell_classes/cell_sealed.h
        src/models/firespin/cell_classes/cell_woody_needle_leaved_trees.cpp src/models/firespin/cell_classes/cell_woody_needle_leaved_trees.h
        src/models/firespin/cell_classes/cell_woody_breadleaved_deciduous_trees.cpp src/models/firespin/cell_classes/cell_woody_breadleaved_deciduous_trees.h
        src/models/firespin/cell_classes/cell_woody_broadleaved_evergreen_trees.cpp src/models/firespin/cell_classes/cell_woody_broadleaved_evergreen_trees.h
        src/models/firespin/cell_classes/cell_low_growing_woody_plants.cpp src/models/firespin/cell_classes/cell_low_growing_woody_plants.h
        src/models/firespin/cell_classes/cell_permanent_herbaceous.cpp src/models/firespin/cell_classes/cell_permanent_herbaceous.h
        src/models/firespin/cell_classes/cell_periodically_herbaceous.cpp src/models/firespin/cell_classes/cell_periodically_herbaceous.h
        src/models/firespin/cell_classes/cell_lichens_and_mosses.cpp src/models/firespin/cell_classes/cell_lichens_and_mosses.h
        src/models/firespin/cell_classes/cell_non_and_sparsley_vegetated.cpp src/models/firespin/cell_classes/cell_non_and_sparsley_vegetated.h
        src/models/firespin/cell_classes/cell_water.cpp src/models/firespin/cell_classes/cell_water.h
        src/models/firespin/cell_classes/cell_snow_and_ice.cpp src/models/firespin/cell_classes/cell_snow_and_ice.h
        src/models/firespin/cell_classes/cell_generic_burning.cpp src/models/firespin/cell_classes/cell_generic_burning.h
        src/models/firespin/cell_classes/cell_outside_area.cpp src/models/firespin/cell_classes/cell_outside_area.h
        src/models/firespin/cell_classes/cell_generic_burned.cpp src/models/firespin/cell_classes/cell_generic_burned.h
        src/models/firespin/rendering/firemodel_pixelbuffer.cpp src/models/firespin/rendering/firemodel_pixelbuffer.h
        src/models/firespin/utils.cpp
        src/models/firespin/rendering/firemodel_imgui.cpp src/models/firespin/rendering/firemodel_imgui.h
)

# Python bindings
pybind11_add_module(firesim src/python_bindings.cpp
        ${CORE_SOURCES}
        ${AGENT_SOURCES}
        ${FIREMODEL_SOURCES}
        ${IMGUI_SOURCES})
target_link_libraries(firesim PRIVATE ${SDL2_LIBRARIES} ${GDAL_LIBRARIES} ${SDL2_IMAGE_LIBRARIES} ${X11_LIBRARIES} OpenGL::GL imgui_external)
target_include_directories(firesim PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/externals/imgui ${CMAKE_SOURCE_DIR}/externals/imgui/backends/ /usr/local/include/SDL2 ${GDAL_INCLUDE_DIRS})

# Main executable
add_executable(ROSHAN
        ${CORE_SOURCES}
        ${AGENT_SOURCES}
        ${FIREMODEL_SOURCES})

target_include_directories(ROSHAN PRIVATE ${CMAKE_SOURCE_DIR}/externals/nlohmann)
target_include_directories(ROSHAN PRIVATE ${CMAKE_SOURCE_DIR})

# Link against libraries
target_link_libraries(ROSHAN ${SDL2_LIBRARIES} ${SDL2_IMAGE_LIBRARIES} ${X11_LIBRARIES} OpenGL::GL imgui_external ${GDAL_LIBRARIES} -ldl -lpython3.9)