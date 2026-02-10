cmake_minimum_required(VERSION 3.25)
project(ROSHAN LANGUAGES CXX)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Set the build path
set(MY_CPP_BUILD_PATH "${CMAKE_BINARY_DIR}")

# Set CMAKE Module Path for FindSDL2_image.cmake
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

# Build type: Release for optimized builds, Debug for development
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -flto -funroll-loops \
            -fno-omit-frame-pointer -ffunction-sections -fdata-sections")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-O2 -Wl,--as-needed -Wl,--gc-sections")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-O2 -Wl,--as-needed -Wl,--gc-sections")
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -g")
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
endif()

find_program(CCACHE_PROGRAM ccache)
if (CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
endif()

# Third parties
find_package(SDL2 REQUIRED)                      # provides SDL2::SDL2
find_package(SDL2_image REQUIRED)                # provides SDL2_image::SDL2_image
find_package(OpenGL REQUIRED)                    # provides OpenGL::GL
find_package(CURL REQUIRED)                      # provides CURL::libcurl (newer CMake) or use ${CURL_LIBRARIES}
find_package(GDAL REQUIRED)                      # may give GDAL::GDAL if recent, else vars
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)

if (Python3_FOUND)
    message(STATUS "Found Python: ${Python3_EXECUTABLE} (found version ${Python3_VERSION})")
    include_directories(${Python3_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "Python3 not found")
endif()

if (SDL2_FOUND AND SDL2_IMAGE_FOUND)
    message(STATUS "SDL2 Include Directories: ${SDL2_INCLUDE_DIRS}")
    message(STATUS "SDL2 Image Include Directories: ${SDL2_IMAGE_INCLUDE_DIR}")
    message(STATUS "SDL2 Libraries: ${SDL2_LIBRARIES}")
    message(STATUS "SDL2 Image Libraries: ${SDL2_IMAGE_LIBRARY}")
    include_directories(${SDL2_INCLUDE_DIRS} ${SDL2_IMAGE_INCLUDE_DIR})
    link_libraries(${SDL2_LIBRARIES} ${SDL2_IMAGE_LIBRARY})
else()
    message(FATAL_ERROR "SDL2 or SDL2_image not found")
endif()

add_subdirectory(externals/pybind11)            # pybind11::module etc.
add_subdirectory(externals/yaml-cpp)            # yaml-cpp target

# ImGui
file(GLOB IMGUI_SOURCES CONFIGURE_DEPENDS
        ${CMAKE_SOURCE_DIR}/externals/imgui/*.cpp
)
add_library(imgui_external ${IMGUI_SOURCES}
        externals/imgui/backends/imgui_impl_sdl2.cpp
        externals/imgui/backends/imgui_impl_opengl3.cpp
        externals/imgui/backends/imgui_impl_opengl3.h
        externals/imgui/backends/imgui_impl_sdl2.h
        externals/imgui/backends/imgui_impl_sdlrenderer2.cpp
        externals/ImGuiFileDialog/ImGuiFileDialog.cpp
)
target_include_directories(imgui_external PUBLIC
        ${CMAKE_SOURCE_DIR}/externals/imgui
        ${CMAKE_SOURCE_DIR}/externals/imgui/backends
)
# ImGui backends include SDL/OpenGL headers; surface those requirements:
target_link_libraries(imgui_external PUBLIC
        ${SDL2_LIBRARIES}
        ${SDL2_IMAGE_LIBRARY}
        OpenGL::GL
)

# Populate the paths config file if it does not exist
get_filename_component(PARENT_DIR "${MY_CPP_BUILD_PATH}" DIRECTORY)
set(CONFIG_DIR  "${PARENT_DIR}")
set(CONFIG_FILE "${CONFIG_DIR}/project_paths.json")

set(ROOT_PATH "${PARENT_DIR}")
set(MODULE_DIRECTORY "${MY_CPP_BUILD_PATH}")
set(MAPS_DIRECTORY "${PARENT_DIR}/maps")
set(OSM_DATA "${PARENT_DIR}/openstreetmap/data.json")
set(DATASET_PATH "${CMAKE_SOURCE_DIR}/assets/dataset/") # Default path for dataset_directory

set(LOCK_FILE "${CONFIG_DIR}/.project_paths.json.lock")
set(TMP_FILE  "${CONFIG_DIR}/.project_paths.json.tmp")
set(GEN_FILE  "${CMAKE_BINARY_DIR}/project_paths.json.gen")  # staging inside build dir

configure_file(
        "${CMAKE_SOURCE_DIR}/project_paths.json.in"
        "${GEN_FILE}"
        @ONLY NEWLINE_STYLE UNIX
)

file(READ "${GEN_FILE}" _json)
# This errors out if JSON is malformed (stops configure early).
string(JSON _check GET "${_json}" root_path)

file(LOCK "${LOCK_FILE}" GUARD PROCESS TIMEOUT 30)
# Copy to TMP next to the final file, then rename = atomic on most filesystems.
file(COPY_FILE "${GEN_FILE}" "${TMP_FILE}")     # CMake ≥3.21; otherwise use execute_process with -E copy
file(RENAME    "${TMP_FILE}" "${CONFIG_FILE}")
file(LOCK "${LOCK_FILE}" RELEASE)

# Include directories Externals and SRC
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/externals/imgui)
include_directories(${CMAKE_SOURCE_DIR}/externals/imgui/backends)
include_directories(${CMAKE_SOURCE_DIR}/externals/nlohmann)
include_directories(${CMAKE_SOURCE_DIR}/externals/yaml-cpp)

# Set Python executable - prefer activated virtual environment, fallback to CMake-discovered Python
if(DEFINED ENV{VIRTUAL_ENV})
    set(PYTHON_EXECUTABLE "$ENV{VIRTUAL_ENV}/bin/python")
elseif(DEFINED ENV{CONDA_PREFIX})
    set(PYTHON_EXECUTABLE "$ENV{CONDA_PREFIX}/bin/python")
else()
    set(PYTHON_EXECUTABLE "${Python3_EXECUTABLE}")
endif()
message(STATUS "Using Python executable: ${PYTHON_EXECUTABLE}")

# Core sources
set(CORE_SOURCES
        src/utils.h
        src/engine_core.cpp src/engine_core.h
        src/corine/dataset_handler.cpp src/corine/dataset_handler.h
        src/model_interface.h
)

# Agent sources
set(AGENT_SOURCES
        src/state.h
        src/reinforcementlearning/start_goal_strategy.h
        src/reinforcementlearning/agents/agent.h src/reinforcementlearning/agents/agent.cpp
        src/reinforcementlearning/agents/agent_factory.h
        src/reinforcementlearning/agents/fly_agent.h src/reinforcementlearning/agents/fly_agent.cpp
        src/reinforcementlearning/agents/explore_agent.h src/reinforcementlearning/agents/explore_agent.cpp
        src/reinforcementlearning/agents/planner_agent.h src/reinforcementlearning/agents/planner_agent.cpp
        src/reinforcementlearning/actions/action.h
        src/reinforcementlearning/actions/fly_action.h src/reinforcementlearning/actions/fly_action.cpp
        src/reinforcementlearning/actions/explore_action.h src/reinforcementlearning/actions/explore_action.cpp
        src/reinforcementlearning/actions/plan_action.h src/reinforcementlearning/actions/plan_action.cpp
        src/reinforcementlearning/agents/agent_state.cpp src/reinforcementlearning/agents/agent_state.h
        src/reinforcementlearning/texturerenderer.cpp src/reinforcementlearning/texturerenderer.h
        src/reinforcementlearning/groundstation.h src/reinforcementlearning/groundstation.cpp
        src/reinforcementlearning/rl_handler.cpp
        src/reinforcementlearning/rl_handler.h)

# Fire model sources
set(FIREMODEL_SOURCES
        src/firespin/firemodel.cpp src/firespin/firemodel.h
        src/firespin/firemodel_gridmap.cpp src/firespin/firemodel_gridmap.h
        src/firespin/rendering/firemodel_renderer.cpp src/firespin/rendering/firemodel_renderer.h
        src/firespin/rendering/firemodel_camera.cpp src/firespin/rendering/firemodel_camera.h
        src/firespin/firemodel_firecell.cpp src/firespin/firemodel_firecell.h
        src/firespin/particles/virtual_particle.cpp src/firespin/particles/virtual_particle.h
        src/firespin/particles/radiation_particle.cpp src/firespin/particles/radiation_particle.h
        src/firespin/fire_generator.cpp src/firespin/fire_generator.h
        src/firespin/wind.cpp src/firespin/wind.h
        src/firespin/utils.h src/firespin/utils.cpp
        src/firespin/model_parameters.h
        src/firespin/firemodel_cell_interface.h
        src/firespin/cell_classes/cell_generic_flooded.cpp src/firespin/cell_classes/cell_generic_flooded.h
        src/firespin/cell_classes/cell_generic_unburned.cpp src/firespin/cell_classes/cell_generic_unburned.h
        src/firespin/cell_classes/cell_sealed.cpp src/firespin/cell_classes/cell_sealed.h
        src/firespin/cell_classes/cell_woody_needle_leaved_trees.cpp src/firespin/cell_classes/cell_woody_needle_leaved_trees.h
        src/firespin/cell_classes/cell_woody_breadleaved_deciduous_trees.cpp src/firespin/cell_classes/cell_woody_breadleaved_deciduous_trees.h
        src/firespin/cell_classes/cell_woody_broadleaved_evergreen_trees.cpp src/firespin/cell_classes/cell_woody_broadleaved_evergreen_trees.h
        src/firespin/cell_classes/cell_low_growing_woody_plants.cpp src/firespin/cell_classes/cell_low_growing_woody_plants.h
        src/firespin/cell_classes/cell_permanent_herbaceous.cpp src/firespin/cell_classes/cell_permanent_herbaceous.h
        src/firespin/cell_classes/cell_periodically_herbaceous.cpp src/firespin/cell_classes/cell_periodically_herbaceous.h
        src/firespin/cell_classes/cell_lichens_and_mosses.cpp src/firespin/cell_classes/cell_lichens_and_mosses.h
        src/firespin/cell_classes/cell_non_and_sparsley_vegetated.cpp src/firespin/cell_classes/cell_non_and_sparsley_vegetated.h
        src/firespin/cell_classes/cell_water.cpp src/firespin/cell_classes/cell_water.h
        src/firespin/cell_classes/cell_snow_and_ice.cpp src/firespin/cell_classes/cell_snow_and_ice.h
        src/firespin/cell_classes/cell_generic_burning.cpp src/firespin/cell_classes/cell_generic_burning.h
        src/firespin/cell_classes/cell_outside_area.cpp src/firespin/cell_classes/cell_outside_area.h
        src/firespin/cell_classes/cell_generic_burned.cpp src/firespin/cell_classes/cell_generic_burned.h
        src/firespin/rendering/firemodel_pixelbuffer.cpp src/firespin/rendering/firemodel_pixelbuffer.h
        src/firespin/rendering/firemodel_imgui.cpp src/firespin/rendering/firemodel_imgui.h
        # New modular ImGui UI
        src/firespin/rendering/imgui/UITypes.h
        src/firespin/rendering/imgui/UIState.h
        src/firespin/rendering/imgui/UICallbacks.h
        src/firespin/rendering/imgui/UIManager.cpp src/firespin/rendering/imgui/UIManager.h
        src/firespin/rendering/imgui/MenuBar.h
        src/firespin/rendering/imgui/components/ConfigTable.h
        src/firespin/rendering/imgui/components/LogConsoleWidget.h
        src/firespin/rendering/imgui/components/GridVisualizer.h
        src/firespin/rendering/imgui/components/DroneInfoWidget.h
        src/firespin/rendering/imgui/components/RewardPlot.h
        src/firespin/rendering/imgui/windows/IWindow.h
        src/firespin/rendering/imgui/windows/SimulationControlsWindow.h
        src/firespin/rendering/imgui/windows/ParameterConfigWindow.h
        src/firespin/rendering/imgui/windows/StartupWizard.h
        src/firespin/rendering/imgui/windows/FileDialogWindow.h
        src/firespin/rendering/imgui/windows/CellPopupManager.h
        src/firespin/rendering/imgui/windows/RLStatusWindow.h
        # New UI enhancements
        src/firespin/rendering/imgui/ThemeManager.h
        src/firespin/rendering/imgui/components/ProgressBar.h
        src/firespin/rendering/imgui/components/StatusIndicator.h
)

# Core library
add_library(roshan_core STATIC
        ${CORE_SOURCES}
        ${AGENT_SOURCES}
        ${FIREMODEL_SOURCES})
set_target_properties(roshan_core PROPERTIES
        UNITY_BUILD ON                # batch-compile sources (cuts memory spikes)
)
target_include_directories(roshan_core PUBLIC
        ${CMAKE_SOURCE_DIR}
        ${GDAL_INCLUDE_DIRS})
target_link_libraries(roshan_core
        PUBLIC
        imgui_external
        yaml-cpp
        SDL2::SDL2
        ${SDL2_IMAGE_LIBRARY}
        OpenGL::GL
        Threads::Threads
        GDAL::GDAL          # if found as target, else use ${GDAL_LIBRARIES}
        PRIVATE
        CURL::libcurl                # if not defined, use ${CURL_LIBRARIES}
        OpenMP::OpenMP_CXX
)

# Python module
pybind11_add_module(firesim src/python_bindings.cpp)
target_link_libraries(firesim PRIVATE roshan_core)

# Main executable
add_executable(ROSHAN src/main.cpp)
target_include_directories(ROSHAN PRIVATE ${CMAKE_SOURCE_DIR}/externals/nlohmann)

# Link against libraries
target_link_libraries(ROSHAN PRIVATE roshan_core
        Python3::Python
        dl
        tiff
        ffi
)