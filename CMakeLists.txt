cmake_minimum_required(VERSION 3.25)
project(ROSHAN)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set the build path
set(MY_CPP_BUILD_PATH "${CMAKE_BINARY_DIR}")

# Set the models path
get_filename_component(PARENT_DIR "${MY_CPP_BUILD_PATH}" DIRECTORY)
set(MODELS_PATH "${PARENT_DIR}/models")

# Set CMAKE Module Path for FindSDL2_image.cmake
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

# Enable optimizations
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -flto -funroll-loops -ffast-math")
# Enable faster-build optimizations
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
# Development flags
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")

# Set the path to SDL2 and SDL2_image
find_package(SDL2 REQUIRED)
find_package(SDL2_image REQUIRED)

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

# Generate a JSON configuration file
set(CONFIG_FILE "${PARENT_DIR}/config.json")

if(NOT EXISTS ${CONFIG_FILE})
    set(MODULE_DIRECTORY "${MY_CPP_BUILD_PATH}")
    set(MODELS_DIRECTORY "${MODELS_PATH}")
    set(OSM_DATA "${PARENT_DIR}/openstreetmap/data.json")
    set(DATASET_PATH "${CMAKE_SOURCE_DIR}/assets/dataset/CLMS_CLCplus_RASTER_2021_010m_eu_03035_V1_1.tif") # Default path for dataset_directory
    set(LOG_DIRECTORY "./logs")

    configure_file(
            "${CMAKE_SOURCE_DIR}/config.json.in"
            "${CONFIG_FILE}"
            @ONLY
    )
endif()

# Include directories Externals and SRC
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/externals/imgui)
include_directories(${CMAKE_SOURCE_DIR}/externals/imgui/backends)
include_directories(${CMAKE_SOURCE_DIR}/externals/nlohmann)

# Find Python
find_package(Python3 COMPONENTS Interpreter Development)

if (Python3_FOUND)
    message(STATUS "Found Python: ${Python3_EXECUTABLE} (found version ${Python3_VERSION})")
    include_directories(${Python3_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "Python3 not found")
endif()

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

# Find CURL (required by GDAL)
find_package(CURL REQUIRED)

# GDAL
find_package(GDAL REQUIRED)
include_directories(${GDAL_INCLUDE_DIRS})

# Core sources
set(CORE_SOURCES
        src/main.cpp
        src/utils.h
        src/engine_core.cpp src/engine_core.h
        src/corine/dataset_handler.cpp src/corine/dataset_handler.h
        externals/ImGuiFileDialog/ImGuiFileDialog.cpp externals/ImGuiFileDialog/ImGuiFileDialog.h
        src/model_interface.h
)

# Agent sources
#src/reinforcementlearning/drone_agent/rendering/DroneRenderer.cpp src/reinforcementlearning/drone_agent/rendering/DroneRenderer.h
set(AGENT_SOURCES
        src/state.h
        src/agent.h
        src/action.h
        src/reinforcementlearning/utils.h src/reinforcementlearning/utils.cpp
        src/reinforcementlearning/drone_agent/drone_state.cpp src/reinforcementlearning/drone_agent/drone_state.h
        src/reinforcementlearning/drone_agent/fly_action.h
        src/reinforcementlearning/drone_agent/explore_action.h
        src/reinforcementlearning/drone_agent/drone.cpp src/reinforcementlearning/drone_agent/drone.h
        src/reinforcementlearning/texturerenderer.cpp src/reinforcementlearning/texturerenderer.h
        src/reinforcementlearning/groundstation.h src/reinforcementlearning/groundstation.cpp
        src/reinforcementlearning/reinforcementlearning_handler.cpp
        src/reinforcementlearning/reinforcementlearning_handler.h)

# Fire model sources
set(FIREMODEL_SOURCES
        src/firespin/firemodel.cpp src/firespin/firemodel.h
        src/firespin/firemodel_gridmap.cpp src/firespin/firemodel_gridmap.h
        src/firespin/rendering/firemodel_renderer.cpp src/firespin/rendering/firemodel_renderer.h
        src/firespin/rendering/firemodel_camera.cpp src/firespin/rendering/firemodel_camera.h
        src/firespin/firemodel_firecell.cpp src/firespin/firemodel_firecell.h
        src/firespin/particles/virtual_particle.cpp src/firespin/particles/virtual_particle.h
        src/firespin/particles/radiation_particle.cpp src/firespin/particles/radiation_particle.h
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
)

# Python bindings
pybind11_add_module(firesim src/python_bindings.cpp
        ${CORE_SOURCES}
        ${AGENT_SOURCES}
        ${FIREMODEL_SOURCES}
        ${IMGUI_SOURCES}
)

target_include_directories(firesim PRIVATE
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/externals/imgui
        ${CMAKE_SOURCE_DIR}/externals/imgui/backends/
        /usr/local/include/SDL2
        ${GDAL_INCLUDE_DIRS}
)

target_link_libraries(firesim PRIVATE
        ${SDL2_LIBRARIES}
        ${SDL2_IMAGE_LIBRARY}
        ${CURL_LIBRARIES}
        ${GDAL_LIBRARIES}
        ${X11_LIBRARIES}
        OpenGL::GL
        imgui_external
)

# Main executable
add_executable(ROSHAN
        ${CORE_SOURCES}
        ${AGENT_SOURCES}
        ${FIREMODEL_SOURCES}
)

target_include_directories(ROSHAN PRIVATE
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/externals/nlohmann
)

# Link against libraries
target_link_libraries(ROSHAN PRIVATE
        ${SDL2_LIBRARIES}
        ${SDL2_IMAGE_LIBRARY}
        ${X11_LIBRARIES}
        ${Python3_LIBRARIES}
        OpenGL::GL
        imgui_external
        ${CURL_LIBRARIES}
        ${GDAL_LIBRARIES}
        dl
        tiff
)