# cmake needs this line
cmake_minimum_required(VERSION 3.1)

# Define project name
project(project-f VERSION 0.1)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find OpenCV, you may need to set OpenCV_DIR variable
# to the absolute path to the directory containing OpenCVConfig.cmake file
# via the command line or GUI
find_package(OpenCV REQUIRED)

# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them
message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# Add OpenCV headers location to your include paths
include_directories(${OpenCV_INCLUDE_DIRS})

# aux_source_directory(. DIR_SRCS)

# Declare the executable target built from your sources

# add_subdirectory(src)
add_executable(project-f src/project-f.cpp)

# Link your application with OpenCV libraries
target_link_libraries(project-f PRIVATE ${OpenCV_LIBS})
# target_link_libraries(project-f PUBLIC ${PROJECT_SOURCE_DIR})

