cmake_minimum_required(VERSION 3.5)  # Adjust version as needed
project(VideoFilters)

# Find OpenCV package
set(OpenCV_DIR /opt/homebrew/opt/opencv/share/opencv4)
find_package(OpenCV REQUIRED)

# Include OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})

# Add source files
set(SOURCE_FILES
    #src/imgDisplay.cpp
    #src/filters.cpp
    #src/faceDetect.cpp
    src/hist.cpp
)

# Define the executable
add_executable(ImageDisplay ${SOURCE_FILES})

# Link the executable to OpenCV libraries
target_link_libraries(ImageDisplay ${OpenCV_LIBS})
