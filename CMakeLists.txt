cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 17)

project(SportVideoAnalysis-SADVision)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} source/detection.cpp include/detection.h source/Player.cpp include/Player.h source/utils.cpp include/utils.h source/detection_segmentation.cpp include/detection_segmentation.h source/BoundingBox.cpp include/BoundingBox.h source/performance_measurement.cpp include/ColorFeaturesSegmentator.h source/ColorFeaturesSegmentator.cpp source/main.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})