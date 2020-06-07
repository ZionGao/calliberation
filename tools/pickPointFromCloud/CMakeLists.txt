cmake_minimum_required(VERSION 3.16)
project(calibrateCamera)
set(CMAKE_CXX_STANDARD 14)

add_executable(calibrateCamera main.cpp)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)

add_definitions(${PCL_DEFINITIONS})
set(CMAKE_CXX_FLAGS "-g -Wall -std=c++11")
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PCL_INCLUDE_DIRS})
set(SOURCE_FILES main.cpp)
target_link_libraries(calibrateCamera ${OpenCV_LIBS} ${PCL_LIBS} ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES} ${PCL_VISUALIZATION_LIBRARIES} ${PCL_RESISTRATION_LIBRARIES} ${PCL_FILTERS_LIBRARIES} ${PCL_FEATURES_LIBRARIES} ${Boost_LIBRARIES})
