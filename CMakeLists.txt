cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

enable_language(CUDA)

project(priority_stream LANGUAGES CXX CUDA)

find_package(CUDAToolkit)
find_package(glog)

add_library(priority_stream priority_stream.cc)
include_directories(priority_stream ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(priority_stream PUBLIC CUDA::cudart CUDA::cublas glog)

add_executable(main main.cc)
target_link_libraries(main priority_stream)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")
