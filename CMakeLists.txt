cmake_minimum_required(VERSION 3.22)
set(CMAKE_CXX_COMPILER "/usr/bin/g++")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.8/bin/nvcc")
project(filtering LANGUAGES CUDA CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

set(CMAKE_BUILD_TYPE Debug)

find_package(CUDAToolkit REQUIRED)

add_executable(filtering
        src/benchmark.cu
        src/main.cpp
        src/thread_filter.cu
        src/block_filter.cu
        src/common.cu
        src/kmp.cu)

set_target_properties(filtering PROPERTIES CMAKE_CUDA_ARCHITECTURES 61)

target_include_directories(filtering PUBLIC includes/)

target_link_libraries(filtering CUDA::cufft CUDA::cudart)

add_custom_command(TARGET filtering PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E create_symlink
        ${CMAKE_SOURCE_DIR}/test.json $<TARGET_FILE_DIR:filtering>/test.json)

set_property(TARGET filtering PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(filtering PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --device-c
        >)