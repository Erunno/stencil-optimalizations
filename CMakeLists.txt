cmake_minimum_required(VERSION 3.21)
include(FetchContent)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_COMPILER "/usr/bin/g++")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_CUDA_STANDARD 20)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -arch=native --use_fast_math")

FetchContent_Declare(
    cxxopts
    GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
    GIT_SHALLOW TRUE
    GIT_PROGRESS ON
    GIT_TAG 3bf2684
)

FetchContent_MakeAvailable(cxxopts)

set(Language "CUDA")
if (UseHip)
  set(Language "HIP")
endif()

project(
  Stencils
  VERSION 1.0
  LANGUAGES CXX CUDA)
  
find_package(CUDAToolkit REQUIRED)

add_subdirectory(src)

target_link_libraries(stencils PRIVATE CUDA::cudart)
target_link_libraries(stencils PRIVATE cxxopts)

if (MSVC)
    target_compile_options(stencils PRIVATE /W4 /WX)
else()
    target_compile_options(stencils PRIVATE -Wall -Wextra)
endif()
