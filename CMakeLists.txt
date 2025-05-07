cmake_minimum_required(VERSION 3.15.0)
project(affine_transform)

find_package(OpenMP)

# Use FetchContent to automatically download dependencies
include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.13.6
)
FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
)
FetchContent_MakeAvailable(pybind11 eigen)

set(CMAKE_CXX_STANDARD 17)

if(MSVC)
    add_compile_options("/W4" "/Ox" "/fp:fast")
endif()
if(UNIX)
    add_compile_options("-Ofast")
endif()

pybind11_add_module(_affine_transform src/main.cpp)
target_include_directories(_affine_transform PUBLIC "${eigen_SOURCE_DIR}")  # Use FetchContent path
target_include_directories(_affine_transform PUBLIC "include")

if(OpenMP_CXX_FOUND)
    target_link_libraries(_affine_transform PUBLIC OpenMP::OpenMP_CXX)
endif()

set(CMAKE_VERBOSE_MAKEFILE ON)