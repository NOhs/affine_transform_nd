cmake_minimum_required(VERSION 3.9.0)
project(affine_transform)

find_package(OpenMP)

add_subdirectory(extern/pybind11)

if(MSVC)
    add_compile_options("/W4" "/Ox" "/fp:fast")
    set(PYBIND11_CPP_STANDARD /std:c++latest)
endif()
if(UNIX)
    add_compile_options("-Ofast" "-std=c++17")
    set(PYBIND11_CPP_STANDARD -std=c++17)
endif()

pybind11_add_module(_affine_transform src/main.cpp)
target_include_directories(_affine_transform PUBLIC "extern/eigen")
target_include_directories(_affine_transform PUBLIC "include")

if(OpenMP_CXX_FOUND)
    target_link_libraries(_affine_transform PUBLIC OpenMP::OpenMP_CXX)
endif()

set(CMAKE_VERBOSE_MAKEFILE ON)
