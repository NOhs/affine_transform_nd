cmake_minimum_required(VERSION 3.15)
project(affine_transform LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# --------------------------
# Fetch pybind11
# --------------------------
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.13.6
)
FetchContent_MakeAvailable(pybind11)

# --------------------------
# Fetch Eigen3 (header-only)
# --------------------------
FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0 
)
FetchContent_MakeAvailable(eigen)

# --------------------------
# Find OpenMP
# --------------------------
find_package(OpenMP REQUIRED)

pybind11_add_module(_affine_transform src/main.cpp)
target_include_directories(_affine_transform PUBLIC "include")
target_include_directories(_affine_transform PRIVATE ${eigen_SOURCE_DIR})

if(MSVC)
    add_compile_options("/W4" "/Ox" "/fp:fast")
endif()
if(UNIX)
    add_compile_options("-Ofast")
endif()


if(OpenMP_CXX_FOUND)
    target_link_libraries(_affine_transform PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(_affine_transform PRIVATE USE_OPENMP)
endif()

set_target_properties(_affine_transform PROPERTIES
    PREFIX ""  # Remove 'lib' prefix
    OUTPUT_NAME "_affine_transform"
)

# Install into the correct location inside the wheel
install(TARGETS _affine_transform
        LIBRARY DESTINATION affine_transform)
