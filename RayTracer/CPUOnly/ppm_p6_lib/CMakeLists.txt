cmake_minimum_required(VERSION 3.16)
project(ppm_p6_lib LANGUAGES CXX)

add_library(ppm_p6 STATIC
    src/ppm_p6.cpp
)

target_compile_features(ppm_p6 PUBLIC cxx_std_17)

target_include_directories(ppm_p6 PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(ppm_p6_test
    examples/ppm_p6_test.cpp
)

target_link_libraries(ppm_p6_test PRIVATE ppm_p6)
