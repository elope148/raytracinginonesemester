cmake_minimum_required(VERSION 3.15)
project(ray_tri_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

find_package(Catch2 3 REQUIRED)

add_library(ray_triangle_intersection 
  ray.cpp
  vec3.cpp
)


target_include_directories(ray_triangle_intersection PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(ray_tri_tests
  test_ray_triangle_inter.cpp
)

target_link_libraries(ray_tri_tests PRIVATE ray_triangle_intersection Catch2::Catch2WithMain)
