#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "camera.h"
#include <glm/glm.hpp>
#include <cmath>

// ========================= Test 1: Constructor guard =========================
TEST_CASE("camera throws on invalid pixel dimensions") {

    glm::dvec3 pos(0, 0, 0);
    glm::dvec3 lookAt(0, 1, 0);
    glm::dvec3 up(0, 0, 1);

    REQUIRE_THROWS_AS(
        camera(pos, lookAt, up, 50.0, 24.0, 36.0, 0, 10),
        std::runtime_error
    );

    REQUIRE_THROWS_AS(
        camera(pos, lookAt, up, 50.0, 24.0, 36.0, 10, 0),
        std::runtime_error
    );
    INFO("Test 1 passed");
}

// =========================  Test 2: 1x1 pixel location =========================
TEST_CASE("1x1 camera pixel lies on optical axis") {

    glm::dvec3 center(0.3, 0.4, 0.5);
    glm::dvec3 lookAt(0.3, 0.4, 1.5);
    glm::dvec3 up(0, 1, 0);

    double focal_mm = 35.0;
    double sensor_height_mm = 24.0;
    double sensor_width_mm = 36.0;

    camera cam(center, lookAt, up, focal_mm, sensor_height_mm, sensor_width_mm, 1, 1);

    glm::dvec3 pixel = cam.get_pixel_position(0, 0);

    glm::dvec3 forward = glm::normalize(lookAt - center);
    double focal_m = focal_mm / 1000.0;

    glm::dvec3 expected = center + focal_m * forward;

    REQUIRE(pixel.x == Catch::Approx(expected.x));
    REQUIRE(pixel.y == Catch::Approx(expected.y));
    REQUIRE(pixel.z == Catch::Approx(expected.z));
    INFO("Test 2 passed");
}

// =========================  Test 3: Pixel plane geometry =========================
TEST_CASE("pixel grid lies in plane orthogonal to view direction") {

    glm::dvec3 center(0, 0, 0);
    glm::dvec3 lookAt(0, 0, 1);
    glm::dvec3 up(0, 1, 0);

    camera cam(center, lookAt, up, 50.0, 24.0, 36.0, 5, 4);

    glm::dvec3 forward = glm::normalize(lookAt - center);

    for (int j = 0; j < cam.pixel_height; ++j) {
        for (int i = 0; i < cam.pixel_width; ++i) {

            glm::dvec3 p = cam.get_pixel_position(i, j);
            glm::dvec3 v = p - center;

            REQUIRE(glm::dot(v, forward) > 0.0);

            if (i + 1 < cam.pixel_width) {
                glm::dvec3 dx = cam.get_pixel_position(i + 1, j) - p;
                REQUIRE(std::abs(glm::dot(dx, forward)) < 1e-12);
            }
        }
    }
    INFO("Test 3 passed");
}
