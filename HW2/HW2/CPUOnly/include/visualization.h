#ifndef VISUALIZATION_H
#define VISUALIZATION_H

#include "camera.h"

#include <array>
#include <vector>

#include <glm/glm.hpp>

#include "polyscope/polyscope.h"
#include "polyscope/curve_network.h"


namespace viz {

// Convert our point3/vec3 type to glm::vec3 for Polyscope
inline glm::vec3 to_glm_vec3(const camera::point3& p) {
    return glm::vec3( static_cast<float>(p.x),
        static_cast<float>(p.y),
        static_cast<float>(p.z));
}

// Initialize Polyscope and set Z-up view convention.
inline void init_polyscope_zup() {
    polyscope::init();
    polyscope::view::setUpDir(polyscope::UpDir::ZUp);
}

// Enter Polyscope UI loop.
inline void show() {
    polyscope::show();
}

struct AxesVizOptions {
    bool radius_is_relative = true;
    double axis_radius = 0.005;

    // Colors (RGB)
    glm::vec3 x_color = glm::vec3(1.0f, 0.0f, 0.0f);
    glm::vec3 y_color = glm::vec3(0.0f, 1.0f, 0.0f);
    glm::vec3 z_color = glm::vec3(0.0f, 0.0f, 1.0f);
};

// Draw XYZ axes as curve networks.
inline void register_axes(double axis_length,
                          const glm::vec3& origin = glm::vec3(0.0f, 0.0f, 0.0f),
                          const AxesVizOptions& opt = {}) {
    // X axis
    std::vector<glm::vec3> x_axis_nodes = {origin, origin + glm::vec3((float)axis_length, 0.0f, 0.0f)};
    std::vector<std::array<size_t, 2>> x_axis_edges = {{0, 1}};
    auto* x_axis = polyscope::registerCurveNetwork("X Axis", x_axis_nodes, x_axis_edges);
    x_axis->setRadius(opt.axis_radius, /*isRelative=*/opt.radius_is_relative);
    x_axis->setColor(opt.x_color);

    // Y axis
    std::vector<glm::vec3> y_axis_nodes = {origin, origin + glm::vec3(0.0f, (float)axis_length, 0.0f)};
    std::vector<std::array<size_t, 2>> y_axis_edges = {{0, 1}};
    auto* y_axis = polyscope::registerCurveNetwork("Y Axis", y_axis_nodes, y_axis_edges);
    y_axis->setRadius(opt.axis_radius, /*isRelative=*/opt.radius_is_relative);
    y_axis->setColor(opt.y_color);

    // Z axis
    std::vector<glm::vec3> z_axis_nodes = {origin, origin + glm::vec3(0.0f, 0.0f, (float)axis_length)};
    std::vector<std::array<size_t, 2>> z_axis_edges = {{0, 1}};
    auto* z_axis = polyscope::registerCurveNetwork("Z Axis", z_axis_nodes, z_axis_edges);
    z_axis->setRadius(opt.axis_radius, /*isRelative=*/opt.radius_is_relative);
    z_axis->setColor(opt.z_color);
}

} // namespace viz

#endif

