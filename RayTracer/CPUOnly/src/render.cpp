// src/render.cpp
#include "MeshOBJ.h"
#include "camera.h"
#include "ray.h"
#include "raytracer.h"
#include "scene_loader.h"
#include "transform.h"
#include "vec3.h"
#include "material.h"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>
#include <limits>
#include <filesystem>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


int main(int argc, char** argv)
{
    // 1) Choose scene JSON
    std::string config_path = "config/sphere_area.json";
    if (argc >= 2) config_path = argv[1];

    // Run from project root
    std::string base_dir = ".";

    SceneConfig config;
    try {
        config = SceneLoader::load(config_path, base_dir);
    } catch (const std::exception& e) {
        std::cerr << "SceneLoader error: " << e.what() << "\n";
        return 1;
    }

    // 2) Camera from config
    camera cam = SceneLoader::make_camera(config.camera);

    // 3) Lights from config (should hold for multiple lights)
    std::vector<Light> lights;
    {
        Light L;
        L.position = make_vec3(config.light.position.x, config.light.position.y, config.light.position.z);
        L.color    = make_vec3(config.light.color.x,    config.light.color.y,    config.light.color.z);
        L.intensity = config.light.intensity;
		L.radius = config.light.radius;
		L.shadow_samples = config.light.shadow_samples;
        lights.push_back(L);
    }

    // 4) Load all scene meshes and bake transforms into triangles; material per node from JSON
    std::vector<Triangle> tris;
    for (const auto& node : config.scene) {
        if (node.type != "mesh" || node.path.empty()) continue;

        std::cout << "Loading mesh node '" << node.name << "': " << node.path << "\n";

        MeshSOA mesh;
        if (!LoadOBJ_ToMeshSOA(node.path, mesh)) {
            std::cerr << "Failed to load OBJ: " << node.path << "\n";
            return 1;
        }
        const size_t vc = mesh.positions.size();
        const size_t tc = mesh.indices.size() / 3;
        std::cout << "  Vertices: " << vc << ", Triangles: " << tc << "\n";

        // Apply node transform to mesh in-place (positions + normals)
        ApplyTransformToMeshSOA(mesh, node.transform);

        Material mat = SceneLoader::make_material(node.material);

        // Convert mesh to triangles once
        const size_t indexCount = mesh.indices.size();
        tris.reserve(tris.size() + indexCount / 3);

        for (size_t k = 0; k < indexCount; k += 3) {
            Triangle tri;
            tri.mat = mat;

            tri.v0 = mesh.positions[mesh.indices[k]];
            tri.v1 = mesh.positions[mesh.indices[k + 1]];
            tri.v2 = mesh.positions[mesh.indices[k + 2]];

            if (mesh.hasNormals()) {
                tri.n0 = mesh.normals[mesh.indices[k]];
                tri.n1 = mesh.normals[mesh.indices[k + 1]];
                tri.n2 = mesh.normals[mesh.indices[k + 2]];
            } else {
                Vec3 faceN = unit_vector(cross(tri.v1 - tri.v0, tri.v2 - tri.v0));
                tri.n0 = tri.n1 = tri.n2 = faceN;
            }

            tris.push_back(tri);
        }
    }

    if (tris.empty()) {
        std::cerr << "Scene contains no triangles.\n";
        return 1;
    }

    // 6) Render the image
    const int pixel_width  = cam.pixel_width;
    const int pixel_height = cam.pixel_height;

    std::vector<Vec3> image(static_cast<size_t>(pixel_width) * static_cast<size_t>(pixel_height));
    const Vec3 center = cam.get_center();

    const int maxDepth = std::max(1, config.settings.max_bounces);

    const int samples_per_pixel = config.settings.samples_per_pixel;
    const int bar_width = 40;

    auto t_start = std::chrono::steady_clock::now();
    for (int j = 0; j < pixel_height; ++j) {
        const int pct = (j + 1) * 100 / pixel_height;
        const int filled = (j + 1) * bar_width / pixel_height;
        std::cerr << "\r[" << std::string(filled, '=') << std::string(bar_width - filled, ' ') << "] " << pct << "%" << std::flush;

        for (int i = 0; i < pixel_width; ++i) {
            Vec3 pixel_color = make_vec3(0, 0, 0);
            for (int s = 0; s < samples_per_pixel; ++s) {
                // sub-pixel jittering only when spp > 1; otherwise sample pixel center
				double du = (samples_per_pixel > 1) ? random_float() : 0.5;
				double dv = (samples_per_pixel > 1) ? random_float() : 0.5;

				double u = static_cast<double>(i) + du;
				double v = static_cast<double>(j) + dv;

                Vec3 target = cam.get_pixel_position(u, v);
                Ray r(center, target - center);
                pixel_color = pixel_color + TraceRay(r, tris, lights, maxDepth, config.settings.diffuse_bounce);
            }
            image[static_cast<size_t>(j) * static_cast<size_t>(pixel_width) + static_cast<size_t>(i)] = pixel_color / static_cast<float>(samples_per_pixel);
        }
    }
    std::cerr << "\r[" << std::string(bar_width, '=') << "] 100%\n";

    auto t_end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();
    std::cout << "Render time: " << elapsed << " s\n";

	// 7) Write PNG: put all renders in output/
	namespace fs = std::filesystem;

	fs::path json_path(config_path);
	std::string stem = json_path.stem().string();

	fs::path out_path = fs::path("output") / (stem + "_output.png");
	std::string out = out_path.string();

	std::cout << "writing: " << out << "\n";

    std::vector<unsigned char> png_data(static_cast<size_t>(pixel_width) * static_cast<size_t>(pixel_height) * 3);
    for (int k = 0; k < pixel_width * pixel_height; ++k) {
        Vec3 c = clamp(image[static_cast<size_t>(k)]);
        png_data[static_cast<size_t>(k) * 3 + 0] = static_cast<unsigned char>(255.99f * c.x);
        png_data[static_cast<size_t>(k) * 3 + 1] = static_cast<unsigned char>(255.99f * c.y);
        png_data[static_cast<size_t>(k) * 3 + 2] = static_cast<unsigned char>(255.99f * c.z);
    }

    stbi_write_png(out.c_str(), pixel_width, pixel_height, 3, png_data.data(), pixel_width * 3);

    std::cout << "Done.\n";
    return 0;
}

