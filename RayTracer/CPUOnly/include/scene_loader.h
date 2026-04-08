#ifndef SCENE_LOADER_H
#define SCENE_LOADER_H

#include "camera.h"
#include "material.h"
#include "transform.h"
#include "vec3.h"

#include <string>
#include <vector>

// Material parameters (corresponds to JSON "material" per node)
struct MaterialParams {
    float albedo[3] = {0.8f, 0.8f, 0.8f};
    float kd = 1.0f;
    float ks = 0.0f;
    float shininess = 32.0f;
    float specular_color[3] = {0.04f, 0.04f, 0.04f};
    float kr = 0.0f;
    float emission[3] = {0.0f, 0.0f, 0.0f};
};

// Global render settings (corresponds to JSON "settings")
struct RenderSettings {
    int max_bounces = 8;
    int samples_per_pixel = 100;  // more samples, less noise, slower
    bool diffuse_bounce = true;   // if true, randomly select diffuse/mirror; if false, only mirror
};

// Light parameters (corresponds to JSON "light")
struct LightParams {
    camera::point3 position{-3.0f, 0.0f, 1.0f};
    camera::vec3 color{1.0f, 1.0f, 1.0f};
    float intensity = 1.0f;  // light intensity multiplier
    float radius = 0.0f;          // 0 = point light, >0 = area light radius
    int shadow_samples = 1;       // e.g. 1 (hard), 8, 16, 32
};

// Camera parameters (corresponds to JSON "camera")
struct CameraParams {
    double focal_length_mm = 50.0;
    double sensor_height_mm = 24.0;
    double sensor_width_mm = 36.0;
    int pixel_width = 540;
    int pixel_height = 360;
    camera::point3 position{0.0f, 0.0f, 0.0f};
    camera::point3 look_at{0.0f, 0.0f, 0.0f};
    camera::vec3 up{0.0f, 0.0f, 1.0f};
};

// A node in the scene graph (one entry in the JSON "scene" array)
struct SceneNode {
    std::string name;
    std::string type;  // e.g. "mesh"
    std::string path;  // e.g. "./assets/meshes/frog.obj"
    Transform transform;   // per-mesh transform (optional in JSON)
    MaterialParams material; // per-mesh material (optional in JSON, has defaults)
};

// Full scene config: settings + camera + light + scene
struct SceneConfig {
    RenderSettings settings;
    CameraParams camera;
    LightParams light;
    std::vector<SceneNode> scene;
};

// Load scene definition from JSON file
class SceneLoader {
public:
    // Read JSON from json_path and parse into SceneConfig.
    // base_dir: base directory for resolving relative paths (e.g. config dir or project root), default ".".
    static SceneConfig load(const std::string& json_path,
                            const std::string& base_dir = ".");

    // Construct camera instance from CameraParams
    static camera make_camera(const CameraParams& p);

    // Build Material from MaterialParams (from JSON)
    static Material make_material(const MaterialParams& p);
};

#endif
