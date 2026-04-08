#include "scene_loader.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <sstream>
#include <stdexcept>

using json = nlohmann::json;

namespace {

camera::point3 parse_vec3(const json& j) {
    if (!j.is_array() || j.size() < 3)
        throw std::runtime_error("Expected array of 3 numbers for vec3");
    return camera::point3{
        static_cast<float>(j[0].get<double>()),
        static_cast<float>(j[1].get<double>()),
        static_cast<float>(j[2].get<double>())
    };
}

Vec3 parse_vec3_v(const json& j) {
    camera::point3 p = parse_vec3(j);
    return make_vec3(p.x, p.y, p.z);
}

Transform parse_transform(const json& node) {
    Transform t{};
    if (!node.contains("transform")) return t;

    const auto& tr = node["transform"];
    if (!tr.is_object()) return t;

    if (tr.contains("position")) {
        t.position = parse_vec3_v(tr["position"]);
    }
    if (tr.contains("rotation")) {
        t.rotation_deg = parse_vec3_v(tr["rotation"]);
    }
    if (tr.contains("scale")) {
        const auto& sc = tr["scale"];
        if (sc.is_number()) {
            const float s = static_cast<float>(sc.get<double>());
            t.scale = make_vec3(s, s, s);
        } else {
            t.scale = parse_vec3_v(sc);
        }
    }
    return t;
}

MaterialParams parse_material(const json& node) {
    MaterialParams m;
    if (!node.contains("material") || !node["material"].is_object())
        return m;

    const auto& mat = node["material"];
    if (mat.contains("albedo")) {
        camera::point3 p = parse_vec3(mat["albedo"]);
        m.albedo[0] = p.x; m.albedo[1] = p.y; m.albedo[2] = p.z;
    }
    if (mat.contains("kd")) m.kd = static_cast<float>(mat["kd"].get<double>());
    if (mat.contains("ks")) m.ks = static_cast<float>(mat["ks"].get<double>());
    if (mat.contains("shininess")) m.shininess = static_cast<float>(mat["shininess"].get<double>());
    if (mat.contains("specular_color")) {
        camera::point3 p = parse_vec3(mat["specular_color"]);
        m.specular_color[0] = p.x; m.specular_color[1] = p.y; m.specular_color[2] = p.z;
    }
    if (mat.contains("kr")) m.kr = static_cast<float>(mat["kr"].get<double>());
    if (mat.contains("emission")) {
        camera::point3 p = parse_vec3(mat["emission"]);
        m.emission[0] = p.x; m.emission[1] = p.y; m.emission[2] = p.z;
    }
    return m;
}

std::string resolve_path(const std::string& base_dir, const std::string& path) {
    if (path.empty()) return path;
    if (path.size() >= 2 && path[0] == '.' && (path[1] == '/' || path[1] == '\\'))
        return base_dir + "/" + path.substr(2);
    if (path[0] == '/' || (path.size() >= 2 && path[1] == ':'))
        return path; // absolute
    return base_dir + "/" + path;
}

} // namespace

SceneConfig SceneLoader::load(const std::string& json_path,
                              const std::string& base_dir) {
    std::ifstream f(json_path);
    if (!f.is_open()) {
        throw std::runtime_error("SceneLoader: cannot open file: " + json_path);
    }
    json j;
    try {
        f >> j;
    } catch (const json::exception& e) {
        throw std::runtime_error(std::string("SceneLoader: JSON parse error: ") + e.what());
    }

    SceneConfig config;

    // settings
    if (j.contains("settings")) {
        const auto& s = j["settings"];
        if (s.contains("max_bounces"))
            config.settings.max_bounces = s["max_bounces"].get<int>();
        if (s.contains("samples_per_pixel"))
            config.settings.samples_per_pixel = s["samples_per_pixel"].get<int>();
        if (s.contains("diffuse_bounce"))
            config.settings.diffuse_bounce = s["diffuse_bounce"].get<bool>();
    }

    // camera
    if (j.contains("camera")) {
        const auto& c = j["camera"];
        if (c.contains("focal_length_mm"))
            config.camera.focal_length_mm = c["focal_length_mm"].get<double>();
        if (c.contains("sensor_height_mm"))
            config.camera.sensor_height_mm = c["sensor_height_mm"].get<double>();
        if (c.contains("sensor_width_mm"))
            config.camera.sensor_width_mm = c["sensor_width_mm"].get<double>();
        if (c.contains("pixel_width"))
            config.camera.pixel_width = c["pixel_width"].get<int>();
        if (c.contains("pixel_height"))
            config.camera.pixel_height = c["pixel_height"].get<int>();
        if (c.contains("position"))
            config.camera.position = parse_vec3(c["position"]);
        if (c.contains("look_at"))
            config.camera.look_at = parse_vec3(c["look_at"]);
        if (c.contains("up"))
            config.camera.up = parse_vec3(c["up"]);
    }

    // light
    if (j.contains("light")) {
        const auto& l = j["light"];
        if (l.contains("position"))
            config.light.position = parse_vec3(l["position"]);
        if (l.contains("color"))
            config.light.color = parse_vec3(l["color"]);
        if (l.contains("intensity"))
            config.light.intensity = static_cast<float>(l["intensity"].get<double>());
        if (l.contains("radius"))
            config.light.radius = static_cast<float>(l["radius"].get<double>());
        if (l.contains("shadow_samples"))
            config.light.shadow_samples = l["shadow_samples"].get<int>();
    }

    // scene
    if (j.contains("scene") && j["scene"].is_array()) {
        for (const auto& node : j["scene"]) {
            SceneNode sn;
            if (node.contains("name")) sn.name = node["name"].get<std::string>();
            if (node.contains("type")) sn.type = node["type"].get<std::string>();
            if (node.contains("path")) {
                std::string raw = node["path"].get<std::string>();
                sn.path = resolve_path(base_dir, raw);
            }
            sn.transform = parse_transform(node);
            sn.material = parse_material(node);
            config.scene.push_back(std::move(sn));
        }
    }

    return config;
}

camera SceneLoader::make_camera(const CameraParams& p) {
    return camera(p.position, p.look_at, p.up,
                  p.focal_length_mm, p.sensor_height_mm, p.sensor_width_mm,
                  p.pixel_width, p.pixel_height);
}

Material SceneLoader::make_material(const MaterialParams& p) {
    Material m;
    m.albedo = make_vec3(p.albedo[0], p.albedo[1], p.albedo[2]);
    m.kd = p.kd;
    m.ks = p.ks;
    m.shininess = p.shininess;
    m.specularColor = make_vec3(p.specular_color[0], p.specular_color[1], p.specular_color[2]);
    m.kr = p.kr;
    m.emission = make_vec3(p.emission[0], p.emission[1], p.emission[2]);
    return m;
}
