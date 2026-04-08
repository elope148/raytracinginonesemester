#ifndef RAYTRACER_H
#define RAYTRACER_H

#include <cmath>
#include <limits>
#include <random>
#include <vector>
#include "ray.h"
#include "brdf.h"

// Random [0, 1) for diffuse sampling
inline float random_float() {
    static std::mt19937 gen(std::random_device{}());
    static std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    return dist(gen);
}

inline Vec3 random_unit_vector() {
    for (;;) {
        float x = 2.0f * random_float() - 1.0f;
        float y = 2.0f * random_float() - 1.0f;
        float z = 2.0f * random_float() - 1.0f;
        float lensq = x*x + y*y + z*z;
        if (lensq > 1e-10f && lensq <= 1.0f) {
            float inv = 1.0f / sqrtf(lensq);
            return make_vec3(x * inv, y * inv, z * inv);
        }
    }
}

inline Vec3 random_on_hemisphere(const Vec3& normal) {
    Vec3 on_unit_sphere = random_unit_vector();
    if (dot(on_unit_sphere, normal) > 0.0f)
        return on_unit_sphere;
    return make_vec3(-on_unit_sphere.x, -on_unit_sphere.y, -on_unit_sphere.z);
}

struct Light {
    Vec3 position;
    Vec3 color;
    float intensity = 1.0f;

    // soft shadow controls
    float radius = 0.0f;        // 0 = point light, >0 = spherical area light
    int   shadow_samples = 1;   // e.g. 1 (hard shadows), 8, 16, 32
};

// Small offset to avoid self-intersection / "shadow acne"
static constexpr float RT_EPS = 1e-4f;

// Optional: point light inverse-square falloff (OFF by default to preserve scene brightness)
#ifndef RT_USE_DISTANCE_ATTENUATION
#define RT_USE_DISTANCE_ATTENUATION 0
#endif

HYBRID_FUNC inline Vec3 clamp(Vec3 color) {
    if (color.x > 1.0f) color.x = 1.0f;
    if (color.y > 1.0f) color.y = 1.0f;
    if (color.z > 1.0f) color.z = 1.0f;
    if (color.x < 0.0f) color.x = 0.0f;
    if (color.y < 0.0f) color.y = 0.0f;
    if (color.z < 0.0f) color.z = 0.0f;
    return color;
}

HYBRID_FUNC inline float length3(const Vec3& v) {
    return sqrtf(dot(v, v));
}

HYBRID_FUNC inline Vec3 reflect_dir(const Vec3& I, const Vec3& N) {
    // I is the ray direction (from origin toward scene)
    // Reflection: R = I - 2*(I·N)*N
    return I - (2.0f * dot(I, N)) * N;
}

inline Vec3 random_in_unit_disk() {
    for (;;) {
        float x = 2.0f * random_float() - 1.0f;
        float y = 2.0f * random_float() - 1.0f;
        float r2 = x*x + y*y;
        if (r2 > 1e-10f && r2 <= 1.0f) {
            return make_vec3(x, y, 0.0f);
        }
    }
}

// Build an orthonormal basis (T,B) around a direction W (unit)
inline void make_basis(const Vec3& W, Vec3& T, Vec3& B) {
    // pick a helper axis not parallel to W
    Vec3 a = (fabsf(W.x) > 0.9f) ? make_vec3(0,1,0) : make_vec3(1,0,0);
    T = unit_vector(cross(a, W));
    B = cross(W, T);
}

// Scene intersection: find closest hit among triangles
inline bool IntersectScene(const Ray& r,
                           const std::vector<Triangle>& tris,
                           double t_min,
                           double t_max,
                           HitRecord& outRec)
{
    bool hit_anything = false;
    double closest = t_max;

    HitRecord temp{};
    for (const auto& tri : tris) {
        temp = ray_intersection(r, tri);
        if (!temp.hit) continue;

        if (temp.t >= t_min && temp.t < closest) {
            hit_anything = true;
            closest = temp.t;
            outRec = temp;
        }
    }

    return hit_anything;
}

// Check if there is any occluder between P and the light + soft area
inline float ShadowVisibility(const Vec3& P,
                             const Vec3& N,
                             const Light& light,
                             const std::vector<Triangle>& tris)
{
    // If point light (hard shadow case)
    int S = (light.radius > 0.0f) ? light.shadow_samples : 1;
    if (S < 1) S = 1;

    float unoccluded = 0.0f;

    // Direction from shaded point to the light center
    Vec3 toC = light.position - P;
    float distC = length3(toC);
    if (distC <= 0.0f) return 1.0f;

    // Disk normal faces the shaded point (from light -> P)
    Vec3 W = (P - light.position) / distC;

    // Disk basis at the light, oriented facing P
    Vec3 T, B;
    make_basis(W, T, B);

    for (int i = 0; i < S; i++) {
        Vec3 lightPos = light.position;

        if (light.radius > 0.0f) {
            Vec3 d = random_in_unit_disk(); // (x,y,0)
            lightPos = light.position + (T * (d.x * light.radius)) + (B * (d.y * light.radius));
        }

        Vec3 toL = lightPos - P;
        float distToL = length3(toL);
        if (distToL <= 0.0f) { unoccluded += 1.0f; continue; }

        Vec3 Ldir = toL / distToL;

        // small offset to avoid acne
        Ray shadowRay(P + N * RT_EPS, Ldir);

        HitRecord shadowHit{};
        bool blocked = IntersectScene(shadowRay, tris, RT_EPS, double(distToL) - RT_EPS, shadowHit);

        if (!blocked) unoccluded += 1.0f;
    }

    return unoccluded / float(S);
}

// Direct lighting (multiple lights supported)
inline Vec3 ShadeDirect(const Ray& r,
                        const HitRecord& rec,
                        const std::vector<Light>& lights,
                        const std::vector<Triangle>& tris)
{
    Vec3 N = unit_vector(rec.normal);
    Vec3 V = unit_vector(r.origin() - rec.p);

    Vec3 Lo = make_vec3(0,0,0);

    Lo = Lo + rec.mat.albedo * 0.05f;  // ambient
    Lo = Lo + rec.mat.emission;        // emission

    for (const auto& light : lights) {
        Vec3 toL = light.position - rec.p;
        float dist = length3(toL);
        if (dist <= 0.0f) continue;

        Vec3 L = toL / dist;

        float NdotL = fmaxf(dot(N, L), 0.0f);
        if (NdotL <= 0.0f) continue;

        // soft shadow factor (0..1)
        float vis = ShadowVisibility(rec.p, N, light, tris);
        if (vis <= 0.0f) continue;

        Vec3 f = EvaluateBRDF(rec.mat, N, V, L);

        Vec3 radiance = light.color * light.intensity;

#if RT_USE_DISTANCE_ATTENUATION
        float dist2 = fmaxf(dist * dist, 1e-6f);
        radiance = radiance / dist2;
#endif

        Lo = Lo + (radiance * f) * (NdotL * vis);
    }

    return Lo;
}

// Recursive tracer: direct + perfect mirror where possible
// diffuse_bounce: true = randomly select diffuse/mirror (Russian Roulette); false = only mirror
inline Vec3 TraceRay(const Ray& r,
                     const std::vector<Triangle>& tris,
                     const std::vector<Light>& lights,
                     int depth,
                     bool diffuse_bounce = true)
{
    if (depth <= 0) return make_vec3(0,0,0);

    HitRecord rec{};
    if (!IntersectScene(r, tris, RT_EPS, std::numeric_limits<double>::infinity(), rec)) {
        // Sky gradient background
        Vec3 unit_dir = unit_vector(r.direction());
        float t = 0.5f * (unit_dir.z + 1.0f);
        return make_vec3(1.0f, 1.0f, 1.0f) * (1.0f - t)
             + make_vec3(0.5f, 0.7f, 1.0f) * t;
    }

    Vec3 N = unit_vector(rec.normal);

    // 1) Direct BRDF lighting (with soft shadows)
    Vec3 Lo = ShadeDirect(r, rec, lights, tris);

    // 2) indirect: randomly select diffuse/mirror (diffuse_bounce is true, otherwise only mirror)
    float kd = rec.mat.kd;
    float kr = rec.mat.kr;
    float total = kd + kr;
    if (total > 0.0f) {
        float xi = random_float();
        if (diffuse_bounce && xi < kd / total) {
            Vec3 diffuse_dir = random_on_hemisphere(N);
            Ray diffuse_ray(rec.p + N * RT_EPS, diffuse_dir);
            Vec3 bounced = TraceRay(diffuse_ray, tris, lights, depth - 1, diffuse_bounce);
            float NdotL = fmaxf(dot(N, diffuse_dir), 0.0f);
            Lo = Lo + (rec.mat.albedo * total * 2.0f * NdotL * bounced);
        } else if (kr > 0.0f) {
            Vec3 refl = reflect_dir(unit_vector(r.direction()), N);
            Ray rr(rec.p + N * RT_EPS, refl);
            Vec3 bounced = TraceRay(rr, tris, lights, depth - 1, diffuse_bounce);
            Vec3 tint = rec.mat.specularColor;
            Lo = Lo + ((diffuse_bounce ? total : rec.mat.kr) * (tint * bounced));
        }
    }

    // Clamp once when writing the final PNG.
    return Lo;
}

#endif
