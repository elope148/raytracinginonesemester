#ifndef RAY_H
#define RAY_H
#include "vec3.h"
#include <iostream>
#include <cmath>
#include <limits>

enum MaterialType { MAT_LAMBERTIAN, MAT_METAL };

struct Material {
    MaterialType type;
    Vec3 albedo;
    float fuzz;
    float shininess;
}; // Should go into its own header file

class Ray {
    using point3 = Vec3;
    public:

        Ray() {

        }

        Ray(const point3& origin, const Vec3& direction) : orig(origin), dir(unit_vector(direction)) {

        }

        const point3& origin() const { return orig; }
        const Vec3& direction() const { return dir; }

        point3 at(double t) const {
            return orig + t*dir;
        }

    private:
        point3 orig;
        Vec3 dir;
};

struct Triangle {

    Vec3 v0;
    Vec3 v1;
    Vec3 v2;

    Vec3 n0;
    Vec3 n1;
    Vec3 n2;
}; // Helper struct, should maybe go into mesh processing file?

struct HitRecord {
    
    bool hit;
    Vec3 p; // position of hit on ray
    Vec3 normal;
    double t; // parameter used to define point on ray direction of hit
    bool front_face;
    Material mat;

    void set_face_normal(const Ray& r, const Vec3& outward_normal) {
        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    } // always points normals against the ray
};

HitRecord ray_intersection(const Ray& r, const Triangle& tri) {

    HitRecord rec;
    float eps = std::numeric_limits<float>::epsilon();

    Vec3 edge_vec_1 = tri.v1 - tri.v0;
    Vec3 edge_vec_2 = tri.v2 - tri.v0;

    auto pvec = cross(r.direction(), edge_vec_2);
    auto det = dot(pvec, edge_vec_1);

    // can add culling here
    if (std::abs(det) < eps) {
        rec.hit = false;
        return rec;
    }

    float invDet = 1.0 / det;

    Vec3 tvec = r.origin() - tri.v0;
    auto u = dot(tvec, pvec) * invDet;
    if (u < 0.0 || u > 1.0) {
        rec.hit = false;
        return rec;
    }

    Vec3 qvec = cross(tvec, edge_vec_1);
    auto v = dot(r.direction(), qvec) * invDet;
    if (v < 0.0 || u + v > 1.0) {
        rec.hit = false;
        return rec;
    }

    auto t = dot(edge_vec_2, qvec) * invDet;
    if (t < 0.0) {
        rec.hit = false;
        return rec;
    }

    rec.hit = true;
    rec.t = t;
    rec.p = r.at(t);
    rec.normal = (1 - u - v) * tri.n0 + u * tri.n1 + v * tri.n2;
    // rec.albedo, rec.normal, rec. etcetc, should we just store u, v values and calculate these later?
    rec.mat.type = MAT_METAL;
    rec.mat.albedo = make_vec3(0.8f, 0.2f, 0.2f);
    rec.mat.fuzz = 0.0f;
    rec.mat.shininess = 64.0f; // is hardcoded for now

    return rec;
}

#endif