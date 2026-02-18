#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include "vec3.h"

// Simple structures, can be updated later
struct Vec2 { float x = 0.0f, y = 0.0f; };
// struct Vec3 { float x = 0.0f, y = 0.0f, z = 0.0f; };

// Mesh (SoA formatting) output.
struct MeshSOA
{
    std::vector<Vec3> positions;   // per-vertex positions
    std::vector<Vec3> normals;     // normals (optional)
    std::vector<Vec2> uvs;         // uvs (optional)
    std::vector<uint32_t> indices; // triangles (3 per tri)

    bool hasNormals() const { return !normals.empty(); } // check if the mesh has normals
    bool hasUVs() const { return !uvs.empty(); } // check if the mesh has uvs
};

// Loads OBJ into a unified indexed mesh.
// Supports: v, vt, vn, f (tri/quad, quad->triangulated).
bool LoadOBJ_ToMeshSOA(const std::string& path, MeshSOA& outMesh);
