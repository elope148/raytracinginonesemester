#include "MeshOBJ.h"

#include <cstdio>
#include <cstdlib>
#include <unordered_map>

// Key representing a unique OBJ vertex reference (pos/uv/norm). Assuming zero-based for now.
struct VertexKey
{
    int p = -1; // position index (0-based)
    int t = -1; // texcoord index (0-based)
    int n = -1; // normal index (0-based)

    bool operator==(const VertexKey& o) const { return p == o.p && t == o.t && n == o.n; }
};

struct VertexKeyHash
{
    size_t operator()(const VertexKey& k) const
    {
        // Simple hash combine. (Speed up referencing/checking)
        size_t h1 = std::hash<int>()(k.p);
        size_t h2 = std::hash<int>()(k.t);
        size_t h3 = std::hash<int>()(k.n);
        size_t h = h1;
        h ^= (h2 + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2));
        h ^= (h3 + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2));
        return h;
    }
};

static void SkipWS(const char*& s)
{
    while (*s == ' ' || *s == '\t') ++s;
}

static bool ParseInt(const char*& s, int& out)
{
    SkipWS(s);

    bool neg = false;
    if (*s == '-') { neg = true; ++s; } // NOTE: assuming we do not implement negative OBJ semantics!!!

    if (*s < '0' || *s > '9') return false;

    int v = 0;
    while (*s >= '0' && *s <= '9')
    {
        v = v * 10 + (*s - '0');
        ++s;
    }

    out = neg ? -v : v;
    return true;
}

static bool ParseFloat(const char*& s, float& out)
{
    SkipWS(s);
    char* endPtr = nullptr;
    out = std::strtof(s, &endPtr);
    if (endPtr == s) return false;
    s = endPtr;
    return true;
}

// Parse one face vertex token: "v/vt/vn", "v//vn", "v/vt", or "v"
static bool ParseFaceVertex(const char*& s, VertexKey& outKey)
{
    int v = 0;
    if (!ParseInt(s, v)) return false;

    outKey.p = v - 1; // (OBJ is 1-based)
    outKey.t = -1;
    outKey.n = -1;

    if (*s != '/') return true; // only "v"

    ++s; // skip '/'

    if (*s == '/')
    {
        // "v//vn"
        ++s;
        int n = 0;
        if (!ParseInt(s, n)) return false;
        outKey.n = n - 1;
        return true;
    }

    // "v/vt" or "v/vt/vn"
    int t = 0;
    if (ParseInt(s, t)) outKey.t = t - 1;

    if (*s != '/') return true;

    ++s; // skip '/'

    int n = 0;
    if (ParseInt(s, n)) outKey.n = n - 1;

    return true;
}

static uint32_t GetOrCreateVertex(
    const VertexKey& key,
    const std::vector<Vec3>& rawPos,
    const std::vector<Vec2>& rawUV,
    const std::vector<Vec3>& rawNrm,
    MeshSOA& out,
    std::unordered_map<VertexKey, uint32_t, VertexKeyHash>& dedup,
    bool wantUV,
    bool wantNrm)
{
    auto it = dedup.find(key);
    if (it != dedup.end())
        return it->second;

    uint32_t idx = (uint32_t)out.positions.size();
    dedup[key] = idx;

    // position required
    out.positions.push_back(rawPos.at((size_t)key.p));

    // optional streams: if file uses them, keep arrays aligned.
    if (wantUV)
    {
        Vec2 uv{};
        if (key.t >= 0 && key.t < (int)rawUV.size()) uv = rawUV[(size_t)key.t];
        out.uvs.push_back(uv);
    }

    if (wantNrm)
    {
        Vec3 n{};
        if (key.n >= 0 && key.n < (int)rawNrm.size()) n = rawNrm[(size_t)key.n];
        out.normals.push_back(n);
    }

    return idx;
}

bool LoadOBJ_ToMeshSOA(const std::string& path, MeshSOA& outMesh)
{
    outMesh = MeshSOA{}; // reset

    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;

    std::vector<Vec3> rawPos;
    std::vector<Vec2> rawUV;
    std::vector<Vec3> rawNrm;

    bool fileHasUV = false;
    bool fileHasNrm = false;

    std::unordered_map<VertexKey, uint32_t, VertexKeyHash> dedup;
    dedup.reserve(10000);

    char line[1024];

    while (std::fgets(line, sizeof(line), f))
    {
        const char* s = line;
        SkipWS(s);

        if (*s == '\0' || *s == '\n' || *s == '#')
            continue;

        // v x y z
        if (s[0] == 'v' && (s[1] == ' ' || s[1] == '\t'))
        {
            s += 1;
            Vec3 p{};
            if (!ParseFloat(s, p.x) || !ParseFloat(s, p.y) || !ParseFloat(s, p.z))
            {
                std::fclose(f);
                return false;
            }
            rawPos.push_back(p);
            continue;
        }

        // vt u v
        if (s[0] == 'v' && s[1] == 't' && (s[2] == ' ' || s[2] == '\t'))
        {
            s += 2;
            Vec2 uv{};
            if (!ParseFloat(s, uv.x) || !ParseFloat(s, uv.y))
            {
                std::fclose(f);
                return false;
            }
            rawUV.push_back(uv);
            fileHasUV = true;
            continue;
        }

        // vn x y z
        if (s[0] == 'v' && s[1] == 'n' && (s[2] == ' ' || s[2] == '\t'))
        {
            s += 2;
            Vec3 n{};
            if (!ParseFloat(s, n.x) || !ParseFloat(s, n.y) || !ParseFloat(s, n.z))
            {
                std::fclose(f);
                return false;
            }
            rawNrm.push_back(n);
            fileHasNrm = true;
            continue;
        }

        // f ...
        if (s[0] == 'f' && (s[1] == ' ' || s[1] == '\t'))
        {
            s += 1;

            VertexKey keys[4];
            int count = 0;

            while (count < 4)
            {
                SkipWS(s);
                if (*s == '\0' || *s == '\n') break;

                VertexKey k{};
                if (!ParseFaceVertex(s, k))
                    break;

                if (k.t >= 0) fileHasUV = true;
                if (k.n >= 0) fileHasNrm = true;

                keys[count++] = k;

                // move to next token
                while (*s != '\0' && *s != '\n' && *s != ' ' && *s != '\t') ++s;
            }

            if (count < 3)
            {
                std::fclose(f);
                return false;
            }

            // create/reuse unified vertex indices
            uint32_t i0 = GetOrCreateVertex(keys[0], rawPos, rawUV, rawNrm, outMesh, dedup, fileHasUV, fileHasNrm);
            uint32_t i1 = GetOrCreateVertex(keys[1], rawPos, rawUV, rawNrm, outMesh, dedup, fileHasUV, fileHasNrm);
            uint32_t i2 = GetOrCreateVertex(keys[2], rawPos, rawUV, rawNrm, outMesh, dedup, fileHasUV, fileHasNrm);

            outMesh.indices.push_back(i0);
            outMesh.indices.push_back(i1);
            outMesh.indices.push_back(i2);

            // quad -> two triangles (assuming we only handle triangles!!!)
            if (count == 4)
            {
                uint32_t i3 = GetOrCreateVertex(keys[3], rawPos, rawUV, rawNrm, outMesh, dedup, fileHasUV, fileHasNrm);
                outMesh.indices.push_back(i0);
                outMesh.indices.push_back(i2);
                outMesh.indices.push_back(i3);
            }

            continue;
        }

        // ignore other lines (o, g, s, mtllib, usemtl, etc.)
    }

    std::fclose(f);

    if (outMesh.positions.empty() || outMesh.indices.empty())
        return false;

    // If OBJ never used UV/norm, keep those arrays empty for now (we can add implementation later).
    // If used, sizes should match positions:
    if (fileHasUV && outMesh.uvs.size() != outMesh.positions.size()) return false;
    if (fileHasNrm && outMesh.normals.size() != outMesh.positions.size()) return false;

    return true;
}
