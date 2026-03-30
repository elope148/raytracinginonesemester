#pragma once

#include "camera.h"
#include "ray.h"
#include "MeshOBJ.h"
#include "brdf.h"
#include "shader.h"
#include "bvh.h"
#include "antialias.h"

struct Light;

void render(    
    const size_t numTriangles,
    int W, int H,
    const Camera cam,
    const Vec3 missColor,
    const int max_depth,
    const int spp,
    const BVHNode* __restrict__ nodes,
    const AABB* __restrict__ aabbs,
    const Triangle* __restrict__ triangles,
    const int32_t* __restrict__ triObjectIds,
    const MaterialData* __restrict__ objectMaterials,
    const int numObjectMaterials,
    const Light* __restrict__ lights,
    const int numLights,
    const bool diffuse_bounce,
    const EmissiveTriInfo* __restrict__ emissiveTris,
    const float* __restrict__ emissiveCDF,
    const int numEmissiveTris,
    const float totalEmissiveArea,
    Vec3* __restrict__ output,
    int nee_mode = 2);


HYBRID_FUNC inline float rng_next(unsigned int& state) {
    state = state * 1664525u + 1013904223u;
    // xorshift-style mixing for better distribution
    unsigned int h = state;
    h = (h ^ 61u) ^ (h >> 16u);
    h *= 9u;
    h ^= h >> 4u;
    h *= 0x27d4eb2du;
    h ^= h >> 15u;
    return float(h) / float(0xFFFFFFFFu);
}

HYBRID_FUNC inline unsigned int make_rng_seed(int x, int y, int sample) {
    return (unsigned int)x * 73856093u
         ^ (unsigned int)y * 19349663u
         ^ (unsigned int)sample * 83492791u;
}

HYBRID_FUNC inline Vec3 random_unit_vector(unsigned int& state) {
    for (;;) {
        float x = 2.0f * rng_next(state) - 1.0f;
        float y = 2.0f * rng_next(state) - 1.0f;
        float z = 2.0f * rng_next(state) - 1.0f;
        float lensq = x*x + y*y + z*z;
        if (lensq > 1e-10f && lensq <= 1.0f) {
            float inv = 1.0f / sqrtf(lensq);
            return make_vec3(x * inv, y * inv, z * inv);
        }
    }
}



HYBRID_FUNC inline Vec3 random_on_hemisphere(const Vec3& normal, unsigned int& state) {
    Vec3 on_unit_sphere = random_unit_vector(state);
    if (dot(on_unit_sphere, normal) > 0.0f)
        return on_unit_sphere;
    return make_vec3(-on_unit_sphere.x, -on_unit_sphere.y, -on_unit_sphere.z);
}

HYBRID_FUNC inline float path_luminance(const Vec3& c) {
    return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

HYBRID_FUNC inline void build_onb(const Vec3& N, Vec3& tangent, Vec3& bitangent) {
    Vec3 up = (fabsf(N.z) < 0.9f) ? make_vec3(0.0f, 0.0f, 1.0f)
                                  : make_vec3(1.0f, 0.0f, 0.0f);
    Vec3 cr = cross(up, N);
    float len = sqrtf(dot(cr, cr));
    tangent = (len > 1e-10f) ? cr * (1.0f / len) : make_vec3(1.0f, 0.0f, 0.0f);
    bitangent = cross(N, tangent);
}

HYBRID_FUNC inline Vec3 cosine_hemisphere_sample(const Vec3& N, unsigned int& rng_state) {
    const float u1 = rng_next(rng_state);
    const float u2 = rng_next(rng_state);
    const float r = sqrtf(u1);
    const float phi = 6.28318530717958647692f * u2;
    const float x = r * cosf(phi);
    const float y = r * sinf(phi);
    const float z = sqrtf(fmaxf(0.0f, 1.0f - u1));

    Vec3 tangent, bitangent;
    build_onb(N, tangent, bitangent);
    Vec3 wi = tangent * x + bitangent * y + N * z;
    float wi_len = sqrtf(dot(wi, wi));
    return (wi_len > 1e-10f) ? wi * (1.0f / wi_len) : N;
}

HYBRID_FUNC inline float BRDFSamplingPdf(const HitRecord& rec,
                                         const Vec3& wo,
                                         const Vec3& wi,
                                         const Vec3& Nshading,
                                         bool allow_diffuse = true)
{
    if (!allow_diffuse && rec.mat.ks <= 0.0f) return 0.0f;
    return allow_diffuse
         ? BRDFpdf(rec, wo, wi, Nshading)
         : BlinnPhongSpecularPDF(normalize(Nshading), wo, wi, rec.mat.shininess);
}

HYBRID_FUNC inline Vec3 sample_blinn_phong(const Vec3& N,
                                           const Vec3& V,
                                           float shininess,
                                           unsigned int& rng_state,
                                           float& pdf)
{
    const float u1 = rng_next(rng_state);
    const float u2 = rng_next(rng_state);
    const float twoPi = 6.28318530717958647692f;

    const float cos_theta_H = powf(fmaxf(u1, 1e-10f), 1.0f / (shininess + 2.0f));
    const float sin_theta_H = sqrtf(fmaxf(0.0f, 1.0f - cos_theta_H * cos_theta_H));
    const float phi = twoPi * u2;

    Vec3 tangent, bitangent;
    build_onb(N, tangent, bitangent);

    Vec3 H_raw = tangent * (sin_theta_H * cosf(phi))
               + bitangent * (sin_theta_H * sinf(phi))
               + N * cos_theta_H;
    float h_len = sqrtf(dot(H_raw, H_raw));
    Vec3 H = (h_len > 1e-10f) ? H_raw * (1.0f / h_len) : N;

    const float VdotH = fmaxf(dot(V, H), 0.0f);
    Vec3 wi_raw = H * (2.0f * VdotH) - V;
    float wi_len = sqrtf(dot(wi_raw, wi_raw));
    Vec3 wi = (wi_len > 1e-10f) ? wi_raw * (1.0f / wi_len) : N;

    pdf = BlinnPhongSpecularPDF(N, V, wi, shininess);
    return wi;
}

HYBRID_FUNC inline Vec3 SampleBRDF(const HitRecord& rec,
                                   const Vec3& Vo,
                                   const Vec3& Nshading,
                                   unsigned int& rng_state,
                                   float& out_pdf,
                                   bool allow_diffuse = true)
{
    const Vec3 N = normalize(Nshading);
    const float kd = allow_diffuse ? rec.mat.kd : 0.0f;
    const float ks = rec.mat.ks;
    const float total = kd + ks;
    if (total <= 0.0f) {
        out_pdf = 0.0f;
        return make_vec3(0.0f, 0.0f, 0.0f);
    }

    Vec3 wi;
    if (kd > 0.0f && rng_next(rng_state) < kd / total) {
        wi = cosine_hemisphere_sample(N, rng_state);
    } else {
        float dummy_pdf = 0.0f;
        wi = sample_blinn_phong(N, Vo, rec.mat.shininess, rng_state, dummy_pdf);
    }

    out_pdf = BRDFSamplingPdf(rec, Vo, wi, N, allow_diffuse);
    return wi;
}

HYBRID_FUNC inline HitRecord intersectTriangle(const Ray& r,
                                          const Triangle& tri,
                                          float tmin,
                                          float tmax)
{
    HitRecord rec{};
    rec.triangleIdx = -1;

    const Vec3 e1 = tri.v1 - tri.v0;
    const Vec3 e2 = tri.v2 - tri.v0;
    const Vec3 pvec = cross(r.direction(), e2);
    const float det = dot(e1, pvec);

    if (fabsf(det) < 1e-8f) {
        rec.hit = false;
        return rec;
    }
    const float invDet = 1.0f / det;

    const Vec3 tvec = r.origin() - tri.v0;
    const float u = dot(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f) {
        rec.hit = false;
        return rec;
    }

    const Vec3 qvec = cross(tvec, e1);
    const float v = dot(r.direction(), qvec) * invDet;
    if (v < 0.0f || (u + v) > 1.0f) {
        rec.hit = false;
        return rec;
    }
    const float t = dot(e2, qvec) * invDet;
    if (t < tmin || t > tmax) {
        rec.hit = false;
        return rec;
    }

    rec.hit = true;
    rec.t = t;
    rec.p = r.origin() + r.direction() * t;

    // Use geometric normal for sidedness (robust), shading normal for BRDF.
    Vec3 geomN = normalize(cross(e1, e2));
    rec.front_face = dot(r.direction(), geomN) < 0.0f;
    if (!rec.front_face) geomN = -geomN;

    Vec3 shadingN = (1.0f - u - v) * tri.n0 + u * tri.n1 + v * tri.n2;
    if (length_squared(shadingN) < 1e-12f) {
        shadingN = geomN;
    } else {
        shadingN = normalize(shadingN);
        // Keep shading normal in same hemisphere as geometric normal.
        if (dot(shadingN, geomN) < 0.0f) shadingN = -shadingN;
    }

    // -------------------- NEW: interpolate UVs --------------------


    rec.u = (1.0f - u - v) * tri.uv0.x + u * tri.uv1.x + v * tri.uv2.x;
    rec.v = (1.0f - u - v) * tri.uv0.y + u * tri.uv1.y + v * tri.uv2.y;

    // -------------------- NEW: COMPUTE TANGENT + BITANGENT --------------------
    Vec3 dp1 = e1;  // p1 - p0
    Vec3 dp2 = e2;  // p2 - p0
    
    Vec2 duv1 = {tri.uv1.x - tri.uv0.x, tri.uv1.y - tri.uv0.y};
    Vec2 duv2 = {tri.uv2.x - tri.uv0.x, tri.uv2.y - tri.uv0.y};
    
    float uv_det = duv1.x * duv2.y - duv2.x * duv1.y;
    
    if (fabsf(uv_det) > 1e-8f) {
        float inv_uv_det = 1.0f / uv_det;
        
        // Tangent (U direction)
        rec.tangent = normalize(
            (duv2.y * dp1 - duv1.y * dp2) * inv_uv_det
        );
        
        // Bitangent (V direction)  
        rec.bitangent = normalize(
            (duv1.x * dp2 - duv2.x * dp1) * inv_uv_det
        );
    } else {
        // Fallback: arbitrary TBN frame
        rec.tangent = unit_vector(cross(shadingN, Vec3{0,1,0}));
        if (length_squared(rec.tangent) < 1e-8f) {
            rec.tangent = unit_vector(cross(shadingN, Vec3{0,0,1}));
        }
        rec.bitangent = cross(shadingN, rec.tangent);
    }

    rec.tangent = normalize(rec.tangent - shadingN * dot(shadingN, rec.tangent));

    rec.bitangent = normalize(cross(shadingN, rec.tangent));

    rec.normal = shadingN;
    rec.mat = MaterialData();

    return rec;
}

HYBRID_FUNC inline void assignMaterialToHit(
    HitRecord& hitRecord,
    const int numTriangles,
    const int32_t* __restrict__ triObjectIds,
    const MaterialData* __restrict__ objectMaterials,
    const int numObjectMaterials)
{
    if (!hitRecord.hit ||
        triObjectIds == nullptr ||
        objectMaterials == nullptr ||
        hitRecord.triangleIdx < 0 ||
        hitRecord.triangleIdx >= numTriangles) {
        return;
    }

    const int objId = triObjectIds[hitRecord.triangleIdx];
    if (objId >= 0 && objId < numObjectMaterials) {
        hitRecord.mat = objectMaterials[objId];
    }
}

HYBRID_FUNC inline int binary_search_cdf(const float* cdf, int n, float u) {
    int lo = 0;
    int hi = n - 1;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (cdf[mid] < u) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

HYBRID_FUNC inline Vec3 TraceRayIterative(
    const Ray& primaryRay,
    const int maxDepth,
    const Vec3 missColor,
    const int numTriangles,
    const BVHNode* __restrict__ nodes,
    const AABB* __restrict__ aabbs,
    const Triangle* __restrict__ triangles,
    const int32_t* __restrict__ triObjectIds,
    const MaterialData* __restrict__ objectMaterials,
    const int numObjectMaterials,
    const Light* __restrict__ lights,
    const int numLights,
    const EmissiveTriInfo* __restrict__ emissiveTris,
    const float* __restrict__ emissiveCDF,
    const int numEmissiveTris,
    const float totalEmissiveArea,
    unsigned int rng_state = 42u,
    bool diffuse_bounce = true,
    int nee_mode = 2)
{
    if (maxDepth <= 0) return make_vec3(0.0f, 0.0f, 0.0f);

    Ray ray = primaryRay;
    Vec3 radiance = make_vec3(0.0f, 0.0f, 0.0f);
    Vec3 throughput = make_vec3(1.0f, 1.0f, 1.0f);
    float prev_pdf = 0.0f;
    bool prev_delta = false;
    Vec3 prev_N = make_vec3(0.0f, 0.0f, 0.0f);

    for (int depth = 0; depth < maxDepth; ++depth) {
        HitRecord hitRecord;
        SearchBVH(numTriangles, ray, nodes, aabbs, triangles, hitRecord);
        if (!hitRecord.hit) {
            radiance = radiance + throughput * missColor;
            break;
        }

        assignMaterialToHit(hitRecord, numTriangles, triObjectIds, objectMaterials, numObjectMaterials);

        const Vec3 N = normalize(hitRecord.normal);
        const Vec3 Ns = GetShadingNormal(hitRecord);
        const Vec3 Vo = unit_vector(-ray.direction());

        const Vec3 Le = hitRecord.mat.emission;
        if (Le.x > 0.0f || Le.y > 0.0f || Le.z > 0.0f) {
            if (depth == 0 || prev_delta) {
                radiance = radiance + throughput * Le;
            } else if (nee_mode == 1) {
                radiance = radiance + throughput * Le;
            } else if (nee_mode == 2 &&
                       numEmissiveTris > 0 &&
                       totalEmissiveArea > 1e-10f &&
                       prev_pdf > 1e-6f) {
                Vec3 hitGeomN = normalize(cross(
                    triangles[hitRecord.triangleIdx].v1 - triangles[hitRecord.triangleIdx].v0,
                    triangles[hitRecord.triangleIdx].v2 - triangles[hitRecord.triangleIdx].v0));
                float cosLight = fabsf(dot(hitGeomN, -unit_vector(ray.direction())));
                float dist2 = static_cast<float>(hitRecord.t * hitRecord.t);
                float G = (dist2 > 1e-10f) ? (cosLight / dist2) : 0.0f;
                float pdf_area_sa = (G > 1e-10f) ? (1.0f / totalEmissiveArea) / G : 0.0f;
                float NdotWi_prev = fmaxf(dot(prev_N, unit_vector(ray.direction())), 0.0f);
                float pdf_cos = NdotWi_prev * 0.31830988618f;
                float pdf_nee = (pdf_area_sa + pdf_cos + prev_pdf) / 3.0f;

                float p2_brdf = prev_pdf * prev_pdf;
                float p2_nee = pdf_nee * pdf_nee;
                float w_brdf = (p2_brdf + p2_nee > 0.0f)
                             ? p2_brdf / (p2_brdf + p2_nee)
                             : 0.0f;
                radiance = radiance + throughput * Le * w_brdf;
            }
        }

        Vec3 direct = ShadeDirect(ray, hitRecord, lights, numLights,
                                  numTriangles, nodes, aabbs, triangles);
        radiance = radiance + throughput * direct;

        if (numEmissiveTris > 0 && totalEmissiveArea > 1e-10f && nee_mode != 1) {
            int strategy = 0;
            if (nee_mode != 0) {
                const float strategy_u = rng_next(rng_state);
                strategy = (strategy_u < 0.333333f) ? 0
                         : (strategy_u < 0.666667f) ? 1 : 2;
            }

            Vec3 wi_nee = make_vec3(0.0f, 0.0f, 0.0f);
            Vec3 Le_nee = make_vec3(0.0f, 0.0f, 0.0f);
            float dist_nee = 0.0f;
            float cosLight_nee = 0.0f;
            float NdotL_nee = 0.0f;
            bool nee_valid = false;

            if (strategy == 0) {
                const float u_sel = rng_next(rng_state);
                const int eidx = binary_search_cdf(emissiveCDF, numEmissiveTris, u_sel);
                const EmissiveTriInfo& emi = emissiveTris[eidx];
                const Triangle& eTri = triangles[emi.triangleIdx];

                float u1 = rng_next(rng_state);
                float u2 = rng_next(rng_state);
                if (u1 + u2 > 1.0f) {
                    u1 = 1.0f - u1;
                    u2 = 1.0f - u2;
                }

                const Vec3 lightPoint = eTri.v0 * (1.0f - u1 - u2)
                                      + eTri.v1 * u1
                                      + eTri.v2 * u2;

                Vec3 toLight = lightPoint - hitRecord.p;
                const float r2 = dot(toLight, toLight);
                if (r2 > 1e-10f) {
                    const float r = sqrtf(r2);
                    wi_nee = toLight * (1.0f / r);
                    NdotL_nee = fmaxf(dot(Ns, wi_nee), 0.0f);
                    cosLight_nee = fabsf(dot(emi.normal, -wi_nee));

                    if (NdotL_nee > 0.0f && cosLight_nee > 0.0f) {
                        Ray shadowRay(hitRecord.p + N * RT_EPS, wi_nee);
                        HitRecord shadowHit{};
                        shadowHit.hit = false;
                        SearchBVH(numTriangles, shadowRay, nodes, aabbs, triangles, shadowHit);

                        if (!shadowHit.hit || shadowHit.t >= r - RT_EPS) {
                            Le_nee = emi.emission;
                            dist_nee = r;
                            nee_valid = true;
                        }
                    }
                }
            } else {
                if (strategy == 1) {
                    wi_nee = cosine_hemisphere_sample(Ns, rng_state);
                } else {
                    float dummy_pdf = 0.0f;
                    wi_nee = SampleBRDF(hitRecord, Vo, Ns, rng_state, dummy_pdf, diffuse_bounce);
                }

                NdotL_nee = fmaxf(dot(Ns, wi_nee), 0.0f);
                if (NdotL_nee > 0.0f) {
                    Ray neeRay(hitRecord.p + N * RT_EPS, wi_nee);
                    HitRecord neeHit{};
                    neeHit.hit = false;
                    SearchBVH(numTriangles, neeRay, nodes, aabbs, triangles, neeHit);

                    if (neeHit.hit) {
                        assignMaterialToHit(neeHit, numTriangles, triObjectIds, objectMaterials, numObjectMaterials);
                        const Vec3 Le_hit = neeHit.mat.emission;
                        if (Le_hit.x > 0.0f || Le_hit.y > 0.0f || Le_hit.z > 0.0f) {
                            Vec3 geomN = normalize(cross(
                                triangles[neeHit.triangleIdx].v1 - triangles[neeHit.triangleIdx].v0,
                                triangles[neeHit.triangleIdx].v2 - triangles[neeHit.triangleIdx].v0));
                            cosLight_nee = fabsf(dot(geomN, -wi_nee));
                            dist_nee = static_cast<float>(neeHit.t);
                            Le_nee = Le_hit;
                            nee_valid = true;
                        }
                    }
                }
            }

            if (nee_valid && dist_nee > 1e-6f && cosLight_nee > 1e-6f) {
                const float r2 = dist_nee * dist_nee;
                const float G = cosLight_nee / r2;
                const float pdf_area_sa = (1.0f / totalEmissiveArea) / G;
                const Vec3 f_nee = EvaluateBRDF(hitRecord, Vo, wi_nee, Ns);

                if (nee_mode == 0) {
                    if (pdf_area_sa > 1e-10f) {
                        const Vec3 nee_contrib = Le_nee * f_nee * (NdotL_nee / pdf_area_sa);
                        radiance = radiance + throughput * nee_contrib;
                    }
                } else {
                    const float pdf_cos = NdotL_nee * 0.31830988618f;
                    const float pdf_brdf = BRDFSamplingPdf(hitRecord, Vo, wi_nee, Ns, diffuse_bounce);
                    const float pdf_combined = (pdf_area_sa + pdf_cos + pdf_brdf) / 3.0f;

                    if (pdf_combined > 1e-10f) {
                        const float p2_nee = pdf_combined * pdf_combined;
                        const float p2_brdf = pdf_brdf * pdf_brdf;
                        const float w_nee = (p2_nee + p2_brdf > 0.0f)
                                          ? p2_nee / (p2_nee + p2_brdf)
                                          : 0.0f;
                        const Vec3 nee_contrib = Le_nee * f_nee
                                               * (NdotL_nee / pdf_combined) * w_nee;
                        radiance = radiance + throughput * nee_contrib;
                    }
                }
            }
        }

        prev_N = Ns;

        const float kd = hitRecord.mat.kd;
        const float ks = hitRecord.mat.ks;
        const float kr = hitRecord.mat.kr;
        if (kd > 1e-6f || ks > 1e-6f) {
            float pdf = 0.0f;
            Vec3 wi = SampleBRDF(hitRecord, Vo, Ns, rng_state, pdf, diffuse_bounce);
            if (pdf < 1e-6f || dot(wi, Ns) < 0.0f) break;
            ray = Ray(hitRecord.p + N * RT_EPS, wi);
            const float NdotWi = fmaxf(dot(Ns, wi), 0.0f);
            const Vec3 f = EvaluateBRDF(hitRecord, Vo, wi, Ns);
            throughput = throughput * (f * (NdotWi / pdf));
            prev_pdf = pdf;
            prev_delta = false;
        } else if (kr > 1e-6f) {
            const Vec3 reflDir = reflect_dir(unit_vector(ray.direction()), Ns);
            ray = Ray(hitRecord.p + N * RT_EPS, reflDir);
            throughput = throughput * (hitRecord.mat.specularColor * kr);
            prev_pdf = 0.0f;
            prev_delta = true;
        } else {
            break;
        }

        const float p_survive = fminf(path_luminance(throughput), 0.95f);
        if (p_survive < 1e-4f || rng_next(rng_state) > p_survive) break;
        throughput = throughput * (1.0f / p_survive);
    }

    return radiance;
}



HYBRID_FUNC inline void SearchBVH(
    const int numTriangles,
    const Ray& ray,
    const BVHNode* __restrict__ nodes,
    const AABB* __restrict__ aabbs,
    const Triangle* __restrict__ triangles,
    HitRecord& hitRecord)
{

    constexpr float kRayTMin = 1e-4f;
    const float tmin = kRayTMin;
    float bestT = FLT_MAX;
    HitRecord bestHit;
    bestHit.triangleIdx = -1;
    bestHit.hit = false;
    bestHit.t = -1.0;
    bestHit.p = make_vec3(0.0f, 0.0f, 0.0f);
    bestHit.normal = make_vec3(0.0f, 0.0f, 0.0f);
    bestHit.front_face = false;
    bestHit.mat = MaterialData();

    constexpr int STACK_CAPACITY = 512;
    std::uint32_t stack[STACK_CAPACITY];
    std::uint32_t* stack_ptr = stack;
    bool stackOverflow = false;
    *stack_ptr++ = 0; // root node is always 0
    

    while (stack_ptr > stack) {
        const std::uint32_t nodeIdx = *--stack_ptr;

        if (!intersectAABB(ray, aabbs[nodeIdx], tmin, bestT)) {
            continue;
        }

        const BVHNode node = nodes[nodeIdx];
        const std::uint32_t obj_idx = node.object_idx;

        if (obj_idx != 0xFFFFFFFF) {
            if (obj_idx < static_cast<std::uint32_t>(numTriangles)) {
                HitRecord rec = intersectTriangle(ray, triangles[obj_idx], tmin, bestT);
                if (rec.hit) {
                    rec.triangleIdx = static_cast<int>(obj_idx);
                    bestT = rec.t;
                    bestHit = rec;
                }
            }
            continue;
        }

        const std::uint32_t left_idx = node.left_idx;
        const std::uint32_t right_idx = node.right_idx;

        if (left_idx != 0xFFFFFFFF) {
            if (intersectAABB(ray, aabbs[left_idx], tmin, bestT)) {
                if (stack_ptr - stack < STACK_CAPACITY) {
                    *stack_ptr++ = left_idx;
                } else {
                    stackOverflow = true;
                }
            }
        }

        if (right_idx != 0xFFFFFFFF) {
            if (intersectAABB(ray, aabbs[right_idx], tmin, bestT)) {
                if (stack_ptr - stack < STACK_CAPACITY) {
                    *stack_ptr++ = right_idx;
                } else {
                    stackOverflow = true;
                }
            }
        }
    }

    // Safety fallback: if traversal overflowed, complete with brute-force test to avoid artifacts.
    if (stackOverflow) {
        for (int i = 0; i < numTriangles; ++i) {
            HitRecord rec = intersectTriangle(ray, triangles[i], tmin, bestT);
            if (rec.hit) {
                rec.triangleIdx = i;
                bestT = rec.t;
                bestHit = rec;
            }
        }
    }

    hitRecord = bestHit;
}
