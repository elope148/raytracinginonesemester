// Unit-testing for Ray-Triangle intersection.
// Author - Anish Vipperla

#include <catch2/catch_test_macros.hpp>
#include <iostream>
#include <cstdint>
#include <string.h>
#include "ray.h"

#define ALPHA_MIN 0.0f
#define ALPHA_MAX 1.0f
#define BETA_MIN  0.0f

#define GRID_STEP 0.1f
#define ORIGIN    (Vec3 {0.0f,0.0f,0.0f})

TEST_CASE( "Ray-Triangle Intersection Test", "[testcase1]" ) {

    HitRecord HITRec;

    Vec3 v0 {-5.0f, -5.0f, -10.0f};
    Vec3 v1 {  0.0f, 5.0f, -10.0f};
    Vec3 v2 { 5.0f, -5.0f, -10.0f};

    Vec3 n0 {0.0f, 0.0f, 1.0f};
    Vec3 n1 {0.0f, 0.0f, 1.0f};
    Vec3 n2 {0.0f, 0.0f, 1.0f};

    Triangle T { v0, v1, v2, n0, n1, n2};

    // std::cout << "Vertex Hit Test\n";
    Ray R1 (ORIGIN, Vec3{0.0f, 5.0f, -10.0f});
    HITRec =  ray_intersection(R1, T);
    REQUIRE (HITRec.hit==true);    //Ray hits vertex1
    std::cout << "Vertex Hit Test Passed\n";


    // std::cout << "Inside Triangle Test\n";
    // For testing purpose, equate ray direction to be same as point inside the triangle.
    Ray R2 (ORIGIN, Vec3{0.0f, 0.0f, -10.0f});
    HITRec =  ray_intersection(R2, T);
    REQUIRE (HITRec.hit == true);
    std::cout << "Inside Triangle Test Passed\n";


    // std::cout << "Outside Triangle Test\n";
    // For testing purpose, equate ray direction to be same as point inside the triangle.
    Ray R3 (ORIGIN, Vec3{0.0f, 20.0f, -10.0f});
    HITRec =  ray_intersection(R3, T);
    REQUIRE (HITRec.hit == false);
    std::cout << "Outside Triangle Test Passed\n";


    // std::cout << "Triangle Edge Test\n";
    // For testing purpose, let's use the midpoint of an edge of a triangle
    Vec3 pointOnEdge = (v2 + v1)*0.5; 
    Ray R4 (ORIGIN, pointOnEdge);
    HITRec =  ray_intersection(R4, T);
    REQUIRE (HITRec.hit == true);
    std::cout << "Triangle Edge Test Passed\n";


    // std::cout << "Parallel Ray Test\n";
    Ray R5 (ORIGIN, Vec3{5.0f, 0.0f, 0.0f});
    HITRec =  ray_intersection(R5, T);
    REQUIRE (HITRec.hit==false);  // Ray is parallel to the plane
    std::cout << "Parallel Ray Test Passed\n";

    //std::cout << "Behind Origin Test\n";
    // Choose a point that would be inside the triangle but in different z-axis.
    Ray R6 (ORIGIN, Vec3{0.0f, 0.0f, 10.0f});
    HITRec =  ray_intersection(R6, T);
    REQUIRE (HITRec.hit == false);
    std::cout << "Behind Origin Test Passed\n";


    //std::cout << "Close to Triangle Edge (Inside) Test\n";
    // Choose a point close to v2-v0
    Ray R7 (ORIGIN, Vec3{0.0f, -4.999f, -10.0f});
    HITRec =  ray_intersection(R7, T);
    REQUIRE (HITRec.hit == true);
    std::cout << "Close to Triangle Edge (Inside) Test Passed\n";


    //std::cout << "Close to Triangle Edge (Outside) Test\n";
    // Choose a point close to v2-v0
    Ray R8 (ORIGIN, Vec3{0.0f, -5.001f, -10.0f});
    HITRec =  ray_intersection(R8, T);
    REQUIRE (HITRec.hit == false);
    std::cout << "Close to Triangle Edge (Outside) Test Passed\n";

}


TEST_CASE( "Ray-Triangle Intersection Test (SWEEP)", "[testcase2]" ) {
    HitRecord HITRec;

    Vec3 v0 {-5.0f, -5.0f, -10.0f};
    Vec3 v1 {  0.0f, 5.0f, -10.0f};
    Vec3 v2 { 5.0f, -5.0f, -10.0f};

    Vec3 n0 {0.0f, 0.0f, 1.0f};
    Vec3 n1 {0.0f, 0.0f, 1.0f};
    Vec3 n2 {0.0f, 0.0f, 1.0f};

    Triangle T { v0, v1, v2, n0, n1, n2};

    std::cout << "Ray-Triangle Intersection SWEEP Test\n";
    std::cout << "Shooting rays at grid with " << GRID_STEP << " step size\n";
    // P = R(t) = O + t*D;
    // Assuming t=1 and O is (0.0f,0.0f,0.0f)
    // According to Barycentric coordinates, point P = alpha*v0 + beta*v1 + gamma*v2;
    // Where, barycentric coordinates (alpha, beta, gamma) satisfy -
    // alpha, beta, gamma >=0 and <=1, alpha + beta + gamma =1
    
    for (float alpha=ALPHA_MIN; alpha<=ALPHA_MAX; alpha+=GRID_STEP){
        for (float beta=BETA_MIN; beta<=1-alpha; beta+=GRID_STEP){
            float gamma = 1-alpha-beta;
            Vec3 ray = alpha*v0 + beta*v1 + gamma*v2;

            // Function Under Test
            HITRec =  ray_intersection(Ray{ORIGIN, ray}, T);
            CHECK (HITRec.hit==true);
        }
    }
}
