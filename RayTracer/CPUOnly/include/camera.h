#ifndef CAMERA_H
#define CAMERA_H

#include "vec3.h"
#include <stdexcept>
#include <cmath>

class camera {
  public:
    using vec3   = Vec3;
    using point3 = Vec3;

    int pixel_width;
    int pixel_height;

    // Note: The defaults here will not be used when rendering, for the real defaults see scene_loader.h/CameraParams
    camera(point3 pos = make_vec3(0, 0, 0), 
           point3 lookAt = make_vec3(0, 1, 0),
           vec3 up = make_vec3(0, 0, 1),
           double focal_length_mm = 50.0,     // e.g. 50mm
           double sensor_height_mm = 24.0,    // e.g. 24mm (full-frame)
           double sensor_width_mm = 36.0,     // e.g. 36mm (full-frame, 3:2 aspect ratio)
           int width = 540, int height = 360)
        : center(pos), look_at(lookAt), up_vector(up), 
          focal_length_mm(focal_length_mm), sensor_height_mm(sensor_height_mm),
          sensor_width_mm(sensor_width_mm),
          pixel_width(width), pixel_height(height) {
            initialize();
          }

    point3 get_center() const {
        return center;
    }

    // Get pixel position by x, y index (integer = pixel center)
    point3 get_pixel_position(int i, int j) const {
        return pixel00_loc + (double(i) * pixel_delta_u) + (double(j) * pixel_delta_v);
    }

    // Sub-pixel position for jittered sampling (i, j can be fractional, e.g. i+[0,1), j+[0,1))
    point3 get_pixel_position(double i, double j) const {
        return pixel00_loc + (i * pixel_delta_u) + (j * pixel_delta_v);
    }

  private:
    point3 center;              // Camera center
    point3 look_at;             // Look-at point (optical axis direction)
    vec3   up_vector;           // Up vector (up direction)
    double focal_length_mm;
    double sensor_height_mm;
    double sensor_width_mm;
    point3 pixel00_loc;         // 3D location of pixel 0, 0
    vec3   pixel_delta_u;       // Offset to pixel to the right
    vec3   pixel_delta_v;       // Offset to pixel to the bottom


    static vec3 unit_vector(const vec3& v, const vec3& fallback = make_vec3(0.0, 0.0, 1.0)) {
        float len = sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
        const double EPS = 1e-12;
        if (len < EPS) return fallback;
        return v / len;
    }

    void initialize() {
        // Validate image dimensions
        if (pixel_width < 1) {
            throw std::runtime_error("Error: pixel_width must be >= 1");
        }
        if (pixel_height < 1) {
            throw std::runtime_error("Error: pixel_height must be >= 1");
        }

        // Calculate camera coordinate system using lookAt and up vector
        vec3 forward = unit_vector(look_at - center);
        vec3 right = unit_vector(cross(forward, up_vector));
        vec3 up_corrected = cross(right, forward);  // ensure it's orthogonal to forward and right

        // convert millimeters to meters (world units)
        double focal_length_m = focal_length_mm / 1000.0;
        double sensor_height_m = sensor_height_mm / 1000.0;
        double sensor_width_m = sensor_width_mm / 1000.0;

        // Compute vertical field of view (vfov)
        // vertical fov = 2 * atan(sensor_height / (2 * focal_length))
        // double vfov_rad = 2.0 * std::atan(sensor_height_m / (2.0 * focal_length_m));
        // horizontal fov = 2 * atan(sensor_width / (2 * focal_length))
        // double hfov_rad = 2.0 * std::atan(sensor_width_m / (2.0 * focal_length_m));

        // Compute viewport size (for a pinhole camera, viewport size = sensor size)
        double viewport_height = sensor_height_m;
        double viewport_width = sensor_width_m;

        // Calculate the vectors across the horizontal and down the vertical viewport edges.
        // viewport_u follows right, viewport_v follows down
        vec3 viewport_u = viewport_width * right;
        vec3 viewport_v = -viewport_height * up_corrected;
        pixel_delta_u = viewport_u / double(pixel_width);
        pixel_delta_v = viewport_v / double(pixel_height);

        // Calculate the location of the upper left pixel.
        point3 viewport_center = center + focal_length_m * forward;  // The viewport center is focal_length_m in front of the camera
        point3 viewport_upper_left = viewport_center - (viewport_u * 0.5) - (viewport_v * 0.5);
        pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);
    }
};

#endif
