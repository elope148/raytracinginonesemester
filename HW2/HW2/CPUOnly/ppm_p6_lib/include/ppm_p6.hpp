#pragma once
// ppm_p6.hpp (C++17, standard library only)
//
// -----------------------------------------------------------------------------//  ppm_p6: PPM P6 (binary) read/write module for ray tracers
//
//  What this module does:
//    - Writes PPM P6 (binary) from an Image buffer of linear RGB doubles in [0..1].
//    - Reads PPM P6 (binary) into the same Image format (linear RGB doubles in [0..1]).
//    - Supports maxval < 256 (8-bit) and maxval >= 256 (16-bit, MSB-first/big-endian).
//    - Robust header parsing: skips whitespace and '#' comments anywhere in the header.
//
//  Public API (what teammates use):
//    namespace ppm_p6 {
//      struct Color;
//      struct WriteOptions;
//      class Image;
//      bool write_p6(path, img, opt, err);
//      bool read_p6(path, img, err);
//    }
//
//  How to use in your ray tracer:
//    - include this header
//    - compile/link ppm_p6.cpp exactly once in the whole project
//
//  MSVC build example:
//    cl /std:c++17 /EHsc your_main.cpp ppm_p6.cpp /Fe:app.exe
//
// -----------------------------------------------------------------------------

#include <cstddef>   // size_t
#include <string>    // std::string
#include <vector>    // std::vector

// Everything in a namespace ppm_p6 to avoid collisions in team repo.

namespace ppm_p6{

// Storing color in linear space as double [0...1]
struct Color {
    double r = 0.0;
    double g = 0.0;
    double b = 0.0;
};

// options for writing to disk
struct WriteOptions{
    int maxval = 255;
    bool clamp = true;   // clamp linear channedl to [0...1]
    bool gamma2 = true;  // apply sqrt(linear) before scaling
    bool flip_y = false; // write roes bottom-to-top
};

// minimal image container
class Image{
public:
    Image();
    Image(int w, int h);
    void resize(int w, int h);

    int width() const;  //get width
    int height() const; //get height

    Color& at(int x, int y);
    const Color& at(int x, int y) const;

    void set (int x, int y, const Color& c); // conversion

    // direct buffer access
    std::vector<Color>& pixels();                 // direct buffer access (mutable)
    const std::vector<Color>& pixels() const;     // direct buffer access (const)

private:
    size_t index(int x, int y) const; // map (x,y) to linear index

    int width_ = 0;
    int height_ = 0;
    std::vector<Color> pixels_;
};

// write_p6 file
bool write_p6(const std::string& path, const Image& img, const WriteOptions& opt, std::string* err);

// read p6 file.
// overwrites the image with loaded data
bool read_p6(const std::string& path, Image& img, std::string* err);

} // namespace ppm_p6
