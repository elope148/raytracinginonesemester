#include "ppm_p6.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>     // EOF (MSVC can require this)
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

// Everything in a namespace ppm_p6 to avoid collisions in team repo.

namespace ppm_p6{

//========================================
// Internal helpers
//========================================

namespace detail{
    // Helper: whitespcae + comment skipping for PPM heder
// return true if a character code is whitespace
static inline bool is_ws(int ch){
    // std::isspcae epects unsign char cast to avoid UB on negative char value
    return (ch != EOF) && (std::isspace(static_cast<unsigned char>(ch)) != 0);
}

// skip the whitespcae and #...endlines in the header
static void skip_ws_and_comments(std::istream& in){
    while (true)
    {
        // peak next char
        int p = in.peek();

        // if EOF, skip
        if (p == EOF){
            return;
        }
        // if whitespace consume and continue
         if (is_ws(p)){
            in.get();
            continue;
        }

        //if comment, consume '#' then ignore until newline.

        if (p == '#'){
            in.get();
            in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            continue;

        }
        // start of the token
        return;
    }

}

// Read the next ASCII token from the header.

static bool next_token(std::istream& in, std::string& token, std::string* err){

    // claeer output token
    token.clear();

    //skip whitespace/comments
    skip_ws_and_comments(in);

    // if we reached EOF, we cannot read the token
    if (in.peek() == EOF){
        if(err) *err= "Unexpected EOF while reading PPM header.";
        return false;
    }

    // Build token until we hit whitespace or "#"
    while (true)
    {
        int p = in.peek();

        // stop if EOF
        if (p == EOF){
            break;
        }

        // stop before whitespace
        if (is_ws(p)){
            break;
        }

        // stop before comment maker
        if(p == '#'){
            break;
        }

        // comsume this non_whitespace byte and append to token.
        token.push_back(static_cast<char>(in.get()));
    }

    // if token is empty, header is malformed

    if (token.empty()){
        if (err) * err = "Failed to read token (empty token)";
        return false;
    }

    return true; // token successfully read
}

//parse int token width good lable-specific errors.

static bool parse_int_token(const std::string& s, int& out, std::string* err, const char* label){
    try
    {
        // conversion
        out = std::stoi(s); // convert ASCII to int
        return true;
    }
    catch(...)
    {
        // FIX: std::("invalid") was illegal; use std::string
        if (err) *err = std::string("Invalid ") + label + " token: '" + s + "'";
        return false;
    }

}

// Clamp a value to [0...1]
static inline double clamp01(double x) {
    if (x < 0.0) return 0.0; // clamp low
    if (x > 1.0) return 1.0; // clamp heigh
    return x; // unchanged
}

// Convert linear [0..1] channedl to int smaple [0..maxval].
static uint16_t float_to_sample(double linear, int maxval, bool clamp, bool gamma2){

    if (gamma2) { //gamma correction request
        if (linear < 0.0) linear = 0.0;
        linear = std::sqrt(linear);
    }

    if (clamp){
        linear = clamp01(linear); // clamp into [0..1]
    }

    double scaled = linear * static_cast<double>(maxval); //scale to sample range
    long rounded = std::lround(scaled); // round to nearest int

    if (rounded < 0 ) rounded = 0; // enforced valid lower bound
    if(rounded > maxval) rounded = maxval; //enforced valid upper bound

    return static_cast<uint16_t>(rounded);
}

// Convert int sample [0...maxval]
static double sample_to_float(uint16_t s, int maxval){
    return static_cast<double>(s) / static_cast<double>(maxval); // normalize
}

// Write one sample in P6:
// - 8/16-bit
static bool write_sample(std::ostream& out, uint16_t s, int maxval, std::string* err){
    if (maxval < 256) {
        unsigned char b = static_cast<unsigned char>(s & 0xFF); // keep low 8 bits
        out.write(reinterpret_cast<const char*>(&b), 1);
    } else{
        unsigned char hi = static_cast<unsigned char>((s >> 8) & 0xFF); //MSB
        unsigned char lo = static_cast<unsigned char>(s & 0xFF); //LSB
        out.write(reinterpret_cast<const char*>(&hi),1);
        out.write(reinterpret_cast<const char*>(&lo), 1); // then LSB
    }

    if (!out){
        if (err) *err= "Filed while writing binary sample bytes.";
        return false;
    }

    return true; // sucess
}
// Read one sample in p6
// - 8 bit/16 bit
static bool read_sample(std::istream& in, uint16_t& s, int maxval, std::string* err){

    if(maxval < 256){
        unsigned char b = 0;
        in.read(reinterpret_cast<char*>(&b), 1);
        if (!in){
            if(err) *err = "Filed while reading 8-bit sample byte.";
            return false;
        }
        s = static_cast<uint16_t>(b);
        return true;
    }

    unsigned char hi = 0;
    unsigned char lo = 0;

    in.read(reinterpret_cast<char*>(&hi), 1);  //read MSB
    in.read(reinterpret_cast<char*>(&lo), 1);

    if(!in){
        if (err) *err = "Filed while reading 16-bit sample bytes.";
        return false;
    }

    s = static_cast<uint16_t>((static_cast<uint16_t>(hi) << 8) | static_cast<uint16_t>(lo)); //combine
    return true;
}

} // namespace detail


//=============================
// Image method definations
//=============================

Image::Image() = default;

Image::Image(int w, int h){resize(w,h);} // alocate image

void Image::resize(int w, int h){

    width_ = w;
    height_ = h;
    pixels_.assign(static_cast<size_t>(w) * static_cast<size_t>(h), Color{}); // allocate pixels
}

int Image::width() const {return width_;}
int Image::height() const {return height_;}

size_t Image::index(int x, int y) const {
    return static_cast<size_t>(y) * static_cast<size_t>(width_) + static_cast<size_t>(x); // row-major
}

Color& Image::at(int x, int y){
    return pixels_.at(index(x, y));  // bounds-checked access
}

const Color& Image::at(int x, int y) const{
    return pixels_.at(index(x,y));
}

void Image::set(int x, int y, const Color& c) {
    at(x,y) = c;    //reuse at()
}

std::vector<Color>& Image::pixels() {return pixels_;} // mutable buffer access
const std::vector<Color>& Image::pixels() const {return pixels_;} //mutable


//==============================
//Public API definitions
//==============================

bool write_p6(const std::string& path, const Image& img, const WriteOptions& opt, std::string* err){
    if (img.width() <= 0 || img.height() <= 0){
        if(err) *err = "Image has non-posative dimenstions.";
        return false;
    }

    // Professional safety check: maxval must be valid per PPM spec.
    if (opt.maxval <= 0 || opt.maxval > 65535){
        if (err) *err = "Invalid maxval (must be 1..65535).";
        return false;
    }

    std::ofstream out(path, std::ios::binary | std::ios::out | std::ios::trunc); // open binary output
    if(!out){
        if (err) *err= "Failed to open output file: " + path;
        return false;
    }

    out << "P6\n";
    out << img.width() << " " << img.height() << "\n"; // width height
    out << opt.maxval << "\n";

    if(!out){
        if (err) *err= "Failed while writing PPM header.";
        return false;
    }

    for (int y = 0; y < img.height(); ++y){ //output rows
        int src_y = opt.flip_y ? (img.height() - 1 - y) : y; // selected row if flipping

        for (int x = 0; x < img.width(); ++x){   // output columns
            const Color& c = img.at(x, src_y);

            uint16_t rs = detail::float_to_sample(c.r, opt.maxval, opt.clamp, opt.gamma2); //R
            uint16_t gs = detail::float_to_sample(c.g, opt.maxval, opt.clamp, opt.gamma2); //G
            uint16_t bs = detail::float_to_sample(c.b, opt.maxval, opt.clamp, opt.gamma2); //B

            if (!detail::write_sample(out, rs, opt.maxval, err)) return false; // Write R
            if (!detail::write_sample(out, gs, opt.maxval, err)) return false; // Write G
            if (!detail::write_sample(out, bs, opt.maxval, err)) return false; // Write B

        }
    }
    return true; //sucess
}

bool read_p6(const std::string& path, Image& img, std::string* err){
    std::ifstream in(path, std::ios::binary | std::ios::in); // open binary input
    if(!in){
        if(err) *err = "Failed to open input file: " + path;
        return false;
    }

    std::string tok;  // token buffer

    if (!detail::next_token(in, tok, err)) return false; //read magic token
    if(tok != "P6"){
        if(err) *err = "Unsupported magic number '" + tok + "' (expected p6).";
        return false;
    }

    if (!detail::next_token(in, tok, err)) return false; //read width token
    int w = 0;
    if(!detail::parse_int_token(tok, w, err, "width")) return false;


    if (!detail::next_token(in, tok, err)) return false; //read height token
    int h = 0;
    if(!detail::parse_int_token(tok, h, err, "height")) return false;


    if (!detail::next_token(in, tok, err)) return false;     // read maxval token
    int maxval = 0;                                          // maxval int
    if (!detail::parse_int_token(tok, maxval, err, "maxval")) return false;

    if (w <= 0 || h <= 0) {                                  // validate dimensions
        if (err) *err = "Invalid image dimensions in header.";
        return false;
    }

    if (maxval <= 0 || maxval > 65535) {                     // validate maxval
        if (err) *err = "Invalid maxval in header (must be 1..65535).";
        return false;
    }

    int ws = in.get();                                       // consume exactly one delimiter byte
    if (ws == EOF) {                                         // missing whitespace?
        if (err) *err = "Unexpected EOF after maxval (missing required whitespace).";
        return false;
    }
    if (!detail::is_ws(ws)) {                                // must be whitespace
        if (err) *err = "Expected whitespace after maxval; found non-whitespace byte.";
        return false;
    }

    img.resize(w, h);                                        // allocate output image

    for (int y = 0; y < img.height(); ++y) {                 // read each row
        for (int x = 0; x < img.width(); ++x) {              // read each pixel
            uint16_t rs = 0, gs = 0, bs = 0;                 // sample buffers

            if (!detail::read_sample(in, rs, maxval, err)) return false; // read R
            if (!detail::read_sample(in, gs, maxval, err)) return false; // read G
            if (!detail::read_sample(in, bs, maxval, err)) return false; // read B

            Color c;                                          // output pixel
            c.r = detail::sample_to_float(rs, maxval);        // normalize R to [0..1]
            c.g = detail::sample_to_float(gs, maxval);        // normalize G to [0..1]
            c.b = detail::sample_to_float(bs, maxval);        // normalize B to [0..1]

            img.set(x, y, c);                                 // store pixel
        }
    }

    return true;                                              // success
}

} // namespace ppm_p6
