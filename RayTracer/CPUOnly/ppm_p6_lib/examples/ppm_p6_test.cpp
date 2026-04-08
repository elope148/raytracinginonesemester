#include "ppm_p6.hpp"

#include <algorithm>
#include <iostream>
#include <string>

// Make a simple gradient image in linear space.
static ppm_p6::Image make_gradient(int w, int h) {
    ppm_p6::Image img(w, h);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            double u = (w == 1) ? 0.0 : static_cast<double>(x) / static_cast<double>(w - 1);
            double v = (h == 1) ? 0.0 : static_cast<double>(y) / static_cast<double>(h - 1);

            img.set(x, y, ppm_p6::Color{u, v, 0.25});
        }
    }
    return img;
}

// Print a few pixel samples for sanity checking.
static void print_samples(const ppm_p6::Image& img, const std::string& label) {
    auto safe_get = [&](int x, int y) -> ppm_p6::Color {
        x = std::max(0, std::min(x, img.width() - 1));
        y = std::max(0, std::min(y, img.height() - 1));
        return img.at(x, y);
    };

    ppm_p6::Color c00 = safe_get(0, 0);
    ppm_p6::Color cm  = safe_get(img.width() / 2, img.height() / 2);
    ppm_p6::Color c11 = safe_get(img.width() - 1, img.height() - 1);

    std::cout << "\n" << label << ":\n";
    std::cout << "  (0,0):     " << c00.r << ", " << c00.g << ", " << c00.b << "\n";
    std::cout << "  (mid,mid): " << cm.r  << ", " << cm.g  << ", " << cm.b  << "\n";
    std::cout << "  (max,max): " << c11.r << ", " << c11.g << ", " << c11.b << "\n";
}

int main() {
    std::string err;

    // 1) Generate test image (linear floats)
    ppm_p6::Image grad = make_gradient(256, 256);

    // 2) Write P6 8-bit (maxval=255)
    ppm_p6::WriteOptions opt8;
    opt8.maxval = 255;
    opt8.clamp = true;
    opt8.gamma2 = false; // OFF for round-trip test (read-back should match original better)
    opt8.flip_y = false;

    if (!ppm_p6::write_p6("gradient_8bit.ppm", grad, opt8, &err)) {
        std::cerr << "write_p6 failed: " << err << "\n";
        return 1;
    }

    // 3) Write P6 16-bit (maxval=65535)
    ppm_p6::WriteOptions opt16 = opt8;
    opt16.maxval = 65535;

    if (!ppm_p6::write_p6("gradient_16bit.ppm", grad, opt16, &err)) {
        std::cerr << "write_p6 (16-bit) failed: " << err << "\n";
        return 1;
    }

    // 4) Read back 8-bit file
    ppm_p6::Image read_back;
    if (!ppm_p6::read_p6("gradient_8bit.ppm", read_back, &err)) {
        std::cerr << "read_p6 failed: " << err << "\n";
        return 1;
    }

    // 5) Print a few samples
    print_samples(grad, "Original gradient (linear floats)");
    print_samples(read_back, "Read-back gradient (should be close)");

    std::cout << "\nWrote:\n";
    std::cout << "  - gradient_8bit.ppm\n";
    std::cout << "  - gradient_16bit.ppm\n";

    std::cout << "\nViewer tip:\n";
    std::cout << "  Some viewers only support P3. If P6 fails, convert using ImageMagick:\n";
    std::cout << "    magick gradient_8bit.ppm gradient.png\n";

    return 0;
}
