Here is your content reformatted to match the style and structure of the example README.

# ppm_p6_lib

A small C++17 library for reading/writing binary PPM files (P6), standard library only.
Designed for ray tracers: pixels stored as linear RGB doubles in [0..1].

## Project Structure

```text
ppm_p6_lib/
├── CMakeLists.txt
├── include/
│   └── ppm_p6.hpp
├── src/
│   └── ppm_p6.cpp
└── examples/
    └── ppm_p6_test.cpp

```

## Prerequisites

* **CMake**
* **C++ Compiler** (MSVC/Clang/GCC) with C++17 support

## Building

### CMake (Recommended)

**Windows (Visual Studio / MSVC)**
Open "Developer Command Prompt for VS" or use VS Code terminal if MSVC is configured.

1. Configure and build:
```bash
cmake -S . -B build
cmake --build build --config Debug

```


2. Run the test:
```bash
.\build\Debug\ppm_p6_test.exe

# Or with arguments:
.\build\Debug\ppm_p6_test.exe input.ppm output.ppm

```


3. Release build (Optional):
```bash
cmake --build build --config Release
.\build\Release\ppm_p6_test.exe

```



**macOS / Linux (Clang / GCC)**

1. Configure and build:
```bash
cmake -S . -B build
cmake --build build -j

```


2. Run the test:
```bash
./build/ppm_p6_test

# Or with arguments:
./build/ppm_p6_test input.ppm output.ppm

```



### Manual Builds (No CMake)

**Windows (MSVC cl + lib)**

```cmd
cl /std:c++17 /EHsc /Iinclude /c src\ppm_p6.cpp
lib /OUT:ppm_p6.lib ppm_p6.obj
cl /std:c++17 /EHsc /Iinclude examples\ppm_p6_test.cpp ppm_p6.lib /Fe:ppm_p6_test.exe
ppm_p6_test.exe
ppm_p6_test.exe input.ppm output.ppm

```

**macOS/Linux (clang++/g++ + ar)**

```bash
clang++ -std=c++17 -O2 -Iinclude -c src/ppm_p6.cpp -o ppm_p6.o
ar rcs libppm_p6.a ppm_p6.o
clang++ -std=c++17 -O2 -Iinclude examples/ppm_p6_test.cpp libppm_p6.a -o ppm_p6_test
./ppm_p6_test
./ppm_p6_test input.ppm output.ppm

```

## Integration

To use this library in another CMake project:

**Option 1:** Copy `ppm_p6_lib` into your repo and add it as a subdirectory.

In your project's `CMakeLists.txt`:

```cmake
add_subdirectory(path/to/ppm_p6_lib)
target_link_libraries(your_app PRIVATE ppm_p6)
target_include_directories(your_app PRIVATE path/to/ppm_p6_lib/include)

```

Then in C++:

```cpp
#include "ppm_p6.hpp" 
// use ppm_p6::Image, ppm_p6::write_p6, ppm_p6::read_p6

```

## Implementation Details & Notes

* **P6 is binary:** Always open files with `ios::binary` on Windows or the raster can be corrupted.
* **Viewer Compatibility:** Some viewers only support P3 (ASCII). If your viewer fails, convert with another tool/viewer.
* **PowerShell:** Requires `.\` to run executables (e.g., `.\build\Debug\ppm_p6_test.exe`).
* **Git Ignore:**
* **YES:** Upload `CMakeLists.txt`, `include/`, `src/`, `examples/`.
* **NO:** Do NOT upload `build/` folder or artifacts (`*.obj`, `*.lib`, `*.exe`, `*.pdb`, `*.dSYM/`).



**Recommended .gitignore:**

```text
build/
.vscode/
*.obj
*.o
*.a
*.lib
*.exe
*.pdb
*.ilk
*.pch
*.idb
*.dSYM/
.DS_Store

```

---

Would you like me to generate the `.gitignore` file content for you as well?
