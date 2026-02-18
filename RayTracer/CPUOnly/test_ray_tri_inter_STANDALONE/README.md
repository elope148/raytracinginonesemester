## Building and Running the Ray Triangle Intersection Test Module

1. This is a STANDALONE module.

2. DEPENDENCIES:
Please ensure you install Catch2 v3. This library is used for performing tests.

3. Tested on C++20.
  
4. Navigate to the build directory or make a new one:
`mkdir build && cd build`

5. Run the following commands from the project root directory:
```
cmake -S . -B build
cmake --build build
```

3. Run the executable with:
`./build/ray_tri_tests -v high`



