cmake_minimum_required(VERSION 3.14)
project(HighPerfLinAlg LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_SOURCE_DIR}/include)

# Common source files (the kernels)
set(KERNEL_SOURCES
    src/multiply_mv_row_major.cpp
    src/multiply_mv_col_major.cpp
    src/multiply_mm_naive.cpp
    src/multiply_mm_transposed_b.cpp
    src/multiply_mm_blocked.cpp
    src/dot_product.cpp
)

# === Release build (default): -O3 optimizations ===
add_executable(test_correctness src/test_correctness.cpp ${KERNEL_SOURCES})
add_executable(benchmark src/benchmark.cpp ${KERNEL_SOURCES})

# === Separate O0 build for inlining experiments ===
add_executable(benchmark_O0 src/benchmark.cpp ${KERNEL_SOURCES})
target_compile_options(benchmark_O0 PRIVATE -O0)

# === Separate O3 build with explicit flags ===
add_executable(benchmark_O3 src/benchmark.cpp ${KERNEL_SOURCES})
target_compile_options(benchmark_O3 PRIVATE -O3 -march=native)

# === Build with debug symbols for profiling (Instruments on macOS) ===
add_executable(benchmark_profile src/benchmark.cpp ${KERNEL_SOURCES})
target_compile_options(benchmark_profile PRIVATE -O2 -g)

# Default to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
