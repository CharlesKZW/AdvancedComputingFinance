CXX = g++
CXXFLAGS_COMMON = -std=c++17 -Wall -Wextra -Iinclude

KERNEL_SRCS = src/multiply_mv_row_major.cpp \
              src/multiply_mv_col_major.cpp \
              src/multiply_mm_naive.cpp \
              src/multiply_mm_transposed_b.cpp \
              src/multiply_mm_blocked.cpp \
              src/dot_product.cpp

.PHONY: all clean test

all: build/test_correctness build/benchmark_O0 build/benchmark_O3 build/benchmark_profile

build:
	mkdir -p build

# Correctness tests (O2)
build/test_correctness: src/test_correctness.cpp $(KERNEL_SRCS) | build
	$(CXX) $(CXXFLAGS_COMMON) -O2 -o $@ src/test_correctness.cpp $(KERNEL_SRCS)

# Benchmark with no optimization (for inlining experiments)
build/benchmark_O0: src/benchmark.cpp $(KERNEL_SRCS) | build
	$(CXX) $(CXXFLAGS_COMMON) -O0 -o $@ src/benchmark.cpp $(KERNEL_SRCS)

# Benchmark with aggressive optimization
build/benchmark_O3: src/benchmark.cpp $(KERNEL_SRCS) | build
	$(CXX) $(CXXFLAGS_COMMON) -O3 -march=native -o $@ src/benchmark.cpp $(KERNEL_SRCS)

# Benchmark with debug symbols for profiling (Instruments)
build/benchmark_profile: src/benchmark.cpp $(KERNEL_SRCS) | build
	$(CXX) $(CXXFLAGS_COMMON) -O2 -g -o $@ src/benchmark.cpp $(KERNEL_SRCS)

test: build/test_correctness
	./build/test_correctness

clean:
	rm -rf build
