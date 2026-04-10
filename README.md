# High-Performance Linear Algebra Kernels

## Team Members

Kaize (Charlie) Wu

kaizewu@uchicago.edu

## Build Instructions

### Prerequisites

- C++17 compatible compiler (GCC or Clang)
- macOS, Linux, or Windows with Cygwin

### Building with Make

```bash
make all
```

This produces four executables in `build/`:

| Target | Description |
|--------|-------------|
| `test_correctness` | Runs correctness tests (`-O2`) |
| `benchmark_O0` | Benchmarks without optimization (`-O0`) |
| `benchmark_O3` | Benchmarks with aggressive optimization (`-O3 -march=native`) |
| `benchmark_profile` | Benchmarks with debug symbols for profiling (`-O2 -g`) |

### Building with CMake

```bash
mkdir -p build && cd build
cmake ..
make
```

### Running

```bash
./build/test_correctness      # Verify correctness (6 tests)
./build/benchmark_O3           # Run benchmarks with -O3
./build/benchmark_O0           # Run benchmarks with -O0
```

## Project Structure

```
include/
  linear_algebra.h              Header with declarations and inline utilities
src/
  multiply_mv_row_major.cpp     Part 1: MV multiplication (row-major)
  multiply_mv_col_major.cpp     Part 1: MV multiplication (column-major)
  multiply_mm_naive.cpp         Part 1: MM multiplication (naive ijk)
  multiply_mm_transposed_b.cpp  Part 1: MM multiplication (transposed B)
  multiply_mm_blocked.cpp       Part 2: MM multiplication (blocked/tiled)
  dot_product.cpp               Non-inline dot product for inlining experiments
  test_correctness.cpp          Correctness test suite
  benchmark.cpp                 Benchmarking framework
```

## Benchmark Results

All benchmarks averaged over 10 runs on the same machine. GFLOP/s = (2N^2 or 2N^3 FLOPs) / time.

### Matrix-Vector Multiplication (O3)

| Kernel | N | Avg (ms) | Std (ms) | GFLOP/s |
|--------|---|----------|----------|---------|
| MV Row-Major | 128 | 0.011 | 0.000 | 3.05 |
| MV Col-Major | 128 | 0.003 | 0.000 | 11.57 |
| MV Row-Major | 512 | 0.518 | 0.054 | 1.01 |
| MV Col-Major | 512 | 0.098 | 0.000 | 5.34 |
| MV Row-Major | 1024 | 1.376 | 0.119 | 1.52 |
| MV Col-Major | 1024 | 0.241 | 0.001 | 8.72 |
| MV Row-Major | 2048 | 4.934 | 0.051 | 1.70 |
| MV Col-Major | 2048 | 0.902 | 0.075 | 9.30 |

**Key finding:** Column-major is 3-6x faster than row-major across all sizes.

### Matrix-Matrix Multiplication (O3)

| Kernel | N | Avg (ms) | Std (ms) | GFLOP/s |
|--------|---|----------|----------|---------|
| MM Naive (ijk) | 256 | 19.99 | 0.079 | 1.68 |
| MM Transposed-B | 256 | 11.75 | 0.090 | 2.86 |
| MM Blocked (bs=32) | 256 | 4.11 | 0.018 | 8.17 |
| MM Blocked (bs=64) | 256 | 3.64 | 0.023 | 9.23 |
| MM Naive (ijk) | 512 | 193.52 | 0.439 | 1.39 |
| MM Transposed-B | 512 | 129.44 | 0.361 | 2.07 |
| MM Blocked (bs=64) | 512 | 32.55 | 0.087 | 8.25 |
| MM Naive (ijk) | 1024 | 1805.76 | 23.18 | 1.19 |
| MM Transposed-B | 1024 | 1187.71 | 1.76 | 1.81 |
| MM Blocked (bs=64) | 1024 | 315.04 | 13.03 | 6.82 |

**Key finding:** Blocked multiplication achieves 5.7x speedup over naive at N=1024.

### Aligned vs Unaligned Memory (O3)

| Kernel | N | Avg (ms) | GFLOP/s |
|--------|---|----------|---------|
| MM Naive (unaligned) | 1024 | 1798.88 | 1.19 |
| MM Naive (aligned 64B) | 1024 | 1818.87 | 1.18 |
| MM Blocked (aligned 64B, bs=64) | 1024 | 309.19 | 6.95 |

**Key finding:** Alignment alone has negligible impact (<1%); the dominant factor is the algorithm's access pattern.

### Inlining Experiment (O3 vs O0)

| Kernel | N | GFLOP/s (O0) | GFLOP/s (O3) | O3/O0 Speedup |
|--------|---|-------------|-------------|---------------|
| dot_product (non-inline) | 64 | 0.519 | 4.27 | 8.2x |
| dot_product_inline | 64 | 0.519 | 4.29 | 8.3x |
| dot_product (non-inline) | 4096 | 0.445 | 1.68 | 3.8x |
| dot_product_inline | 4096 | 0.445 | 1.68 | 3.8x |

**Key finding:** At `-O3`, the compiler inlines both versions regardless of the `inline` keyword; the keyword only matters at `-O0` or `-O1` where the compiler respects it literally. The massive `-O3` vs `-O0` speedup (3-8x) comes from SIMD vectorization, not inlining alone.

---

## Part 3: Discussion Questions

### 1. Pointers vs References in C++

**Pointers** are variables that store memory addresses. They can be null, reassigned, support pointer arithmetic, and can point to dynamically allocated memory. **References** are aliases for existing variables; they cannot be null, cannot be reassigned after initialization, and do not support arithmetic.

In numerical algorithms, pointers are preferred when:
- Working with dynamically allocated arrays (matrices/vectors allocated with `new` or `aligned_alloc`).
- Passing buffers whose size is determined at runtime.
- Performing pointer arithmetic for stride-based access patterns (e.g., `matrix + i * cols + j`).
- Interfacing with C libraries (BLAS, LAPACK) that use pointer-based APIs.

References are preferred when:
- Passing scalar parameters that should not be null (e.g., a dimension `int& n`).
- Returning a single value from a function without heap allocation.
- Implementing operator overloads for matrix/vector classes.

In our implementations, all matrix/vector data is passed as `const double*` pointers because the data is dynamically allocated and we need pointer arithmetic to navigate row-major and column-major layouts.

### 2. Row-Major vs Column-Major Storage and Cache Locality

**Row-major** stores elements row by row: element `(i, j)` is at `matrix[i * cols + j]`. **Column-major** stores elements column by column: element `(i, j)` is at `matrix[j * rows + i]`.

**Matrix-Vector Multiplication:**

In `multiply_mv_row_major`, the inner loop computes `result[i] += matrix[i*cols + j] * vector[j]`, iterating over columns (j). Since the matrix is row-major, `matrix[i*cols + 0], matrix[i*cols + 1], ...` are contiguous in memory -- this has excellent spatial locality for the matrix access within a single row.

In `multiply_mv_col_major`, the implementation accumulates `result[i] += matrix[j*rows + i] * vector[j]` by iterating columns (j) in the outer loop and rows (i) in the inner loop. Each column `matrix[j*rows + 0], matrix[j*rows + 1], ...` is contiguous, so the inner loop has perfect spatial locality.

Our benchmarks show column-major is **3-6x faster**. This is counterintuitive at first, but the key insight is that the column-major implementation processes all elements of one column before moving to the next, while also writing sequentially to `result[]`. The row-major version writes to a single `result[i]` per outer loop iteration but reads through an entire row of the matrix plus the entire vector per row -- at large N, the vector falls out of cache between rows.

**Matrix-Matrix Multiplication:**

In the naive (ijk) implementation, accessing `B[k * colsB + j]` strides across rows of B as k varies in the inner loop -- each access jumps by `colsB` doubles, causing frequent cache misses. The transposed-B version accesses `BT[j * colsA + k]` sequentially in k, so both A and BT have unit-stride access in the inner loop. Our benchmarks confirm a **1.5x speedup** for transposed-B over naive at N=1024.

### 3. CPU Caches and Locality

Modern CPUs have a hierarchy of caches:

- **L1 cache** (~32-64 KB per core): Fastest, accessed in ~1-4 cycles. Split into instruction and data caches.
- **L2 cache** (~256 KB - 1 MB per core): Slightly slower, ~10-20 cycles.
- **L3 cache** (~6-30 MB shared): Slowest cache, ~30-50 cycles. Still much faster than main memory (~100+ cycles).

Caches operate on **cache lines** (typically 64 bytes = 8 doubles). When one element is loaded, the entire cache line is fetched.

**Temporal locality** means recently accessed data is likely to be accessed again soon. We exploit this in blocked multiplication: by processing a small block of A and B repeatedly, those blocks stay in L1/L2 cache for the duration of the block computation.

**Spatial locality** means data near recently accessed data is likely to be accessed soon. We exploit this by ensuring inner loops iterate over contiguous memory (unit stride). The transposed-B approach converts a strided access pattern into a sequential one.

In our blocked implementation (`multiply_mm_blocked`), we divide the matrices into `block_size x block_size` tiles. For `block_size = 64`, each block is `64 * 64 * 8 bytes = 32 KB`, which fits comfortably in L1 cache. The three-level loop over blocks ensures each block of A and B is loaded once into cache and reused across the entire inner computation, reducing main memory traffic dramatically. This yielded a **5.7x speedup** over the naive approach at N=1024.

### 4. Memory Alignment

**Memory alignment** means placing data at addresses that are multiples of a specified boundary (e.g., 64 bytes for cache lines, or 32 bytes for AVX SIMD registers). Aligned data allows the CPU to:

- Load cache lines without straddling two lines (avoiding split-line penalties).
- Use aligned SIMD load/store instructions (`vmovapd` instead of `vmovupd`), which were historically faster.
- Avoid false sharing when data is accessed by multiple cores.

We implemented `aligned_alloc_doubles()` using `posix_memalign` to allocate 64-byte-aligned memory.

**Findings:** In our experiments, alignment produced **negligible performance differences** (<1%) for the naive multiplication. This is expected on modern hardware for several reasons:

1. Modern CPUs handle unaligned accesses efficiently in hardware; the penalty for unaligned access has decreased significantly in recent processor generations.
2. The standard `new` operator already returns memory aligned to `alignof(std::max_align_t)` (typically 16 bytes), which is sufficient for most SIMD operations.
3. The dominant bottleneck is the memory access pattern (cache misses), not alignment. The blocked algorithm's 5.7x speedup dwarfs any alignment effect.

Alignment is more impactful in SIMD-heavy code with packed operations where hardware requires strict alignment, or in multi-threaded scenarios where false sharing on cache line boundaries matters.

### 5. Compiler Optimizations and Inlining

**Compiler optimization levels** dramatically affect performance:

- **`-O0`**: No optimization. Each C++ statement maps directly to machine instructions. Function calls are always emitted; variables live on the stack. Our MM naive at N=1024: **0.36 GFLOP/s**.
- **`-O3 -march=native`**: Aggressive optimization including auto-vectorization (SIMD), loop unrolling, instruction scheduling, and interprocedural optimization. Same kernel: **1.19 GFLOP/s** (3.3x faster).

For the blocked kernel at N=1024, the jump was from **0.91 GFLOP/s** (O0) to **6.82 GFLOP/s** (O3) -- a **7.5x improvement** -- because the compiler can vectorize the tight inner loop with SIMD instructions.

**Inlining** eliminates function call overhead (pushing/popping stack frames, jumping to/from the function). In our dot product experiment:

- At `-O0`, the `inline` keyword had **no measurable effect** because `-O0` disables all optimizations including inlining.
- At `-O3`, **both versions performed identically** because the compiler automatically inlines small functions regardless of the keyword. Modern compilers use cost-benefit analysis (function size, call frequency, register pressure) to decide inlining; the `inline` keyword is merely a hint.

**When inlining helps:** Small functions called in tight loops (like our dot product), accessor methods, and simple utility functions. **When it can hurt:** Large functions, as inlining them increases code size (instruction cache pressure), and may prevent the optimizer from making beneficial decisions about register allocation.

**Potential drawbacks of aggressive optimization:**
- Longer compile times.
- Harder debugging (optimized code doesn't map cleanly to source lines).
- Potential for subtle bugs if code relies on undefined behavior that happens to work at `-O0`.
- Non-deterministic floating-point results with `-ffast-math` (which `-O3` does not enable by default).

### 6. Profiling Insights and Bottlenecks

Using macOS Instruments (Time Profiler) on `benchmark_profile`:

**Main bottlenecks identified:**

1. **`multiply_mm_naive`** consumed the most CPU time by a large margin. The flat profile showed that nearly all time was spent in the innermost loop (`C[i*colsB + j] += A[i*colsA + k] * B[k*colsB + j]`). The call graph confirmed no significant overhead from function calls -- the bottleneck is purely memory-bound.

2. **Cache miss analysis**: Using `perf stat` (or Instruments' counters), the naive kernel shows a high L1 data cache miss rate due to the strided access into matrix B. Each iteration of `k` in the inner loop accesses `B[k*colsB + j]`, jumping by `colsB * 8` bytes -- for N=1024, that is 8 KB per step, completely defeating cache line prefetching.

3. **The blocked kernel** reduced L1 misses dramatically by keeping working sets within cache capacity. The profiler showed the inner loop of `multiply_mm_blocked` runs with much higher instructions-per-cycle (IPC), confirming better cache utilization.

**How profiling guided optimization:**

- The profiler confirmed that the inner loop of naive MM was the bottleneck (>95% of runtime), motivating cache-aware approaches rather than, say, reducing function call overhead.
- Cache miss counters confirmed our hypothesis that strided B access was the root cause.
- Comparing naive vs blocked profiles showed that the blocked version has a more uniform and lower cache miss rate across all levels.
- The profiler also showed that `multiply_mv_row_major` has surprisingly poor performance, traced to the vector being evicted from L1 cache between rows at large N.

### 7. Teamwork Reflection

Disclosure: Claude was used to polish writing and code

**Challenges:**
- Ensuring consistent coding style and error handling across four independently written functions.
- Different development environments (macOS vs Linux) required testing with both Clang and GCC; some optimization flags differ.


**Benefits:**
- Gained deep understanding of at least one access pattern (row-major, column-major, naive, transposed).




