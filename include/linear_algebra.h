#ifndef LINEAR_ALGEBRA_H
#define LINEAR_ALGEBRA_H

#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <new>

// ============================================================================
// Part 1: Baseline Implementations
// ============================================================================

// Team Member 1: Matrix-Vector Multiplication (Row-Major)
void multiply_mv_row_major(const double* matrix, int rows, int cols,
                           const double* vector, double* result);

// Team Member 2: Matrix-Vector Multiplication (Column-Major)
void multiply_mv_col_major(const double* matrix, int rows, int cols,
                           const double* vector, double* result);

// Team Member 3: Matrix-Matrix Multiplication (Naive)
void multiply_mm_naive(const double* matrixA, int rowsA, int colsA,
                       const double* matrixB, int rowsB, int colsB,
                       double* result);

// Team Member 4: Matrix-Matrix Multiplication (Transposed B)
void multiply_mm_transposed_b(const double* matrixA, int rowsA, int colsA,
                              const double* matrixB_transposed, int rowsB, int colsB,
                              double* result);

// ============================================================================
// Part 2: Optimized Implementations
// ============================================================================

// Blocked/Tiled Matrix-Matrix Multiplication
void multiply_mm_blocked(const double* matrixA, int rowsA, int colsA,
                         const double* matrixB, int rowsB, int colsB,
                         double* result, int block_size = 64);

// ============================================================================
// Aligned Memory Allocation Utilities
// ============================================================================

// Allocate memory aligned to a specified boundary (default 64 bytes for cache line)
inline double* aligned_alloc_doubles(size_t count, size_t alignment = 64) {
    void* ptr = nullptr;
#if defined(_MSC_VER)
    ptr = _aligned_malloc(count * sizeof(double), alignment);
    if (!ptr) throw std::bad_alloc();
#else
    if (posix_memalign(&ptr, alignment, count * sizeof(double)) != 0)
        throw std::bad_alloc();
#endif
    return static_cast<double*>(ptr);
}

inline void aligned_free(double* ptr) {
#if defined(_MSC_VER)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

// ============================================================================
// Inline helper functions (for inlining experiments)
// ============================================================================

// Non-inline version of dot product
double dot_product(const double* a, const double* b, int n);

// Inline version of dot product
inline double dot_product_inline(const double* a, const double* b, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

#endif // LINEAR_ALGEBRA_H
