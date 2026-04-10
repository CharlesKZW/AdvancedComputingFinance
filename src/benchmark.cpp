#include "linear_algebra.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>
#include <functional>

// ============================================================================
// Benchmarking Framework
// ============================================================================

struct BenchResult {
    std::string name;
    int size;
    double avg_ms;
    double std_ms;
    double gflops;
};

// Run a function multiple times and collect timing statistics
static BenchResult benchmark(const std::string& name, int size, double flops,
                             int num_runs, std::function<void()> fn) {
    // Warmup
    fn();

    std::vector<double> times(num_runs);
    for (int r = 0; r < num_runs; ++r) {
        auto start = std::chrono::high_resolution_clock::now();
        fn();
        auto end = std::chrono::high_resolution_clock::now();
        times[r] = std::chrono::duration<double, std::milli>(end - start).count();
    }

    double sum = 0.0;
    for (double t : times) sum += t;
    double avg = sum / num_runs;

    double var = 0.0;
    for (double t : times) var += (t - avg) * (t - avg);
    double stddev = std::sqrt(var / num_runs);

    double gflops = (flops / (avg * 1e-3)) / 1e9;

    return {name, size, avg, stddev, gflops};
}

static void print_header() {
    std::cout << std::left
              << std::setw(42) << "Kernel"
              << std::setw(8)  << "N"
              << std::setw(14) << "Avg (ms)"
              << std::setw(14) << "Std (ms)"
              << std::setw(12) << "GFLOP/s"
              << "\n";
    std::cout << std::string(90, '-') << "\n";
}

static void print_result(const BenchResult& r) {
    std::cout << std::left
              << std::setw(42) << r.name
              << std::setw(8)  << r.size
              << std::fixed << std::setprecision(3)
              << std::setw(14) << r.avg_ms
              << std::setw(14) << r.std_ms
              << std::setw(12) << r.gflops
              << "\n";
}

// ============================================================================
// Helper: fill arrays with deterministic data
// ============================================================================

static void fill_array(double* arr, size_t n) {
    for (size_t i = 0; i < n; ++i)
        arr[i] = static_cast<double>(i % 100) * 0.01;
}

// Transpose a row-major NxM matrix into MxN row-major
static void transpose(const double* src, double* dst, int rows, int cols) {
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            dst[j * rows + i] = src[i * cols + j];
}

// ============================================================================
// Benchmark Suites
// ============================================================================

static void bench_matrix_vector(const std::vector<int>& sizes, int num_runs) {
    std::cout << "\n=== Matrix-Vector Multiplication ===\n\n";
    print_header();

    for (int N : sizes) {
        // Row-major matrix
        double* M_row = new double[static_cast<size_t>(N) * N];
        // Column-major matrix (same logical matrix, different layout)
        double* M_col = new double[static_cast<size_t>(N) * N];
        double* vec   = new double[N];
        double* res   = new double[N];

        fill_array(M_row, static_cast<size_t>(N) * N);
        fill_array(vec, N);

        // Convert row-major to column-major
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                M_col[j * N + i] = M_row[i * N + j];

        double flops = 2.0 * N * N; // N rows, each has N mul + N add

        auto r1 = benchmark("MV Row-Major", N, flops, num_runs, [&]() {
            multiply_mv_row_major(M_row, N, N, vec, res);
        });
        print_result(r1);

        auto r2 = benchmark("MV Col-Major", N, flops, num_runs, [&]() {
            multiply_mv_col_major(M_col, N, N, vec, res);
        });
        print_result(r2);

        delete[] M_row; delete[] M_col; delete[] vec; delete[] res;
    }
}

static void bench_matrix_matrix(const std::vector<int>& sizes, int num_runs) {
    std::cout << "\n=== Matrix-Matrix Multiplication ===\n\n";
    print_header();

    for (int N : sizes) {
        size_t n2 = static_cast<size_t>(N) * N;
        double* A   = new double[n2];
        double* B   = new double[n2];
        double* BT  = new double[n2];
        double* C   = new double[n2];

        fill_array(A, n2);
        fill_array(B, n2);
        transpose(B, BT, N, N);

        double flops = 2.0 * N * N * N; // N^3 mul + N^3 add

        auto r1 = benchmark("MM Naive (ijk)", N, flops, num_runs, [&]() {
            multiply_mm_naive(A, N, N, B, N, N, C);
        });
        print_result(r1);

        auto r2 = benchmark("MM Transposed-B", N, flops, num_runs, [&]() {
            multiply_mm_transposed_b(A, N, N, BT, N, N, C);
        });
        print_result(r2);

        auto r3 = benchmark("MM Blocked (bs=32)", N, flops, num_runs, [&]() {
            multiply_mm_blocked(A, N, N, B, N, N, C, 32);
        });
        print_result(r3);

        auto r4 = benchmark("MM Blocked (bs=64)", N, flops, num_runs, [&]() {
            multiply_mm_blocked(A, N, N, B, N, N, C, 64);
        });
        print_result(r4);

        delete[] A; delete[] B; delete[] BT; delete[] C;
    }
}

static void bench_aligned_vs_unaligned(const std::vector<int>& sizes, int num_runs) {
    std::cout << "\n=== Aligned vs Unaligned Memory ===\n\n";
    print_header();

    for (int N : sizes) {
        size_t n2 = static_cast<size_t>(N) * N;
        double flops = 2.0 * N * N * N;

        // Unaligned (standard new)
        {
            double* A = new double[n2];
            double* B = new double[n2];
            double* C = new double[n2];
            fill_array(A, n2);
            fill_array(B, n2);

            auto r = benchmark("MM Naive (unaligned, new)", N, flops, num_runs, [&]() {
                multiply_mm_naive(A, N, N, B, N, N, C);
            });
            print_result(r);
            delete[] A; delete[] B; delete[] C;
        }

        // Aligned (64-byte boundary)
        {
            double* A = aligned_alloc_doubles(n2, 64);
            double* B = aligned_alloc_doubles(n2, 64);
            double* C = aligned_alloc_doubles(n2, 64);
            fill_array(A, n2);
            fill_array(B, n2);

            auto r = benchmark("MM Naive (aligned 64B)", N, flops, num_runs, [&]() {
                multiply_mm_naive(A, N, N, B, N, N, C);
            });
            print_result(r);
            aligned_free(A); aligned_free(B); aligned_free(C);
        }

        // Aligned + Blocked
        {
            double* A = aligned_alloc_doubles(n2, 64);
            double* B = aligned_alloc_doubles(n2, 64);
            double* C = aligned_alloc_doubles(n2, 64);
            fill_array(A, n2);
            fill_array(B, n2);

            auto r = benchmark("MM Blocked (aligned 64B, bs=64)", N, flops, num_runs, [&]() {
                multiply_mm_blocked(A, N, N, B, N, N, C, 64);
            });
            print_result(r);
            aligned_free(A); aligned_free(B); aligned_free(C);
        }
    }
}

static void bench_inlining(const std::vector<int>& sizes, int num_runs) {
    std::cout << "\n=== Inlining Experiment: dot_product vs dot_product_inline ===\n\n";
    print_header();

    for (int N : sizes) {
        double* A = new double[N];
        double* B = new double[N];
        fill_array(A, N);
        fill_array(B, N);

        double flops = 2.0 * N;
        int inner_runs = (N < 1000) ? 100000 : 1000;

        volatile double sink = 0.0;

        auto r1 = benchmark("dot_product (non-inline)", N, flops * inner_runs, num_runs, [&]() {
            for (int r = 0; r < inner_runs; ++r)
                sink = dot_product(A, B, N);
        });
        print_result(r1);

        auto r2 = benchmark("dot_product_inline", N, flops * inner_runs, num_runs, [&]() {
            for (int r = 0; r < inner_runs; ++r)
                sink = dot_product_inline(A, B, N);
        });
        print_result(r2);

        delete[] A; delete[] B;
    }
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "===============================================================\n";
    std::cout << " High-Performance Linear Algebra Kernels — Benchmark Suite\n";
    std::cout << "===============================================================\n";

    const int NUM_RUNS = 10;

    std::vector<int> mv_sizes = {128, 256, 512, 1024, 2048};
    bench_matrix_vector(mv_sizes, NUM_RUNS);

    std::vector<int> mm_sizes = {64, 128, 256, 512, 1024};
    bench_matrix_matrix(mm_sizes, NUM_RUNS);

    std::vector<int> align_sizes = {128, 256, 512, 1024};
    bench_aligned_vs_unaligned(align_sizes, NUM_RUNS);

    std::vector<int> inline_sizes = {64, 256, 1024, 4096};
    bench_inlining(inline_sizes, NUM_RUNS);

    std::cout << "\nBenchmark complete.\n";
    return 0;
}
