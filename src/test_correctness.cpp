#include "linear_algebra.h"
#include <iostream>
#include <cmath>
#include <cstring>

static const double TOLERANCE = 1e-9;

static bool check_close(const double* a, const double* b, int n, const char* test_name) {
    for (int i = 0; i < n; ++i) {
        if (std::fabs(a[i] - b[i]) > TOLERANCE) {
            std::cerr << "FAIL: " << test_name << " — mismatch at index " << i
                      << ": got " << a[i] << ", expected " << b[i] << "\n";
            return false;
        }
    }
    std::cout << "PASS: " << test_name << "\n";
    return true;
}

// --- Test: Matrix-Vector Row-Major ---
static bool test_mv_row_major() {
    // 2x3 matrix (row-major): [[1,2,3],[4,5,6]]
    // vector: [1,1,1]
    // expected: [6, 15]
    double matrix[] = {1, 2, 3, 4, 5, 6};
    double vec[] = {1, 1, 1};
    double result[2];
    double expected[] = {6.0, 15.0};

    multiply_mv_row_major(matrix, 2, 3, vec, result);
    return check_close(result, expected, 2, "multiply_mv_row_major (basic)");
}

// --- Test: Matrix-Vector Column-Major ---
static bool test_mv_col_major() {
    // Same 2x3 matrix, but stored column-major:
    // Col 0: [1,4], Col 1: [2,5], Col 2: [3,6]
    // Storage: [1,4, 2,5, 3,6]
    double matrix[] = {1, 4, 2, 5, 3, 6};
    double vec[] = {1, 1, 1};
    double result[2];
    double expected[] = {6.0, 15.0};

    multiply_mv_col_major(matrix, 2, 3, vec, result);
    return check_close(result, expected, 2, "multiply_mv_col_major (basic)");
}

// --- Test: Matrix-Matrix Naive ---
static bool test_mm_naive() {
    // A = 2x3: [[1,2,3],[4,5,6]]
    // B = 3x2: [[7,8],[9,10],[11,12]]
    // C = A*B = [[58,64],[139,154]]
    double A[] = {1, 2, 3, 4, 5, 6};
    double B[] = {7, 8, 9, 10, 11, 12};
    double result[4];
    double expected[] = {58, 64, 139, 154};

    multiply_mm_naive(A, 2, 3, B, 3, 2, result);
    return check_close(result, expected, 4, "multiply_mm_naive (basic)");
}

// --- Test: Matrix-Matrix Transposed B ---
static bool test_mm_transposed_b() {
    // A = 2x3: [[1,2,3],[4,5,6]]
    // B = 3x2: [[7,8],[9,10],[11,12]]
    // B^T = 2x3: [[7,9,11],[8,10,12]]
    // C = A*B = [[58,64],[139,154]]
    double A[] = {1, 2, 3, 4, 5, 6};
    double BT[] = {7, 9, 11, 8, 10, 12};
    double result[4];
    double expected[] = {58, 64, 139, 154};

    multiply_mm_transposed_b(A, 2, 3, BT, 3, 2, result);
    return check_close(result, expected, 4, "multiply_mm_transposed_b (basic)");
}

// --- Test: Blocked Matrix-Matrix ---
static bool test_mm_blocked() {
    double A[] = {1, 2, 3, 4, 5, 6};
    double B[] = {7, 8, 9, 10, 11, 12};
    double result[4];
    double expected[] = {58, 64, 139, 154};

    multiply_mm_blocked(A, 2, 3, B, 3, 2, result, 2);
    return check_close(result, expected, 4, "multiply_mm_blocked (basic)");
}

// --- Test: Larger random matrix (cross-verify naive vs transposed_b vs blocked) ---
static bool test_cross_verify() {
    const int N = 64;
    double* A = new double[N * N];
    double* B = new double[N * N];
    double* BT = new double[N * N];
    double* C_naive = new double[N * N];
    double* C_trans = new double[N * N];
    double* C_block = new double[N * N];

    // Fill with deterministic values
    for (int i = 0; i < N * N; ++i) {
        A[i] = static_cast<double>(i % 17) - 8.0;
        B[i] = static_cast<double>(i % 13) - 6.0;
    }

    // Compute B^T
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            BT[j * N + i] = B[i * N + j];

    multiply_mm_naive(A, N, N, B, N, N, C_naive);
    multiply_mm_transposed_b(A, N, N, BT, N, N, C_trans);
    multiply_mm_blocked(A, N, N, B, N, N, C_block, 16);

    bool ok = true;
    ok &= check_close(C_naive, C_trans, N * N, "cross-verify: naive vs transposed_b (64x64)");
    ok &= check_close(C_naive, C_block, N * N, "cross-verify: naive vs blocked (64x64)");

    delete[] A; delete[] B; delete[] BT;
    delete[] C_naive; delete[] C_trans; delete[] C_block;
    return ok;
}

int main() {
    int passed = 0, total = 0;
    auto run = [&](bool result) { total++; if (result) passed++; };

    run(test_mv_row_major());
    run(test_mv_col_major());
    run(test_mm_naive());
    run(test_mm_transposed_b());
    run(test_mm_blocked());
    run(test_cross_verify());

    std::cout << "\n" << passed << "/" << total << " tests passed.\n";
    return (passed == total) ? 0 : 1;
}
