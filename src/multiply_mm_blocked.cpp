#include "linear_algebra.h"
#include <cstring>
#include <stdexcept>
#include <algorithm>

// Optimized: Blocked/Tiled Matrix-Matrix Multiplication
// Divides the computation into blocks that fit into L1/L2 cache
// C[i][j] += A[i][k] * B[k][j], processed in block_size x block_size tiles
void multiply_mm_blocked(const double* matrixA, int rowsA, int colsA,
                         const double* matrixB, int rowsB, int colsB,
                         double* result, int block_size) {
    if (!matrixA || !matrixB || !result)
        throw std::invalid_argument("Null pointer passed to multiply_mm_blocked");
    if (rowsA <= 0 || colsA <= 0 || rowsB <= 0 || colsB <= 0)
        throw std::invalid_argument("Invalid dimensions in multiply_mm_blocked");
    if (colsA != rowsB)
        throw std::invalid_argument("Incompatible dimensions: colsA != rowsB");

    // Initialize result to zero
    std::memset(result, 0, static_cast<size_t>(rowsA) * colsB * sizeof(double));

    // Blocked multiplication: iterate over tiles
    for (int ii = 0; ii < rowsA; ii += block_size) {
        int i_end = std::min(ii + block_size, rowsA);
        for (int kk = 0; kk < colsA; kk += block_size) {
            int k_end = std::min(kk + block_size, colsA);
            for (int jj = 0; jj < colsB; jj += block_size) {
                int j_end = std::min(jj + block_size, colsB);

                // Multiply the block A[ii:i_end, kk:k_end] * B[kk:k_end, jj:j_end]
                // and accumulate into C[ii:i_end, jj:j_end]
                for (int i = ii; i < i_end; ++i) {
                    for (int k = kk; k < k_end; ++k) {
                        double a_ik = matrixA[i * colsA + k];
                        for (int j = jj; j < j_end; ++j) {
                            result[i * colsB + j] += a_ik * matrixB[k * colsB + j];
                        }
                    }
                }
            }
        }
    }
}
