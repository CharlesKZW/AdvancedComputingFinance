#include "linear_algebra.h"
#include <cstring>
#include <stdexcept>

// Team Member 3: Matrix-Matrix Multiplication (Naive, Row-Major)
// C[i][j] = sum_k( A[i][k] * B[k][j] )
// All matrices in row-major order
void multiply_mm_naive(const double* matrixA, int rowsA, int colsA,
                       const double* matrixB, int rowsB, int colsB,
                       double* result) {
    if (!matrixA || !matrixB || !result)
        throw std::invalid_argument("Null pointer passed to multiply_mm_naive");
    if (rowsA <= 0 || colsA <= 0 || rowsB <= 0 || colsB <= 0)
        throw std::invalid_argument("Invalid dimensions in multiply_mm_naive");
    if (colsA != rowsB)
        throw std::invalid_argument("Incompatible dimensions: colsA != rowsB");

    // Initialize result to zero
    std::memset(result, 0, static_cast<size_t>(rowsA) * colsB * sizeof(double));

    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            double sum = 0.0;
            for (int k = 0; k < colsA; ++k) {
                // A[i][k] = A[i * colsA + k] — sequential access along row of A
                // B[k][j] = B[k * colsB + j] — strided access down column of B
                sum += matrixA[i * colsA + k] * matrixB[k * colsB + j];
            }
            result[i * colsB + j] = sum;
        }
    }
}
