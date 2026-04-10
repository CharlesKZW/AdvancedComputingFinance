#include "linear_algebra.h"
#include <cstring>
#include <stdexcept>

// Team Member 4: Matrix-Matrix Multiplication (Transposed B)
// C[i][j] = sum_k( A[i][k] * B^T[j][k] )
// matrixB_transposed stores B^T in row-major order, so B^T[j][k] = matrixB_transposed[j * colsA + k]
// where colsA == rowsB (the shared inner dimension)
void multiply_mm_transposed_b(const double* matrixA, int rowsA, int colsA,
                              const double* matrixB_transposed, int rowsB, int colsB,
                              double* result) {
    if (!matrixA || !matrixB_transposed || !result)
        throw std::invalid_argument("Null pointer passed to multiply_mm_transposed_b");
    if (rowsA <= 0 || colsA <= 0 || rowsB <= 0 || colsB <= 0)
        throw std::invalid_argument("Invalid dimensions in multiply_mm_transposed_b");
    if (colsA != rowsB)
        throw std::invalid_argument("Incompatible dimensions: colsA != rowsB");

    for (int i = 0; i < rowsA; ++i) {
        const double* rowA = matrixA + i * colsA;
        for (int j = 0; j < colsB; ++j) {
            // B^T[j] gives us what was column j of B, now stored as a contiguous row
            const double* rowBT = matrixB_transposed + j * rowsB;
            double sum = 0.0;
            for (int k = 0; k < colsA; ++k) {
                // Both accesses are sequential — excellent cache locality
                sum += rowA[k] * rowBT[k];
            }
            result[i * colsB + j] = sum;
        }
    }
}
