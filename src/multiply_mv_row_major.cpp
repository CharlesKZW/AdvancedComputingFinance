#include "linear_algebra.h"
#include <cstring>
#include <stdexcept>

// Team Member 1: Matrix-Vector Multiplication (Row-Major)
// matrix[i][j] = matrix[i * cols + j]
// result[i] = sum_j( matrix[i][j] * vector[j] )
void multiply_mv_row_major(const double* matrix, int rows, int cols,
                           const double* vector, double* result) {
    if (!matrix || !vector || !result)
        throw std::invalid_argument("Null pointer passed to multiply_mv_row_major");
    if (rows <= 0 || cols <= 0)
        throw std::invalid_argument("Invalid dimensions in multiply_mv_row_major");

    for (int i = 0; i < rows; ++i) {
        double sum = 0.0;
        const double* row = matrix + i * cols;
        for (int j = 0; j < cols; ++j) {
            sum += row[j] * vector[j];
        }
        result[i] = sum;
    }
}
