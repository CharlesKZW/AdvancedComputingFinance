#include "linear_algebra.h"
#include <cstring>
#include <stdexcept>

// Team Member 2: Matrix-Vector Multiplication (Column-Major)
// matrix[i][j] = matrix[j * rows + i]  (column-major layout)
// result[i] = sum_j( matrix[i][j] * vector[j] )
void multiply_mv_col_major(const double* matrix, int rows, int cols,
                           const double* vector, double* result) {
    if (!matrix || !vector || !result)
        throw std::invalid_argument("Null pointer passed to multiply_mv_col_major");
    if (rows <= 0 || cols <= 0)
        throw std::invalid_argument("Invalid dimensions in multiply_mv_col_major");

    // Initialize result to zero
    for (int i = 0; i < rows; ++i) {
        result[i] = 0.0;
    }

    // Accumulate column by column
    // This accesses matrix columns sequentially (contiguous in memory for col-major)
    for (int j = 0; j < cols; ++j) {
        const double* col = matrix + j * rows;
        double vj = vector[j];
        for (int i = 0; i < rows; ++i) {
            result[i] += col[i] * vj;
        }
    }
}
