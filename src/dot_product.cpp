#include "linear_algebra.h"

// Non-inline version of dot product (for inlining experiments)
// Compare with the inline version in linear_algebra.h
double dot_product(const double* a, const double* b, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}
