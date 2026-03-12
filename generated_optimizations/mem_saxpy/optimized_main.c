#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

static uint64_t checksum_double(const double* a, size_t n) {
    double s = 0.0;
    for (size_t i = 0; i < n; i++) s += a[i];
    return (uint64_t)(s * 1e3);
}

#if defined(__GNUC__) || defined(__clang__)
#define ASSUME_ALIGNED_64(p) (__builtin_assume_aligned((p), 64))
#else
#define ASSUME_ALIGNED_64(p) (p)
#endif

int main(int argc, char** argv) {
    size_t n = 50 * 1000 * 1000;  // 50M doubles ~ 400MB total across arrays if 3 arrays => adjust if needed
    int reps = 3;

    if (argc > 1) n = (size_t)atoll(argv[1]);
    if (argc > 2) reps = atoi(argv[2]);

    double* x = (double*)aligned_alloc(64, n * sizeof(double));
    double* y = (double*)aligned_alloc(64, n * sizeof(double));
    double* z = (double*)aligned_alloc(64, n * sizeof(double));
    if (!x || !y || !z) {
        fprintf(stderr, "Allocation failed. Try smaller n.\n");
        return 2;
    }

    for (size_t i = 0; i < n; i++) {
        x[i] = (double)(i % 100) * 0.001;
        y[i] = (double)((i + 7) % 100) * 0.002;
        z[i] = 0.0;
    }

    double a = 2.5;

    for (int r = 0; r < reps; r++) {
        // Memory-bound: z = a*x + y
        // Compiler should vectorize at -O3; this is just a stable benchmark.
        double* __restrict xr = (double* __restrict)ASSUME_ALIGNED_64(x);
        double* __restrict yr = (double* __restrict)ASSUME_ALIGNED_64(y);
        double* __restrict zr = (double* __restrict)ASSUME_ALIGNED_64(z);

        #if defined(_OPENMP)
        #pragma omp simd
        #endif
        for (size_t i = 0; i < n; i++) {
            zr[i] = a * xr[i] + yr[i];
        }
    }

    printf("CHECKSUM=%llu\n", (unsigned long long)checksum_double(z, n));

    free(x); free(y); free(z);
    return 0;
}