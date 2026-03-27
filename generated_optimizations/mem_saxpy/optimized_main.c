#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L)
#  define RESTRICT restrict
#elif defined(_MSC_VER)
#  define RESTRICT __restrict
#else
#  define RESTRICT __restrict__
#endif

static uint64_t checksum_double(const double* a, size_t n) {
    double s = 0.0;
    for (size_t i = 0; i < n; i++) s += a[i];
    return (uint64_t)(s * 1e3);
}

int main(int argc, char** argv) {
    size_t n = 50 * 1000 * 1000;  // 50M doubles ~ 400MB total across arrays if 3 arrays => adjust if needed
    int reps = 3;

    if (argc > 1) n = (size_t)atoll(argv[1]);
    if (argc > 2) reps = atoi(argv[2]);

    double* x0 = (double*)aligned_alloc(64, n * sizeof(double));
    double* y0 = (double*)aligned_alloc(64, n * sizeof(double));
    double* z0 = (double*)aligned_alloc(64, n * sizeof(double));
    if (!x0 || !y0 || !z0) {
        fprintf(stderr, "Allocation failed. Try smaller n.\n");
        return 2;
    }

    double* RESTRICT x = x0;
    double* RESTRICT y = y0;
    double* RESTRICT z = z0;

#if defined(__GNUC__) || defined(__clang__)
    x = (double* RESTRICT)__builtin_assume_aligned(x, 64);
    y = (double* RESTRICT)__builtin_assume_aligned(y, 64);
    z = (double* RESTRICT)__builtin_assume_aligned(z, 64);
#endif

    for (size_t i = 0; i < n; i++) {
        x[i] = (double)(i % 100) * 0.001;
        y[i] = (double)((i + 7) % 100) * 0.002;
        z[i] = 0.0;
    }

    double a = 2.5;

    for (int r = 0; r < reps; r++) {
        // Memory-bound: z = a*x + y
        // Compiler should vectorize at -O3; this is just a stable benchmark.
#if defined(__INTEL_COMPILER)
#       pragma ivdep
#       pragma vector always
#elif defined(__clang__)
#       pragma clang loop vectorize(enable) interleave(enable)
#elif defined(__GNUC__)
#       pragma GCC ivdep
#endif
#if defined(_OPENMP)
#       pragma omp simd
#endif
        for (size_t i = 0; i < n; i++) {
            z[i] = a * x[i] + y[i];
        }
    }

    printf("CHECKSUM=%llu\n", (unsigned long long)checksum_double(z, n));

    free(x0); free(y0); free(z0);
    return 0;
}