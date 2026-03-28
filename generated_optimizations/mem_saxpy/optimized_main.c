#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

static uint64_t checksum_float(const float* a, size_t n) {
    double s = 0.0;
    for (size_t i = 0; i < n; i++) s += (double)a[i];
    return (uint64_t)(s * 1e3);
}

int main(int argc, char** argv) {
    size_t n = 50 * 1000 * 1000;  // 50M elements
    int reps = 3;

    if (argc > 1) n = (size_t)atoll(argv[1]);
    if (argc > 2) reps = atoi(argv[2]);

    float* x = (float*)aligned_alloc(64, n * sizeof(float));
    float* y = (float*)aligned_alloc(64, n * sizeof(float));
    float* z = (float*)aligned_alloc(64, n * sizeof(float));
    if (!x || !y || !z) {
        fprintf(stderr, "Allocation failed. Try smaller n.\n");
        return 2;
    }

    for (size_t i = 0; i < n; i++) {
        x[i] = (float)(i % 100) * 0.001f;
        y[i] = (float)((i + 7) % 100) * 0.002f;
        z[i] = 0.0f;
    }

    float a = 2.5f;

    for (int r = 0; r < reps; r++) {
        // Memory-bound: z = a*x + y
        // Compiler should vectorize at -O3; this is just a stable benchmark.
        for (size_t i = 0; i < n; i++) {
            z[i] = a * x[i] + y[i];
        }
    }

    printf("CHECKSUM=%llu\n", (unsigned long long)checksum_float(z, n));

    free(x);
    free(y);
    free(z);
    return 0;
}