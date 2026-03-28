#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// Simple deterministic pseudo-work
static inline double work(int iters) {
    double x = 1.0;
    for (int i = 0; i < iters; i++) {
        x = x * 1.0000001 + 0.0000003;
        if (x > 2.0) x -= 1.0;
    }
    return x;
}

int main(int argc, char** argv) {
    int n = 200000;      // iterations
    int heavy = 400;     // heavy work iters
    int light = 40;      // light work iters
    int skew = 10;       // every skew-th iteration is heavy

    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) skew = atoi(argv[2]);

#ifdef _OPENMP
    // Stabilize OpenMP execution by defaulting to thread pinning/affinity.
    // Only set defaults if user/job launcher hasn't specified them.
    // This is a low-risk way to reduce OS scheduling migration that can
    // inflate barrier time and worsen imbalance.
    if (getenv("OMP_PROC_BIND") == NULL) {
        (void)setenv("OMP_PROC_BIND", "true", 0);
    }
    if (getenv("OMP_PLACES") == NULL) {
        (void)setenv("OMP_PLACES", "cores", 0);
    }
#endif

    double sum = 0.0;

    // Imbalanced loop: periodic heavy iterations
    #pragma omp parallel for reduction(+:sum) schedule(runtime)
    for (int i = 0; i < n; i++) {
        int iters = (i % skew == 0) ? heavy : light;
        sum += work(iters);
    }

    // Print checksum so tests can verify correctness across schedules
    // (should be stable for fixed n,skew)
    uint64_t cs = (uint64_t)(sum * 1e6);
    printf("CHECKSUM=%llu\n", (unsigned long long)cs);

    return 0;
}