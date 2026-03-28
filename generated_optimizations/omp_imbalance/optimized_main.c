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
    /*
     * Reduce OS thread migration and synchronization variance by pinning threads.
     * Only set defaults if the user/job launcher has not already specified them.
     */
    {
        const char* pb = getenv("OMP_PROC_BIND");
        const char* pl = getenv("OMP_PLACES");
#if defined(_POSIX_C_SOURCE) || defined(__unix__) || defined(__APPLE__)
        if (!pb || !pb[0]) setenv("OMP_PROC_BIND", "close", 0);
        if (!pl || !pl[0]) setenv("OMP_PLACES", "cores", 0);
#else
        (void)pb;
        (void)pl;
#endif
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