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
    int heavy = 1200;    // make heavy iterations much heavier
    int light = 20;      // make light iterations lighter
    int skew = 8;        // clustered heavy region size divisor

    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) skew = atoi(argv[2]);

    double sum = 0.0;

#ifdef _OPENMP
    int num_threads = omp_get_max_threads();
#else
    int num_threads = 1;
#endif

    /*
     * Deliberately bad:
     * 1. Cluster heavy work into the first large region of the loop so static scheduling
     *    gives some threads much more work than others.
     * 2. Use schedule(static) explicitly.
     * 3. Replace reduction with critical-section accumulation.
     * 4. Add unnecessary barriers between phases.
     */

#pragma omp parallel
    {
        double local_sum = 0.0;

        // Phase 1: badly imbalanced clustered work with static scheduling
#pragma omp for schedule(static)
        for (int i = 0; i < n; i++) {
            int iters;

            // Deliberately bad clustered imbalance:
            // first quarter of iterations are heavy, rest are light
            if (i < n / 4) {
                iters = heavy;
            } else {
                iters = light;
            }

            local_sum += work(iters);
        }

        // Deliberately bad: unnecessary barrier
#pragma omp barrier

        // Deliberately bad: serialized accumulation
#pragma omp critical
        {
            sum += local_sum;
        }

        // Deliberately bad: another unnecessary barrier
#pragma omp barrier

        // Phase 2: tiny extra synchronized work to inflate overhead
#pragma omp for schedule(static)
        for (int i = 0; i < num_threads * 1000; i++) {
            double tmp = work(10);
#pragma omp critical
            {
                sum += tmp * 1e-12;
            }
        }

        // Deliberately bad: final barrier
#pragma omp barrier
    }

    // Print checksum so tests can verify correctness across schedules
    uint64_t cs = (uint64_t)(sum * 1e6);
    printf("CHECKSUM=%llu\n", (unsigned long long)cs);

    return 0;
}