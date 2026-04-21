# Parallel Benchmark Bundle

This directory holds public benchmark suites that fit the HPC focus of this
project better than the current serial PolyBench kernels.

The suites are tracked as git submodules:

- `epcc-openmp-microbenchmarks`: OpenMP microbenchmarks
- `npb`: NAS Parallel Benchmarks
- `osu-micro-benchmarks`: OSU MPI microbenchmarks

## Selected Benchmarks

The current 15-benchmark shortlist is:

1. `syncbench`
2. `schedbench`
3. `taskbench`
4. `arraybench_81`
5. `osu_latency`
6. `osu_bw`
7. `osu_bibw`
8. `osu_allreduce`
9. `osu_alltoall`
10. `osu_barrier`
11. `cg.S.x`
12. `mg.S.x`
13. `ft.S.x`
14. `bt-mz.S.x`
15. `sp-mz.S.x`

## Build

Run:

```bash
./benchmarks/external/build_selected.sh
```

This script builds:

- EPCC OpenMP v3.1 benchmarks in place
- OSU MPI benchmarks into `preflight_build/osu`
- NPB MPI and NPB-MZ binaries into their suite-local `bin/` directories

## Smoke Test

Run:

```bash
./benchmarks/external/smoke_test_selected.sh
```

The smoke test runs one example from each family:

- EPCC `syncbench`
- OSU `osu_latency`
- NPB-MZ `bt-mz.S.x`

## Notes

- EPCC v4.0 did not link cleanly against the local OpenMP runtime because of
  `omp_init_lock_with_hint`, so the validated path uses EPCC v3.1.
- NPB 3.4 requires compatibility flags with modern `gfortran`:
  `-fallow-argument-mismatch -fallow-invalid-boz`
- MPI execution may fail inside restricted sandboxes even when compilation is
  correct. The smoke-test script is meant for a normal cluster login or batch
  environment.
