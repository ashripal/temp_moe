# AGENTS Notes: Static Analysis Snapshot (no-polybench)

Date: 2026-05-11
Branch analyzed: `no-polybench`

## Scope

This note captures the static analysis completed so far, with emphasis on:

1. The new benchmark strategy in this branch
2. Whether benchmark-related tests are meaningful for the new benchmark set
3. Immediate risks before cluster execution


## Executive Summary

- The new branch direction is better aligned with real HPC benchmarking than the prior PolyBench-only path.
- The repository now includes a managed external benchmark bundle (EPCC + OSU + NPB), which is a clear improvement for MPI/OpenMP/hybrid coverage.
- However, test coverage is still aligned to the old local benchmarks and does not validate the new external benchmark execution path.
- Conclusion: benchmark selection quality improved, but test relevance lags behind implementation.


## What Changed in This Branch

Compared to `origin/main`, the branch adds/changes:

- `benchmarks/external/` bundle management and documentation
- Git submodules for external suites (`.gitmodules`)
- Large refactor/extension of `cluster_tests/orchestrator.py`
- Updated cluster runner `runner.sbatch`
- Prompt constraints for generator safety
- Updated optimization catalog files

No test files were changed relative to `origin/main`.


## Benchmark Representativeness (HPC)

### Managed External Benchmark Families

Defined in `cluster_tests/orchestrator.py` as 15 managed IDs:

- EPCC OpenMP microbenchmarks (OpenMP overhead/scheduling/tasking)
- OSU micro-benchmarks (MPI latency/bandwidth/collectives)
- NPB MPI kernels
- NPB-MZ hybrid kernels (MPI + OpenMP)

This is substantially more HPC-relevant than serial PolyBench kernels for distributed and hybrid behavior.


### Static signal checks (quick scan)

Observed code-level parallel markers in external suites:

- EPCC v3.1: OpenMP pragmas/API usage present
- OSU MPI suite: widespread `MPI_*` usage
- NPB MPI: widespread `MPI_*` usage in Fortran sources
- NPB-MZ MPI: MPI plus OpenMP directives in MZ sources

These suites clearly include real parallel communication/synchronization patterns.


## Test Relevance Audit

### Current status

- Existing tests in `tests/` still target the legacy trio:
  - `mem_saxpy`
  - `omp_imbalance`
  - `mpi_pingpong`
- No tests currently cover:
  - external benchmark ID expansion (`parallel-shortlist`, `all-external`)
  - external command construction and task constraints
  - external benchmark execution/reporting logic


### Concrete defects/mismatches found

1. Miswired MPI benchmark test
   - `tests/test_bench_mpi_pingpong.py` points to OpenMP benchmark paths/names (`omp_imbalance`) instead of `mpi_pingpong`.
   - This weakens MPI test confidence.

2. New benchmark path has no automated validation
   - `cluster_tests/orchestrator.py` external path lacks unit/integration tests.

3. External benchmark correctness signal is weak
   - In `run_external_benchmark`, result marks `"verified": True` by construction.
   - Runtime success is checked, but no benchmark-specific output parsing is used to validate metric integrity.


## Static Quality Findings (New Path)

### Strengths

- External benchmark metadata is explicit (`ExternalBenchmarkSpec` model).
- Task-count constraints are encoded (exact/min/power-of-two policies).
- `runner.sbatch` supports external-only execution without requiring OpenAI API key.
- Shell scripts in `benchmarks/external/` pass syntax checks.


### Risks

1. High complexity concentration
   - `cluster_tests/orchestrator.py` is very large; `main` and multiple long functions increase regression risk.

2. Import-time coupling for external-only mode
   - `cluster_tests/orchestrator.py` imports `openai` at module import time.
   - If `openai` is missing, even external-only mode cannot start.

3. Obsolete local smoke test script
   - `cluster_tests/test.sh` assumes old file locations/old orchestrator behavior (including expected `NotImplementedError`).

4. Documentation drift
   - `README.md` references `cluster_tests/README.md`, but that file is missing.

5. Line ending inconsistency
   - `cluster_tests/orchestrator.py` uses CRLF in a Linux-oriented repo.


## Do Benchmark Tests Make Sense Right Now?

Short answer: partially.

- Legacy tests still provide some smoke signal for old local benchmarks.
- They do not validate the new external benchmark strategy that is central to this branch.
- Therefore, as-is, benchmark tests are not sufficient evidence for the branch's new HPC benchmark direction.


## Recommended Priority Actions

### P0 (immediate before deeper cluster studies)

1. Fix `tests/test_bench_mpi_pingpong.py` to target actual MPI benchmark files/executable.
2. Add unit tests for external benchmark orchestration primitives:
   - `expand_requested_benchmarks`
   - `resolve_external_task_count`
   - `build_external_command`


### P1 (stabilize external benchmark path)

1. Add integration test(s) for external mode with mocked subprocess responses.
2. Parse minimal expected output signatures per family:
   - OSU: expected latency/bandwidth table markers
   - NPB: run summary markers and class labels
3. Replace unconditional `verified=True` with lightweight output sanity checks.


### P2 (maintainability)

1. Move `openai` import into the MoE-specific code path (lazy import) to keep external-only path decoupled.
2. Split `cluster_tests/orchestrator.py` into modules (specs, launch policy, execution, reporting).
3. Refresh or replace `cluster_tests/test.sh` to reflect current architecture.


### P3 (docs and hygiene)

1. Update top-level README with explicit external benchmark workflow.
2. Add missing `cluster_tests/README.md` or remove stale reference.
3. Normalize CRLF/LF style.


## Suggested Cluster Validation Path

For immediate cluster-side continuation, validate external shortlist first:

1. `./benchmarks/external/build_selected.sh`
2. `./benchmarks/external/smoke_test_selected.sh`
3. `python3 cluster_tests/orchestrator.py --kernels parallel-shortlist --use-srun --nodes <N> --tasks-per-node <T> --cpus <C> --runs <R>`

Then compare behavior with and without PolyBench kernels to isolate external benchmark infrastructure issues.
