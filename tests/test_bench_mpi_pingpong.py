from __future__ import annotations

import os
import pytest
from pathlib import Path
from statistics import mean, stdev

from .bench_utils import make_build, run_binary_repeated, parse_checksum, which

BENCH = Path("benchmarks/omp_imbalance")
EXE_NAME = "omp_imbalance"


def _std(times):
    return stdev(times) if len(times) > 1 else 0.0


@pytest.mark.benchmark
def test_omp_imbalance_schedule_tuning():
    """
    Compares baseline OpenMP schedule vs a tuned schedule on an intentionally
    imbalanced workload. This test is designed to:

      1) Ensure the benchmark builds and runs
      2) Ensure correctness is preserved (checksum invariant)
      3) Print quantitative runtime results (mean/std/speedup) for reporting
      4) Avoid flaky failures due to laptop noise by only failing on large regressions
    """

    # If no compiler, skip
    if which("cc") is None and which("clang") is None and which("gcc") is None:
        pytest.skip("No C compiler found")

    # Build; if OpenMP not configured, build will fail -> skip rather than fail
    try:
        make_build(BENCH)
    except RuntimeError as e:
        pytest.skip(f"OpenMP benchmark build failed (likely missing OpenMP runtime/toolchain): {e}")

    # Fixed args => deterministic checksum
    # (n_work, n_repeats) — adjust as needed for stronger signal
    args = ["250000", "10"]

    repeats = 5
    warmups = 1

    # Baseline: static schedule
    env_base = os.environ.copy()
    env_base["OMP_NUM_THREADS"] = env_base.get("OMP_NUM_THREADS", "4")
    env_base["OMP_SCHEDULE"] = "static"

    base = run_binary_repeated(
        Path(EXE_NAME),
        args=args,
        cwd=BENCH,
        env=env_base,
        repeats=repeats,
        warmups=warmups,
    )
    base_cs = parse_checksum(base.stdout_last)
    assert base_cs, "Expected checksum output in baseline run"

    # Tuned: dynamic schedule (often helps imbalance, but not guaranteed on all machines)
    env_tuned = os.environ.copy()
    env_tuned["OMP_NUM_THREADS"] = env_base["OMP_NUM_THREADS"]
    env_tuned["OMP_SCHEDULE"] = "dynamic,16"

    tuned = run_binary_repeated(
        Path(EXE_NAME),
        args=args,
        cwd=BENCH,
        env=env_tuned,
        repeats=repeats,
        warmups=warmups,
    )
    tuned_cs = parse_checksum(tuned.stdout_last)
    assert tuned_cs, "Expected checksum output in tuned run"

    # Correctness: schedule change must not change result
    assert base_cs == tuned_cs, "Checksum mismatch: schedule change altered results"

    # Compute metrics (mean already present; compute std from collected times)
    base_mean = base.mean
    tuned_mean = tuned.mean
    base_std = _std(base.times_s)
    tuned_std = _std(tuned.times_s)

    sp = (base_mean / tuned_mean) if tuned_mean > 0 else float("inf")

    # Print results for table building (use -s to show prints)
    print("\nOMP Imbalance Results")
    print(f"config: OMP_NUM_THREADS={env_base['OMP_NUM_THREADS']} args={args}")
    print(f"baseline: schedule={env_base['OMP_SCHEDULE']} mean_sec={base_mean:.6f} std_sec={base_std:.6f}")
    print(f"tuned:    schedule={env_tuned['OMP_SCHEDULE']} mean_sec={tuned_mean:.6f} std_sec={tuned_std:.6f}")
    print(f"speedup={sp:.3f}")
    print(f"checksum={base_cs}")

    # We DO NOT require tuned to be faster (laptop noise + scheduling overhead can invert results).
    # We only fail if tuned is catastrophically slower (indicates something broken).
    assert tuned_mean <= base_mean * 2.0, f"Tuned regressed too much (speedup={sp:.3f})"