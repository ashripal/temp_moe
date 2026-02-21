from __future__ import annotations
import os
import pytest
from pathlib import Path

from .bench_utils import make_build, run_binary_repeated, parse_checksum, speedup, which

BENCH = Path("benchmarks/omp_imbalance")
EXE = BENCH / "omp_imbalance"

@pytest.mark.benchmark
def test_omp_imbalance_schedule_tuning():
    # If no compiler, skip
    if which("cc") is None and which("clang") is None and which("gcc") is None:
        pytest.skip("No C compiler found")

    # Build; if OpenMP not configured, build will fail → skip rather than fail
    try:
        make_build(BENCH)
    except RuntimeError as e:
        pytest.skip(f"OpenMP benchmark build failed (likely missing OpenMP runtime): {e}")

    # Fixed args => deterministic checksum
    args = ["250000", "10"]

    # Baseline: static schedule tends to be worse for imbalance
    env_base = os.environ.copy()
    env_base["OMP_NUM_THREADS"] = env_base.get("OMP_NUM_THREADS", "4")
    env_base["OMP_SCHEDULE"] = "static"

    base = run_binary_repeated(EXE, args=args, cwd=BENCH, env=env_base, repeats=5, warmups=1)
    base_cs = parse_checksum(base.stdout_last)

    # Tuned: dynamic/guided often helps
    env_tuned = os.environ.copy()
    env_tuned["OMP_NUM_THREADS"] = env_base["OMP_NUM_THREADS"]
    env_tuned["OMP_SCHEDULE"] = "dynamic,16"

    tuned = run_binary_repeated(EXE, args=args, cwd=BENCH, env=env_tuned, repeats=5, warmups=1)
    tuned_cs = parse_checksum(tuned.stdout_last)

    assert base_cs == tuned_cs, "Checksum mismatch: schedule change changed results"

    sp = speedup(base.mean, tuned.mean)
    # We don't require huge gains on every machine, but should not regress badly
    assert tuned.mean <= base.mean * 1.10, f"Tuned regressed too much (speedup={sp:.3f})"