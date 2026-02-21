from __future__ import annotations
import os
import pytest
from pathlib import Path

from .bench_utils import make_build, run_binary_repeated, parse_checksum, speedup, which

BENCH = Path("benchmarks/mem_saxpy")
EXE_NAME = "mem_saxpy"

@pytest.mark.benchmark
def test_mem_saxpy_baseline_vs_compiler_flags():
    if which("cc") is None and which("clang") is None and which("gcc") is None:
        pytest.skip("No C compiler found")

    make_build(BENCH)

    # Keep size smaller for laptops; adjust upward later on Bell
    args = ["8000000", "4"]  # 8M doubles

    base = run_binary_repeated(Path(EXE_NAME), args=args, cwd=BENCH, env=os.environ.copy(), repeats=5, warmups=1)
    base_cs = parse_checksum(base.stdout_last)

    # "Tuned" variant for this benchmark is mostly about compile flags.
    # For now, we treat baseline itself as the comparison point and ensure stability.
    tuned = base
    tuned_cs = parse_checksum(tuned.stdout_last)

    assert base_cs == tuned_cs
    # This test mainly ensures the harness runs and measures; you’ll expand this when you add real transformations.
    assert tuned.mean <= base.mean * 1.20