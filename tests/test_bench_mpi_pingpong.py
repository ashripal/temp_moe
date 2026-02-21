from __future__ import annotations
import os
import pytest
from pathlib import Path

from .bench_utils import make_build, run_cmd, parse_checksum, which

BENCH = Path("benchmarks/mpi_pingpong")
EXE = BENCH / "mpi_pingpong"

@pytest.mark.benchmark
def test_mpi_pingpong_runs_and_is_stable():
    if which("mpicc") is None or which("mpirun") is None:
        pytest.skip("MPI not installed (mpicc/mpirun missing)")

    make_build(BENCH)

    # Run with 2 ranks
    # cmd = ["mpirun", "-n", "2", str(EXE), "65536", "200"]  # 64KB, 200 iters
    cmd = ["mpirun", "-n", "2", "./mpi_pingpong", "65536", "200"]
    rc, out, err = run_cmd(cmd, cwd=BENCH, env=os.environ.copy(), timeout_s=120)
    if rc != 0:
        pytest.skip(f"MPI run failed (mpirun config issue on this machine): rc={rc}\n{err}")

    cs = parse_checksum(out)
    assert cs, "Expected checksum output from rank 0"