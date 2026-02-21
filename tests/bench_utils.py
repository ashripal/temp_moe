from __future__ import annotations
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class RunResult:
    times_s: List[float]
    stdout_last: str
    stderr_last: str

    @property
    def mean(self) -> float:
        return sum(self.times_s) / max(1, len(self.times_s))

    @property
    def stdev(self) -> float:
        if len(self.times_s) < 2:
            return 0.0
        m = self.mean
        var = sum((t - m) ** 2 for t in self.times_s) / (len(self.times_s) - 1)
        return var ** 0.5


def which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


def run_cmd(cmd: List[str], cwd: Path, env: Optional[Dict[str, str]] = None, timeout_s: int = 120) -> Tuple[int, str, str]:
    p = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    return p.returncode, p.stdout, p.stderr


def make_build(bench_dir: Path, target: str = "all") -> None:
    rc, out, err = run_cmd(["make", "-s", target], cwd=bench_dir)
    if rc != 0:
        raise RuntimeError(f"Build failed in {bench_dir} (rc={rc}).\nSTDOUT:\n{out}\nSTDERR:\n{err}")


def run_binary_repeated(
    exe_path: Path,
    args: List[str],
    cwd: Path,
    env: Optional[Dict[str, str]] = None,
    repeats: int = 5,
    warmups: int = 1,
    timeout_s: int = 120,
) -> RunResult:
    times: List[float] = []
    stdout_last = ""
    stderr_last = ""

    # Always execute relative to cwd
    exe_cmd = f"./{exe_path.name}"

    # Warmups (not counted)
    for _ in range(warmups):
        rc, out, err = run_cmd([exe_cmd] + args, cwd=cwd, env=env, timeout_s=timeout_s)
        if rc != 0:
            raise RuntimeError(f"Warmup failed: {exe_cmd} rc={rc}\nSTDOUT:\n{out}\nSTDERR:\n{err}")

    for _ in range(repeats):
        t0 = time.perf_counter()
        rc, out, err = run_cmd([exe_cmd] + args, cwd=cwd, env=env, timeout_s=timeout_s)
        t1 = time.perf_counter()
        if rc != 0:
            raise RuntimeError(f"Run failed: {exe_cmd} rc={rc}\nSTDOUT:\n{out}\nSTDERR:\n{err}")
        times.append(t1 - t0)
        stdout_last, stderr_last = out, err

    return RunResult(times_s=times, stdout_last=stdout_last, stderr_last=stderr_last)

def parse_checksum(stdout: str) -> str:
    """
    Each benchmark prints: CHECKSUM=<value>
    """
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("CHECKSUM="):
            return line.split("=", 1)[1].strip()
    raise ValueError(f"Could not find CHECKSUM=... in stdout:\n{stdout}")


def speedup(baseline_mean: float, tuned_mean: float) -> float:
    if tuned_mean <= 0:
        return 0.0
    return baseline_mean / tuned_mean