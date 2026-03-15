from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import difflib


# ---------- Data models ----------


@dataclass
class BuildMetrics:
    compile_success: bool
    return_code: int
    build_time_seconds: float
    warning_count: int
    error_count: int
    binary_size_bytes: Optional[int]
    stdout: str
    stderr: str
    compile_command: List[str]


@dataclass
class RunTrial:
    success: bool
    return_code: int
    wall_time_seconds: float
    stdout: str
    stderr: str
    timed_out: bool


@dataclass
# class RunMetrics:
#     run_success: bool
#     successful_runs: int
#     total_runs: int
#     timeout_count: int
#     crash_count: int
#     exit_codes: List[int]
#     wall_time_mean: Optional[float]
#     wall_time_median: Optional[float]
#     wall_time_std: Optional[float]
#     wall_time_min: Optional[float]
#     wall_time_max: Optional[float]
#     representative_stdout: str
#     representative_stderr: str
#     trials: List[Dict[str, Any]]
@dataclass
class RunMetrics:
    run_success: bool
    successful_runs: int
    total_runs: int
    timed_trials_count: int
    warmup_trials_count: int
    timeout_count: int
    crash_count: int
    exit_codes: List[int]
    wall_time_mean: Optional[float]
    wall_time_median: Optional[float]
    wall_time_std: Optional[float]
    wall_time_min: Optional[float]
    wall_time_max: Optional[float]
    wall_time_trimmed_mean: Optional[float]
    runtime_coefficient_of_variation: Optional[float]
    representative_stdout: str
    representative_stderr: str
    trials: List[Dict[str, Any]]


@dataclass
class OutputComparison:
    exit_code_match: bool
    stdout_match: bool
    stderr_match: bool
    correctness_pass: bool


@dataclass
class DiffMetrics:
    lines_original: int
    lines_optimized: int
    lines_added: int
    lines_deleted: int
    lines_changed: int
    diff_hunk_count: int
    percent_file_changed: float


@dataclass
# class ComparisonReport:
#     original_source: str
#     optimized_source: str
#     original_build: BuildMetrics
#     optimized_build: BuildMetrics
#     original_run: Optional[RunMetrics]
#     optimized_run: Optional[RunMetrics]
#     output_comparison: Optional[OutputComparison]
#     diff_metrics: DiffMetrics
#     speedup: Optional[float]
#     percent_improvement: Optional[float]
@dataclass
class ComparisonReport:
    original_source: str
    optimized_source: str
    original_build: BuildMetrics
    optimized_build: BuildMetrics
    original_run: Optional[RunMetrics]
    optimized_run: Optional[RunMetrics]
    output_comparison: Optional[OutputComparison]
    diff_metrics: DiffMetrics
    speedup: Optional[float]
    percent_improvement: Optional[float]
    percent_change: Optional[float]
    percent_slowdown: Optional[float]
    is_regression: Optional[bool]
    median_speedup: Optional[float]
    median_percent_improvement: Optional[float]
    trimmed_mean_speedup: Optional[float]
    trimmed_mean_percent_improvement: Optional[float]
    likely_significant: Optional[bool]
    significance_reason: Optional[str]


# ---------- Build helpers ----------
def trimmed_mean(values: List[float], proportion_to_cut: float = 0.1) -> Optional[float]:
    if not values:
        return None
    if len(values) < 3:
        return statistics.mean(values)

    sorted_vals = sorted(values)
    cut = int(len(sorted_vals) * proportion_to_cut)

    if cut == 0 or (2 * cut) >= len(sorted_vals):
        return statistics.mean(sorted_vals)

    trimmed = sorted_vals[cut:-cut]
    if not trimmed:
        return statistics.mean(sorted_vals)

    return statistics.mean(trimmed)


def coefficient_of_variation(values: List[float]) -> Optional[float]:
    if not values:
        return None
    mean_val = statistics.mean(values)
    if mean_val == 0:
        return None
    if len(values) == 1:
        return 0.0
    return statistics.stdev(values) / mean_val


def compare_float_lists_summary(
    original_times: List[float],
    optimized_times: List[float],
) -> Dict[str, Optional[float | bool | str]]:
    if not original_times or not optimized_times:
        return {
            "mean_speedup": None,
            "median_speedup": None,
            "trimmed_mean_speedup": None,
            "mean_percent_improvement": None,
            "median_percent_improvement": None,
            "trimmed_mean_percent_improvement": None,
            "likely_significant": None,
            "significance_reason": "Insufficient timing data.",
        }

    orig_mean = statistics.mean(original_times)
    opt_mean = statistics.mean(optimized_times)
    orig_median = statistics.median(original_times)
    opt_median = statistics.median(optimized_times)
    orig_trimmed = trimmed_mean(original_times)
    opt_trimmed = trimmed_mean(optimized_times)

    mean_speedup = (orig_mean / opt_mean) if opt_mean > 0 else None
    median_speedup = (orig_median / opt_median) if opt_median > 0 else None
    trimmed_mean_speedup = (
        (orig_trimmed / opt_trimmed)
        if (orig_trimmed is not None and opt_trimmed is not None and opt_trimmed > 0)
        else None
    )

    mean_improvement = ((orig_mean - opt_mean) / orig_mean * 100.0) if orig_mean > 0 else None
    median_improvement = ((orig_median - opt_median) / orig_median * 100.0) if orig_median > 0 else None
    trimmed_mean_improvement = (
        ((orig_trimmed - opt_trimmed) / orig_trimmed * 100.0)
        if (orig_trimmed is not None and opt_trimmed is not None and orig_trimmed > 0)
        else None
    )

    orig_cv = coefficient_of_variation(original_times)
    opt_cv = coefficient_of_variation(optimized_times)

    likely_significant = None
    reason = "Insufficient data for heuristic."

    if (
        median_improvement is not None
        and trimmed_mean_improvement is not None
        and orig_cv is not None
        and opt_cv is not None
    ):
        # Simple heuristic:
        # treat the result as likely meaningful only if the robust improvement
        # is at least 5% and runtime noise is not too large.
        likely_significant = (
            median_improvement >= 5.0
            and trimmed_mean_improvement >= 5.0
            and orig_cv < 0.10
            and opt_cv < 0.10
        )

        reason = (
            f"median_improvement={median_improvement:.2f}%, "
            f"trimmed_mean_improvement={trimmed_mean_improvement:.2f}%, "
            f"original_cv={orig_cv:.3f}, optimized_cv={opt_cv:.3f}"
        )

    return {
        "mean_speedup": mean_speedup,
        "median_speedup": median_speedup,
        "trimmed_mean_speedup": trimmed_mean_speedup,
        "mean_percent_improvement": mean_improvement,
        "median_percent_improvement": median_improvement,
        "trimmed_mean_percent_improvement": trimmed_mean_improvement,
        "likely_significant": likely_significant,
        "significance_reason": reason,
    }


def count_warnings(stderr_text: str) -> int:
    # Generic compiler-warning count.
    return len(re.findall(r"\bwarning\b", stderr_text, flags=re.IGNORECASE))


def count_errors(stderr_text: str) -> int:
    # Generic compiler-error count.
    return len(re.findall(r"\berror\b", stderr_text, flags=re.IGNORECASE))


def build_c_file(
    source_path: Path,
    output_binary: Path,
    compiler: str,
    cflags: List[str],
) -> BuildMetrics:
    compile_cmd = [compiler, str(source_path), "-o", str(output_binary), *cflags]

    start = time.perf_counter()
    completed = subprocess.run(
        compile_cmd,
        capture_output=True,
        text=True,
    )
    end = time.perf_counter()

    binary_size = output_binary.stat().st_size if output_binary.exists() else None

    return BuildMetrics(
        compile_success=(completed.returncode == 0 and output_binary.exists()),
        return_code=completed.returncode,
        build_time_seconds=end - start,
        warning_count=count_warnings(completed.stderr),
        error_count=count_errors(completed.stderr),
        binary_size_bytes=binary_size,
        stdout=completed.stdout,
        stderr=completed.stderr,
        compile_command=compile_cmd,
    )


# ---------- Run helpers ----------


def run_binary_once(
    binary_path: Path,
    program_args: List[str],
    timeout_seconds: float,
    run_prefix: List[str]
) -> RunTrial:
    # cmd = [str(binary_path), *program_args]
    cmd = [*run_prefix, str(binary_path), *program_args]

    start = time.perf_counter()
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        end = time.perf_counter()

        return RunTrial(
            success=(completed.returncode == 0),
            return_code=completed.returncode,
            wall_time_seconds=end - start,
            stdout=completed.stdout,
            stderr=completed.stderr,
            timed_out=False,
        )

    except subprocess.TimeoutExpired as exc:
        end = time.perf_counter()
        return RunTrial(
            success=False,
            return_code=-1,
            wall_time_seconds=end - start,
            stdout=exc.stdout or "",
            stderr=exc.stderr or "",
            timed_out=True,
        )


# def summarize_trials(trials: List[RunTrial]) -> RunMetrics:
#     successful = [t for t in trials if t.success and not t.timed_out]
#     wall_times = [t.wall_time_seconds for t in successful]

#     representative_stdout = trials[0].stdout if trials else ""
#     representative_stderr = trials[0].stderr if trials else ""

#     crash_count = sum(1 for t in trials if (not t.timed_out and t.return_code != 0))
#     timeout_count = sum(1 for t in trials if t.timed_out)

#     return RunMetrics(
#         run_success=(len(successful) == len(trials) and len(trials) > 0),
#         successful_runs=len(successful),
#         total_runs=len(trials),
#         timeout_count=timeout_count,
#         crash_count=crash_count,
#         exit_codes=[t.return_code for t in trials],
#         wall_time_mean=(statistics.mean(wall_times) if wall_times else None),
#         wall_time_median=(statistics.median(wall_times) if wall_times else None),
#         wall_time_std=(statistics.stdev(wall_times) if len(wall_times) >= 2 else 0.0 if len(wall_times) == 1 else None),
#         wall_time_min=(min(wall_times) if wall_times else None),
#         wall_time_max=(max(wall_times) if wall_times else None),
#         representative_stdout=representative_stdout,
#         representative_stderr=representative_stderr,
#         trials=[asdict(t) for t in trials],
#     )
def summarize_trials(
    trials: List[RunTrial],
    warmup_trials: int,
) -> RunMetrics:
    representative_stdout = trials[0].stdout if trials else ""
    representative_stderr = trials[0].stderr if trials else ""

    timed_trials = trials[warmup_trials:] if warmup_trials < len(trials) else []
    successful_timed = [t for t in timed_trials if t.success and not t.timed_out]
    wall_times = [t.wall_time_seconds for t in successful_timed]

    crash_count = sum(1 for t in timed_trials if (not t.timed_out and t.return_code != 0))
    timeout_count = sum(1 for t in timed_trials if t.timed_out)

    return RunMetrics(
        run_success=(len(successful_timed) == len(timed_trials) and len(timed_trials) > 0),
        successful_runs=len(successful_timed),
        total_runs=len(trials),
        timed_trials_count=len(timed_trials),
        warmup_trials_count=min(warmup_trials, len(trials)),
        timeout_count=timeout_count,
        crash_count=crash_count,
        exit_codes=[t.return_code for t in timed_trials],
        wall_time_mean=(statistics.mean(wall_times) if wall_times else None),
        wall_time_median=(statistics.median(wall_times) if wall_times else None),
        wall_time_std=(statistics.stdev(wall_times) if len(wall_times) >= 2 else 0.0 if len(wall_times) == 1 else None),
        wall_time_min=(min(wall_times) if wall_times else None),
        wall_time_max=(max(wall_times) if wall_times else None),
        wall_time_trimmed_mean=trimmed_mean(wall_times, proportion_to_cut=0.1),
        runtime_coefficient_of_variation=coefficient_of_variation(wall_times),
        representative_stdout=representative_stdout,
        representative_stderr=representative_stderr,
        trials=[asdict(t) for t in trials],
    )


# def run_binary_repeated(
#     binary_path: Path,
#     program_args: List[str],
#     timeout_seconds: float,
#     trials: int,
# ) -> RunMetrics:
#     collected: List[RunTrial] = []
#     for _ in range(trials):
#         collected.append(
#             run_binary_once(
#                 binary_path=binary_path,
#                 program_args=program_args,
#                 timeout_seconds=timeout_seconds,
#             )
#         )
#     return summarize_trials(collected)
def run_binary_repeated(
    binary_path: Path,
    program_args: List[str],
    timeout_seconds: float,
    trials: int,
    warmup_trials: int,
    run_prefix: List[str]
) -> RunMetrics:
    collected: List[RunTrial] = []
    total_runs = warmup_trials + trials

    for _ in range(total_runs):
        collected.append(
            run_binary_once(
                binary_path=binary_path,
                program_args=program_args,
                timeout_seconds=timeout_seconds,
                run_prefix=run_prefix
            )
        )

    return summarize_trials(collected, warmup_trials=warmup_trials)


# ---------- Comparison helpers ----------


# def compare_outputs(
#     original_run: RunMetrics,
#     optimized_run: RunMetrics,
# ) -> OutputComparison:
#     original_exit = original_run.exit_codes[0] if original_run.exit_codes else None
#     optimized_exit = optimized_run.exit_codes[0] if optimized_run.exit_codes else None

#     stdout_match = (
#         original_run.representative_stdout == optimized_run.representative_stdout
#     )
#     stderr_match = (
#         original_run.representative_stderr == optimized_run.representative_stderr
#     )
#     exit_code_match = (original_exit == optimized_exit)

#     return OutputComparison(
#         exit_code_match=exit_code_match,
#         stdout_match=stdout_match,
#         stderr_match=stderr_match,
#         correctness_pass=(stdout_match and stderr_match and exit_code_match),
#     )
def normalize_stdout(text: str) -> str:
    """
    Remove non-deterministic lines (e.g., timing output) so correctness
    comparison only considers stable semantic outputs like CHECKSUM.
    """
    filtered = []
    for line in text.splitlines():
        line = line.strip()

        # Ignore timing lines
        if line.startswith("TIME_SEC="):
            continue

        filtered.append(line)

    return "\n".join(filtered)


def compare_outputs(
    original_run: RunMetrics,
    optimized_run: RunMetrics,
) -> OutputComparison:
    original_exit = original_run.exit_codes[0] if original_run.exit_codes else None
    optimized_exit = optimized_run.exit_codes[0] if optimized_run.exit_codes else None

    original_stdout = normalize_stdout(original_run.representative_stdout)
    optimized_stdout = normalize_stdout(optimized_run.representative_stdout)

    stdout_match = (original_stdout == optimized_stdout)

    stderr_match = (
        original_run.representative_stderr == optimized_run.representative_stderr
    )

    exit_code_match = (original_exit == optimized_exit)

    correctness_pass = (
        original_run.run_success
        and optimized_run.run_success
        and exit_code_match
        and stdout_match
        and stderr_match
    )

    return OutputComparison(
        exit_code_match=exit_code_match,
        stdout_match=stdout_match,
        stderr_match=stderr_match,
        correctness_pass=correctness_pass,
    )

def compute_performance_change(
    original_run: Optional[RunMetrics],
    optimized_run: Optional[RunMetrics],
) -> Dict[str, Optional[float | bool]]:
    if (
        original_run is None
        or optimized_run is None
        or original_run.wall_time_mean is None
        or optimized_run.wall_time_mean is None
        or original_run.wall_time_mean <= 0
        or optimized_run.wall_time_mean <= 0
    ):
        return {
            "speedup": None,
            "percent_improvement": None,
            "percent_change": None,
            "percent_slowdown": None,
            "is_regression": None,
        }

    orig = original_run.wall_time_mean
    opt = optimized_run.wall_time_mean

    speedup = orig / opt
    percent_improvement = ((orig - opt) / orig) * 100.0
    percent_change = ((opt - orig) / orig) * 100.0
    is_regression = opt > orig

    return {
        "speedup": speedup,
        "percent_improvement": percent_improvement,
        "percent_change": percent_change,
        "percent_slowdown": percent_change if is_regression else 0.0,
        "is_regression": is_regression,
    }


def compute_speedup(
    original_run: Optional[RunMetrics],
    optimized_run: Optional[RunMetrics],
) -> tuple[Optional[float], Optional[float]]:
    if (
        original_run is None
        or optimized_run is None
        or original_run.wall_time_mean is None
        or optimized_run.wall_time_mean is None
        or optimized_run.wall_time_mean <= 0
        or original_run.wall_time_mean <= 0
    ):
        return None, None

    speedup = original_run.wall_time_mean / optimized_run.wall_time_mean
    percent_improvement = (
        (original_run.wall_time_mean - optimized_run.wall_time_mean)
        / original_run.wall_time_mean
    ) * 100.0
    return speedup, percent_improvement


def compute_diff_metrics(
    original_source: Path,
    optimized_source: Path,
) -> DiffMetrics:
    original_lines = original_source.read_text(encoding="utf-8").splitlines()
    optimized_lines = optimized_source.read_text(encoding="utf-8").splitlines()

    matcher = difflib.SequenceMatcher(a=original_lines, b=optimized_lines)
    opcodes = matcher.get_opcodes()

    lines_added = 0
    lines_deleted = 0
    lines_changed = 0
    diff_hunk_count = 0

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            continue

        diff_hunk_count += 1

        if tag == "insert":
            lines_added += (j2 - j1)
        elif tag == "delete":
            lines_deleted += (i2 - i1)
        elif tag == "replace":
            deleted = i2 - i1
            added = j2 - j1
            lines_deleted += deleted
            lines_added += added
            lines_changed += max(deleted, added)

    total_original = max(len(original_lines), 1)
    percent_file_changed = (
        (lines_added + lines_deleted) / total_original
    ) * 100.0

    return DiffMetrics(
        lines_original=len(original_lines),
        lines_optimized=len(optimized_lines),
        lines_added=lines_added,
        lines_deleted=lines_deleted,
        lines_changed=lines_changed,
        diff_hunk_count=diff_hunk_count,
        percent_file_changed=percent_file_changed,
    )


# ---------- Main comparison logic ----------

def compare_versions(
    original_source: Path,
    optimized_source: Path,
    compiler: str,
    cflags: List[str],
    program_args: List[str],
    timeout_seconds: float,
    trials: int,
    warmup_trials: int,
    run_prefix: List[str]
) -> ComparisonReport:
    with tempfile.TemporaryDirectory(prefix="compare_c_versions_") as tmpdir:
        tmpdir_path = Path(tmpdir)

        original_binary = tmpdir_path / "original_bin"
        optimized_binary = tmpdir_path / "optimized_bin"

        original_build = build_c_file(
            source_path=original_source,
            output_binary=original_binary,
            compiler=compiler,
            cflags=cflags,
        )
        optimized_build = build_c_file(
            source_path=optimized_source,
            output_binary=optimized_binary,
            compiler=compiler,
            cflags=cflags,
        )

        original_run: Optional[RunMetrics] = None
        optimized_run: Optional[RunMetrics] = None
        output_comparison: Optional[OutputComparison] = None

        if original_build.compile_success:
            original_run = run_binary_repeated(
                binary_path=original_binary,
                program_args=program_args,
                timeout_seconds=timeout_seconds,
                trials=trials,
                warmup_trials=warmup_trials,
                run_prefix=run_prefix,
            )

        if optimized_build.compile_success:
            optimized_run = run_binary_repeated(
                binary_path=optimized_binary,
                program_args=program_args,
                timeout_seconds=timeout_seconds,
                trials=trials,
                warmup_trials=warmup_trials,
                run_prefix=run_prefix,
            )

        if original_run is not None and optimized_run is not None:
            output_comparison = compare_outputs(original_run, optimized_run)

        # speedup, percent_improvement = compute_speedup(original_run, optimized_run)
        perf = compute_performance_change(original_run, optimized_run)

        timing_summary = {
            "mean_speedup": None,
            "median_speedup": None,
            "trimmed_mean_speedup": None,
            "mean_percent_improvement": None,
            "median_percent_improvement": None,
            "trimmed_mean_percent_improvement": None,
            "likely_significant": None,
            "significance_reason": None,
        }

        if (
            original_run is not None
            and optimized_run is not None
            and original_run.trials
            and optimized_run.trials
        ):
            original_times = [
                t["wall_time_seconds"]
                for t in original_run.trials[original_run.warmup_trials_count:]
                if t["success"] and not t["timed_out"]
            ]
            optimized_times = [
                t["wall_time_seconds"]
                for t in optimized_run.trials[optimized_run.warmup_trials_count:]
                if t["success"] and not t["timed_out"]
            ]
            timing_summary = compare_float_lists_summary(original_times, optimized_times)

        diff_metrics = compute_diff_metrics(
            original_source=original_source,
            optimized_source=optimized_source,
        )

        return ComparisonReport(
            original_source=str(original_source),
            optimized_source=str(optimized_source),
            original_build=original_build,
            optimized_build=optimized_build,
            original_run=original_run,
            optimized_run=optimized_run,
            output_comparison=output_comparison,
            diff_metrics=diff_metrics,
            speedup=perf["speedup"],
            percent_improvement=perf["percent_improvement"],
            percent_change=perf["percent_change"],
            percent_slowdown=perf["percent_slowdown"],
            is_regression=perf["is_regression"],
            median_speedup=timing_summary["median_speedup"],
            median_percent_improvement=timing_summary["median_percent_improvement"],
            trimmed_mean_speedup=timing_summary["trimmed_mean_speedup"],
            trimmed_mean_percent_improvement=timing_summary["trimmed_mean_percent_improvement"],
            likely_significant=timing_summary["likely_significant"],
            significance_reason=timing_summary["significance_reason"],
        )
# ---------- CLI ----------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare original vs optimized C source files."
    )
    parser.add_argument(
        "--original",
        required=True,
        help="Path to the original C source file.",
    )
    parser.add_argument(
        "--optimized",
        required=True,
        help="Path to the optimized C source file.",
    )
    parser.add_argument(
        "--compiler",
        default="cc",
        help="C compiler to use, e.g. cc, gcc, clang, mpicc.",
    )
    parser.add_argument(
        "--cflags",
        default="-O2",
        help='Compiler flags as a single shell-style string, e.g. "-O3 -fopenmp".',
    )
    parser.add_argument(
        "--program-args",
        default="",
        help='Program arguments as a single shell-style string, e.g. "--size 1000000".',
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=20.0,
        help="Per-run timeout in seconds.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=20,
        help="Number of execution trials per binary.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write the full JSON report.",
    )

    parser.add_argument(
        "--warmup-trials",
        type=int,
        default=2,
        help="Number of warm-up runs to discard before timed trials.",
    )

    parser.add_argument(
        "--run-prefix",
        default="",
        help='Optional launcher prefix as a shell-style string, e.g. "mpirun -n 2".',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    original_source = Path(args.original).resolve()
    optimized_source = Path(args.optimized).resolve()

    if not original_source.exists():
        raise FileNotFoundError(f"Original source not found: {original_source}")
    if not optimized_source.exists():
        raise FileNotFoundError(f"Optimized source not found: {optimized_source}")

    # report = compare_versions(
    #     original_source=original_source,
    #     optimized_source=optimized_source,
    #     compiler=args.compiler,
    #     cflags=shlex.split(args.cflags),
    #     program_args=shlex.split(args.program_args),
    #     timeout_seconds=args.timeout_seconds,
    #     trials=args.trials,
    # )
    report = compare_versions(
        original_source=original_source,
        optimized_source=optimized_source,
        compiler=args.compiler,
        cflags=shlex.split(args.cflags),
        program_args=shlex.split(args.program_args),
        timeout_seconds=args.timeout_seconds,
        trials=args.trials,
        warmup_trials=args.warmup_trials,
        run_prefix=shlex.split(args.run_prefix),
    )

    report_dict = asdict(report)

    print(json.dumps(report_dict, indent=2))

    if args.output_json:
        output_path = Path(args.output_json).resolve()
        output_path.write_text(json.dumps(report_dict, indent=2), encoding="utf-8")
        print(f"\nWrote report to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()