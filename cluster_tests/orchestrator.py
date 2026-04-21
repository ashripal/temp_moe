from __future__ import annotations

import argparse
import filecmp
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any, TYPE_CHECKING

# --- REPO ROOT SETUP ---
REPO_ROOT = Path(os.environ.get("AUTOUP_REPO_ROOT", Path(__file__).resolve().parent.parent))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openai import OpenAI

if TYPE_CHECKING:
    from implementation.advisor import MoEAdvisor
    from implementation.generator import CodeGenerator
    from implementation.generator.generator_schema import GeneratorInput, GeneratorResult
    from implementation.llm import LLMMessage


# ---------------------------------------------------------------------------
# OPENAI BACKENDS
# ---------------------------------------------------------------------------

class OpenAIExpertLLM:
    """OpenAI-backed LLM client for the advisor/expert stage."""

    def __init__(self, model_name: str = "gpt-5", api_key: Optional[str] = None) -> None:
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def complete(self, messages: List[LLMMessage]) -> str:
        system_instruction = (
            "You are an HPC optimization expert assistant.\n"
            "Return exactly one valid JSON array and nothing else.\n"
            "The output must start with '[' and end with ']'.\n"
            "Do not include markdown fences, prose, or explanation.\n"
            "Each array element must be a JSON object representing one optimization candidate."
        )
        input_text = "\n\n".join(f"{m.role}: {m.content}" for m in messages)
        response = self.client.responses.create(
            model=self.model_name,
            instructions=system_instruction,
            input=input_text,
        )
        return response.output_text.strip()


class OpenAIGeneratorBackend:
    """OpenAI-backed generator backend."""

    def __init__(
        self,
        prompts_dir: Path,
        model_name: str = "gpt-5",
        api_key: Optional[str] = None,
    ) -> None:
        from jinja2 import Environment, FileSystemLoader, select_autoescape

        self.prompts_dir = Path(prompts_dir)
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.env = Environment(
            loader=FileSystemLoader(str(self.prompts_dir)),
            autoescape=select_autoescape(disabled_extensions=("jinja",)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def generate(self, generator_input: GeneratorInput) -> GeneratorResult:
        from implementation.generator.generator_utils import (
            build_failure_result,
            build_prompt_payload,
            extract_json_object,
            normalize_generator_response,
            sanitize_generated_text,
        )

        selected_candidate = generator_input.selected_candidate

        try:
            payload = build_prompt_payload(generator_input)
            template_name = (
                "generator_feedback_prompt.jinja"
                if payload["has_feedback"]
                else "generator_prompt.jinja"
            )
            prompt = self.env.get_template(template_name).render(**payload)
        except Exception as exc:
            return build_failure_result(
                original_code=generator_input.original_code,
                selected_candidate=selected_candidate,
                failure_reason=f"Prompt rendering failed: {exc}",
                used_feedback=bool(generator_input.resolved_feedback_text()),
            )

        try:
            response = self.client.responses.create(
                model=self.model_name,
                instructions=(
                    "You are an expert HPC code optimization generator. "
                    "Return exactly one valid JSON object and nothing else."
                ),
                input=prompt,
            )
            raw_text = response.output_text
        except Exception as exc:
            return build_failure_result(
                original_code=generator_input.original_code,
                selected_candidate=selected_candidate,
                failure_reason=f"Model inference failed: {exc}",
                used_feedback=bool(generator_input.resolved_feedback_text()),
            )

        cleaned = sanitize_generated_text(raw_text)
        parsed = extract_json_object(cleaned)

        if parsed is None:
            return build_failure_result(
                original_code=generator_input.original_code,
                selected_candidate=selected_candidate,
                failure_reason="Model output could not be parsed into the required JSON schema.",
                used_feedback=bool(generator_input.resolved_feedback_text()),
            )

        result = normalize_generator_response(
            response_dict=parsed,
            selected_candidate=selected_candidate,
            used_feedback=bool(generator_input.resolved_feedback_text()),
        )

        if not result.final_code.strip():
            return build_failure_result(
                original_code=generator_input.original_code,
                selected_candidate=selected_candidate,
                failure_reason="Model returned empty final_code.",
                used_feedback=bool(generator_input.resolved_feedback_text()),
            )

        return result


# ---------------------------------------------------------------------------
# BENCHMARK-SPECIFIC TELEMETRY
# ---------------------------------------------------------------------------
def discover_polybench_kernels() -> List[str]:
    """
    Discover all benchmark kernel source files under kernels/, excluding
    utility/support files, backup sources, and generated optimized copies.

    Returns paths relative to REPO_ROOT and without the .c suffix, matching
    the existing --kernels argument format.
    """
    kernels_root = REPO_ROOT / "kernels"
    discovered: List[str] = []

    for path in kernels_root.rglob("*.c"):
        rel = path.relative_to(REPO_ROOT)
        rel_posix = rel.as_posix()

        # Skip utility/support sources; they are not standalone benchmarks.
        if rel.parts[:2] == ("kernels", "utilities"):
            continue

        # Skip backup/original sources such as Nussinov.orig.c.
        if rel_posix.endswith(".orig.c"):
            continue

        # Skip generated optimized files if any were written into source dirs.
        if path.stem.endswith("_opt"):
            continue

        discovered.append(str(rel.with_suffix("")).replace("\\", "/"))

    return sorted(discovered)


@dataclass(frozen=True)
class ExternalBenchmarkSpec:
    benchmark_id: str
    family: str
    executable_rel: str
    cwd_rel: str
    kind: str
    default_args: tuple[str, ...] = ()
    exact_total_tasks: Optional[int] = None
    min_total_tasks: int = 1
    power_of_two_tasks: bool = False

    @property
    def executable_path(self) -> Path:
        return REPO_ROOT / self.executable_rel

    @property
    def cwd_path(self) -> Path:
        return REPO_ROOT / self.cwd_rel


EXTERNAL_BUILD_SCRIPT = REPO_ROOT / "benchmarks" / "external" / "build_selected.sh"

EXTERNAL_BENCHMARKS: Dict[str, ExternalBenchmarkSpec] = {
    "external/epcc-syncbench": ExternalBenchmarkSpec(
        benchmark_id="external/epcc-syncbench",
        family="epcc-openmp",
        executable_rel="benchmarks/external/epcc-openmp-microbenchmarks/openmpbench_C_v31/syncbench",
        cwd_rel="benchmarks/external/epcc-openmp-microbenchmarks/openmpbench_C_v31",
        kind="openmp",
        default_args=("--outer-repetitions", "1", "--test-time", "100"),
        exact_total_tasks=1,
    ),
    "external/epcc-schedbench": ExternalBenchmarkSpec(
        benchmark_id="external/epcc-schedbench",
        family="epcc-openmp",
        executable_rel="benchmarks/external/epcc-openmp-microbenchmarks/openmpbench_C_v31/schedbench",
        cwd_rel="benchmarks/external/epcc-openmp-microbenchmarks/openmpbench_C_v31",
        kind="openmp",
        default_args=("--outer-repetitions", "1", "--test-time", "100"),
        exact_total_tasks=1,
    ),
    "external/epcc-taskbench": ExternalBenchmarkSpec(
        benchmark_id="external/epcc-taskbench",
        family="epcc-openmp",
        executable_rel="benchmarks/external/epcc-openmp-microbenchmarks/openmpbench_C_v31/taskbench",
        cwd_rel="benchmarks/external/epcc-openmp-microbenchmarks/openmpbench_C_v31",
        kind="openmp",
        default_args=("--outer-repetitions", "1", "--test-time", "100"),
        exact_total_tasks=1,
    ),
    "external/epcc-arraybench-81": ExternalBenchmarkSpec(
        benchmark_id="external/epcc-arraybench-81",
        family="epcc-openmp",
        executable_rel="benchmarks/external/epcc-openmp-microbenchmarks/openmpbench_C_v31/arraybench_81",
        cwd_rel="benchmarks/external/epcc-openmp-microbenchmarks/openmpbench_C_v31",
        kind="openmp",
        default_args=("--outer-repetitions", "1", "--test-time", "100"),
        exact_total_tasks=1,
    ),
    "external/osu_latency": ExternalBenchmarkSpec(
        benchmark_id="external/osu_latency",
        family="osu-mpi",
        executable_rel="preflight_build/osu/mpi/pt2pt/osu_latency",
        cwd_rel="preflight_build/osu/mpi/pt2pt",
        kind="mpi",
        default_args=("-x", "10", "-i", "100"),
        exact_total_tasks=2,
    ),
    "external/osu_bw": ExternalBenchmarkSpec(
        benchmark_id="external/osu_bw",
        family="osu-mpi",
        executable_rel="preflight_build/osu/mpi/pt2pt/osu_bw",
        cwd_rel="preflight_build/osu/mpi/pt2pt",
        kind="mpi",
        default_args=("-x", "10", "-i", "100"),
        exact_total_tasks=2,
    ),
    "external/osu_bibw": ExternalBenchmarkSpec(
        benchmark_id="external/osu_bibw",
        family="osu-mpi",
        executable_rel="preflight_build/osu/mpi/pt2pt/osu_bibw",
        cwd_rel="preflight_build/osu/mpi/pt2pt",
        kind="mpi",
        default_args=("-x", "10", "-i", "100"),
        exact_total_tasks=2,
    ),
    "external/osu_allreduce": ExternalBenchmarkSpec(
        benchmark_id="external/osu_allreduce",
        family="osu-mpi",
        executable_rel="preflight_build/osu/mpi/collective/osu_allreduce",
        cwd_rel="preflight_build/osu/mpi/collective",
        kind="mpi",
        default_args=("-x", "10", "-i", "100"),
        min_total_tasks=2,
    ),
    "external/osu_alltoall": ExternalBenchmarkSpec(
        benchmark_id="external/osu_alltoall",
        family="osu-mpi",
        executable_rel="preflight_build/osu/mpi/collective/osu_alltoall",
        cwd_rel="preflight_build/osu/mpi/collective",
        kind="mpi",
        default_args=("-x", "10", "-i", "100"),
        min_total_tasks=2,
    ),
    "external/osu_barrier": ExternalBenchmarkSpec(
        benchmark_id="external/osu_barrier",
        family="osu-mpi",
        executable_rel="preflight_build/osu/mpi/collective/osu_barrier",
        cwd_rel="preflight_build/osu/mpi/collective",
        kind="mpi",
        default_args=("-x", "10", "-i", "100"),
        min_total_tasks=2,
    ),
    "external/npb-cg-s": ExternalBenchmarkSpec(
        benchmark_id="external/npb-cg-s",
        family="npb-mpi",
        executable_rel="benchmarks/external/npb/NPB3.4/NPB3.4-MPI/bin/cg.S.x",
        cwd_rel="benchmarks/external/npb/NPB3.4/NPB3.4-MPI",
        kind="mpi",
        min_total_tasks=1,
        power_of_two_tasks=True,
    ),
    "external/npb-mg-s": ExternalBenchmarkSpec(
        benchmark_id="external/npb-mg-s",
        family="npb-mpi",
        executable_rel="benchmarks/external/npb/NPB3.4/NPB3.4-MPI/bin/mg.S.x",
        cwd_rel="benchmarks/external/npb/NPB3.4/NPB3.4-MPI",
        kind="mpi",
        min_total_tasks=1,
        power_of_two_tasks=True,
    ),
    "external/npb-ft-s": ExternalBenchmarkSpec(
        benchmark_id="external/npb-ft-s",
        family="npb-mpi",
        executable_rel="benchmarks/external/npb/NPB3.4/NPB3.4-MPI/bin/ft.S.x",
        cwd_rel="benchmarks/external/npb/NPB3.4/NPB3.4-MPI",
        kind="mpi",
        min_total_tasks=1,
        power_of_two_tasks=True,
    ),
    "external/npb-bt-mz-s": ExternalBenchmarkSpec(
        benchmark_id="external/npb-bt-mz-s",
        family="npb-mz-hybrid",
        executable_rel="benchmarks/external/npb/NPB3.4-MZ/NPB3.4-MZ-MPI/bin/bt-mz.S.x",
        cwd_rel="benchmarks/external/npb/NPB3.4-MZ/NPB3.4-MZ-MPI",
        kind="hybrid",
        min_total_tasks=1,
    ),
    "external/npb-sp-mz-s": ExternalBenchmarkSpec(
        benchmark_id="external/npb-sp-mz-s",
        family="npb-mz-hybrid",
        executable_rel="benchmarks/external/npb/NPB3.4-MZ/NPB3.4-MZ-MPI/bin/sp-mz.S.x",
        cwd_rel="benchmarks/external/npb/NPB3.4-MZ/NPB3.4-MZ-MPI",
        kind="hybrid",
        min_total_tasks=1,
    ),
}

EXTERNAL_BENCHMARK_ALIASES: Dict[str, List[str]] = {
    "all-external": list(EXTERNAL_BENCHMARKS.keys()),
    "parallel-shortlist": list(EXTERNAL_BENCHMARKS.keys()),
}


def expand_requested_benchmarks(requested: List[str]) -> List[str]:
    expanded: List[str] = []
    seen: set[str] = set()
    external_lookup = {name.lower(): name for name in EXTERNAL_BENCHMARKS}

    for item in requested:
        token = item.lower()
        if token == "all":
            items = discover_polybench_kernels()
        elif token in EXTERNAL_BENCHMARK_ALIASES:
            items = EXTERNAL_BENCHMARK_ALIASES[token]
        elif token in external_lookup:
            items = [external_lookup[token]]
        else:
            items = [item]

        for resolved in items:
            if resolved not in seen:
                expanded.append(resolved)
                seen.add(resolved)

    return expanded


def is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def ensure_external_benchmarks_built(
    selected_specs: List[ExternalBenchmarkSpec],
    *,
    rebuild: bool,
) -> None:
    if not selected_specs:
        return

    if not EXTERNAL_BUILD_SCRIPT.exists():
        raise RuntimeError(
            f"External benchmark build script is missing: {EXTERNAL_BUILD_SCRIPT}"
        )

    needs_build = rebuild or any(not spec.executable_path.exists() for spec in selected_specs)
    if not needs_build:
        return

    print("Preparing managed external benchmarks via build_selected.sh ...")
    result = subprocess.run(
        [str(EXTERNAL_BUILD_SCRIPT)],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Failed to build external benchmarks.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )


def resolve_external_task_count(
    spec: ExternalBenchmarkSpec,
    *,
    requested_total_tasks: int,
) -> tuple[int, List[str]]:
    warnings: List[str] = []

    if spec.kind == "openmp":
        if requested_total_tasks != 1:
            warnings.append(
                f"{spec.benchmark_id} is OpenMP-only; forcing a single task instead of {requested_total_tasks}."
            )
        return 1, warnings

    total_tasks = requested_total_tasks

    if spec.exact_total_tasks is not None and total_tasks != spec.exact_total_tasks:
        warnings.append(
            f"{spec.benchmark_id} requires exactly {spec.exact_total_tasks} MPI task(s); "
            f"using that instead of the requested {total_tasks}."
        )
        total_tasks = spec.exact_total_tasks

    if total_tasks < spec.min_total_tasks:
        raise RuntimeError(
            f"{spec.benchmark_id} requires at least {spec.min_total_tasks} MPI task(s), "
            f"but the current layout only provides {total_tasks}."
        )

    if spec.power_of_two_tasks and not is_power_of_two(total_tasks):
        fallback = 1 << (total_tasks.bit_length() - 1)
        if fallback < spec.min_total_tasks:
            raise RuntimeError(
                f"{spec.benchmark_id} requires a power-of-two MPI task count, "
                f"and there is no valid fallback for {total_tasks} task(s)."
            )
        warnings.append(
            f"{spec.benchmark_id} works best with a power-of-two MPI task count; "
            f"falling back from {total_tasks} to {fallback}."
        )
        total_tasks = fallback

    return total_tasks, warnings


def build_external_command(
    spec: ExternalBenchmarkSpec,
    *,
    use_srun: bool,
    nodes: int,
    tasks_per_node: int,
    cpus_per_task: int,
) -> tuple[List[str], List[str], int]:
    requested_total_tasks = max(1, nodes * tasks_per_node)
    total_tasks, warnings = resolve_external_task_count(
        spec,
        requested_total_tasks=requested_total_tasks,
    )
    exe_and_args = [str(spec.executable_path), *spec.default_args]

    if spec.kind == "mpi" and cpus_per_task != 1:
        warnings.append(
            f"{spec.benchmark_id} is MPI-only; OMP_NUM_THREADS={cpus_per_task} will not change the benchmark itself."
        )

    if use_srun:
        if spec.kind == "openmp":
            cmd = [
                "srun",
                "--nodes=1",
                "--ntasks=1",
                f"--cpus-per-task={cpus_per_task}",
                *exe_and_args,
            ]
        else:
            actual_nodes = min(nodes, total_tasks)
            cmd = [
                "srun",
                f"--nodes={actual_nodes}",
                f"--ntasks={total_tasks}",
                f"--cpus-per-task={cpus_per_task}",
                *exe_and_args,
            ]
    else:
        if spec.kind == "openmp":
            cmd = exe_and_args
        else:
            cmd = ["mpirun", "-np", str(total_tasks), *exe_and_args]

    return cmd, warnings, total_tasks


def run_external_benchmark(
    *,
    spec: ExternalBenchmarkSpec,
    build_dir: Path,
    nodes: int,
    tasks_per_node: int,
    cpus_per_task: int,
    runs: int,
    use_srun: bool,
) -> Dict[str, Any]:
    if not spec.executable_path.exists():
        raise RuntimeError(
            f"Managed external benchmark binary is missing: {spec.executable_path}"
        )

    command, warnings, total_tasks = build_external_command(
        spec,
        use_srun=use_srun,
        nodes=nodes,
        tasks_per_node=tasks_per_node,
        cpus_per_task=cpus_per_task,
    )

    log_path = build_dir / f"{spec.benchmark_id.replace('/', '_').replace('-', '_')}_external.log"
    log_path.write_text("", encoding="utf-8")

    times: List[float] = []
    env = build_runtime_env(cpus_per_task)

    for i in range(runs):
        started = time.perf_counter()
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd=str(spec.cwd_path),
            env=env,
        )
        elapsed = time.perf_counter() - started

        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(f"=== Run {i + 1}/{runs} ===\n")
            fh.write(f"Command: {' '.join(command)}\n")
            fh.write(f"Wall time: {elapsed:.6f}\n")
            fh.write("STDOUT:\n")
            fh.write(result.stdout)
            if not result.stdout.endswith("\n"):
                fh.write("\n")
            fh.write("STDERR:\n")
            fh.write(result.stderr)
            if not result.stderr.endswith("\n"):
                fh.write("\n")
            fh.write("\n")

        if result.returncode != 0:
            raise RuntimeError(
                f"External benchmark failed on run {i + 1}: {spec.benchmark_id}\n"
                f"Command: {' '.join(command)}\n"
                f"STDOUT:\n{result.stdout}\n"
                f"STDERR:\n{result.stderr}"
            )

        times.append(elapsed)

    sorted_times = sorted(times)
    n = len(sorted_times)
    median = (
        (sorted_times[n // 2 - 1] + sorted_times[n // 2]) / 2.0
        if n % 2 == 0
        else sorted_times[n // 2]
    )

    return {
        "median": median,
        "mean": statistics.mean(times),
        "stdev": statistics.stdev(times) if len(times) > 1 else 0.0,
        "all_times": times,
        "verified": True,
        "output_file": log_path,
        "command": command,
        "warnings": warnings,
        "total_tasks": total_tasks,
    }


_KERNEL_PROFILES: Dict[str, Dict[str, Any]] = {
    "gemm": {
        "profiling_summary": (
            "Hotspots: dense matrix multiply, compute-bound, high arithmetic intensity, "
            "potential for loop tiling, vectorization, and cache blocking."
        ),
        "telemetry_struct": {
            "mpi_wait_pct": 0.0,
            "omp_barrier_pct": 5.0,
            "omp_imbalance_ratio": 1.1,
            "memory_bound_score": 0.3,
        },
    },
    "jacobi-2d": {
        "profiling_summary": (
            "Hotspots: 2D stencil sweep, memory bandwidth bound, low arithmetic intensity, "
            "potential for tiling, prefetching, and OpenMP parallelism."
        ),
        "telemetry_struct": {
            "mpi_wait_pct": 0.0,
            "omp_barrier_pct": 10.0,
            "omp_imbalance_ratio": 1.2,
            "memory_bound_score": 0.75,
        },
    },
}

_DEFAULT_PROFILE = {
    "profiling_summary": "Generic HPC kernel profile.",
    "telemetry_struct": {
        "mpi_wait_pct": 0.0,
        "omp_barrier_pct": 0.0,
        "omp_imbalance_ratio": 1.0,
        "memory_bound_score": 0.3,
    },
}


def benchmark_inputs(kernel_stem: str) -> Dict[str, Any]:
    profile = _KERNEL_PROFILES.get(kernel_stem, _DEFAULT_PROFILE)
    telemetry_struct = profile["telemetry_struct"]
    telemetry_summary = ", ".join(f"{k}={v}" for k, v in telemetry_struct.items())
    return {
        "profiling_summary": profile["profiling_summary"],
        "telemetry_summary": telemetry_summary,
        "telemetry_struct": telemetry_struct,
    }


# ---------------------------------------------------------------------------
# TRANSFORMATION ENTRY POINT
# ---------------------------------------------------------------------------


@dataclass
class TransformationOutcome:
    status: str
    accepted: bool
    opt_source_path: Path
    generator_result: GeneratorResult
    used_retry: bool = False
    rejection_reason: Optional[str] = None


def compile_validation_feedback(kernel_stem: str, compile_error: str) -> str:
    return (
        f"Compile validation failed for generated code for kernel '{kernel_stem}'.\n"
        "Repair only the compile issues conservatively. "
        "If the requested optimization is unsafe or relies on unsupported "
        "platform-specific APIs, return the original code unchanged.\n\n"
        f"{compile_error}"
    )


def validate_generated_kernel(
    *,
    source_path: Path,
    validation_binary: Path,
    dataset_size: str,
    kernel_include_dir: Path,
) -> None:
    compile_kernel(
        file_path=source_path,
        output_binary=validation_binary,
        dataset_size=dataset_size,
        extra_flags=["-DPOLYBENCH_DUMP_ARRAYS"],
        extra_include_dirs=[kernel_include_dir],
    )


def apply_catalog_transformations(
    source_code: str,
    kernel_stem: str,
    kernel_label: str,
    advisor: MoEAdvisor,
    generator: CodeGenerator,
    build_dir: Path,
    dataset_size: str,
    kernel_include_dir: Path,
) -> TransformationOutcome:
    from implementation.generator.generator_schema import GeneratorResult

    bench_meta = benchmark_inputs(kernel_stem)
    opt_source_path = build_dir / f"{kernel_label}_opt.c"
    validation_binary = build_dir / f"{kernel_label}_opt_compile_check.exe"

    advisor_result = advisor.run(
        code_snippets=source_code,
        profiling_summary=bench_meta["profiling_summary"],
        telemetry_summary=bench_meta["telemetry_summary"],
        telemetry_struct=bench_meta["telemetry_struct"],
    )

    generator_input = generator.from_advisor_result(
        advisor_result=advisor_result,
        original_code=source_code,
        profiling_summary=bench_meta["profiling_summary"],
        telemetry_summary=bench_meta["telemetry_summary"],
        telemetry_struct=bench_meta["telemetry_struct"],
        ast=None,
        flame_report=None,
    )

    if not advisor_result.final_ranked_candidates:
        no_candidate_result = GeneratorResult(
            analysis="Advisor did not select a safe source-level rewrite candidate.",
            selected_candidate_pattern="NO_CANDIDATE",
            selected_candidate_target="N/A",
            applied_changes_summary=(
                "No changes applied; no safe source-level rewrite candidate was selected."
            ),
            final_code=source_code,
            correctness_risks=[
                "No rewrite candidate selected; original code preserved."
            ],
            expected_metrics=[],
            compile_ready=False,
            used_feedback=False,
            generation_succeeded=False,
            failure_reason="No safe source-level rewrite candidate was selected.",
        )
        return TransformationOutcome(
            status="no_rewrite_candidate",
            accepted=False,
            opt_source_path=opt_source_path,
            generator_result=no_candidate_result,
            rejection_reason=(
                f"No safe source-level rewrite candidate was selected for {kernel_stem}.\n"
                f"Advisor reason: {advisor_result.routing.reason}"
            ),
        )

    generator_result = generator.generate(generator_input)

    if not generator_result.generation_succeeded or not generator_result.final_code.strip():
        return TransformationOutcome(
            status="generation_failed",
            accepted=False,
            opt_source_path=opt_source_path,
            generator_result=generator_result,
            rejection_reason=(
                f"Code generation failed for {kernel_stem}.\n"
                f"Result: {generator_result.to_dict()}"
            ),
        )

    opt_source_path.write_text(generator_result.final_code, encoding="utf-8")

    try:
        validate_generated_kernel(
            source_path=opt_source_path,
            validation_binary=validation_binary,
            dataset_size=dataset_size,
            kernel_include_dir=kernel_include_dir,
        )
        return TransformationOutcome(
            status="candidate_ready",
            accepted=True,
            opt_source_path=opt_source_path,
            generator_result=generator_result,
        )
    except RuntimeError as compile_error:
        retry_result = generator.retry_with_feedback(
            generator_input,
            evaluator_feedback=compile_validation_feedback(
                kernel_stem=kernel_stem,
                compile_error=str(compile_error),
            ),
        )

    if not retry_result.generation_succeeded or not retry_result.final_code.strip():
        return TransformationOutcome(
            status="compile_failed",
            accepted=False,
            opt_source_path=opt_source_path,
            generator_result=retry_result,
            used_retry=True,
            rejection_reason=(
                f"Retry generation failed for {kernel_stem} after compile validation.\n"
                f"Result: {retry_result.to_dict()}"
            ),
        )

    opt_source_path.write_text(retry_result.final_code, encoding="utf-8")

    try:
        validate_generated_kernel(
            source_path=opt_source_path,
            validation_binary=validation_binary,
            dataset_size=dataset_size,
            kernel_include_dir=kernel_include_dir,
        )
    except RuntimeError as retry_compile_error:
        return TransformationOutcome(
            status="compile_failed",
            accepted=False,
            opt_source_path=opt_source_path,
            generator_result=retry_result,
            used_retry=True,
            rejection_reason=(
                "Optimization rejected after compile validation retry.\n"
                f"{retry_compile_error}"
            ),
        )

    return TransformationOutcome(
        status="candidate_ready",
        accepted=True,
        opt_source_path=opt_source_path,
        generator_result=retry_result,
        used_retry=True,
    )


# ---------------------------------------------------------------------------
# COMPILATION
# ---------------------------------------------------------------------------

def compile_kernel(
    file_path: Path,
    output_binary: Path,
    dataset_size: str,
    extra_flags: Optional[List[str]] = None,
    extra_include_dirs: Optional[List[Path]] = None,
) -> None:
    """
    Compiles a PolyBench kernel using mpicc with OpenMP support.
    Raises RuntimeError if compilation fails.
    """
    flags = extra_flags or []
    extra_include_dirs = extra_include_dirs or []

    polybench_c = REPO_ROOT / "kernels" / "utilities" / "polybench.c"
    include_dirs = [REPO_ROOT / "kernels" / "utilities"] + extra_include_dirs

    cmd = [
        "mpicc",
        "-O3",
        "-march=native",
        "-fopenmp",
    ]

    for inc in include_dirs:
        cmd += ["-I", str(inc)]

    cmd += [
        str(polybench_c),
        str(file_path),
        f"-D{dataset_size}",
        "-lm",
        "-o", str(output_binary),
    ] + flags

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        raise RuntimeError(
            f"Compilation failed for {file_path}:\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )


# ---------------------------------------------------------------------------
# VERIFICATION
# ---------------------------------------------------------------------------

def verify_output(baseline_output: Path, candidate_output: Path) -> bool:
    """Byte-for-byte comparison of two PolyBench dump files."""
    return filecmp.cmp(str(baseline_output), str(candidate_output), shallow=False)


# ---------------------------------------------------------------------------
# RUNTIME HELPERS
# ---------------------------------------------------------------------------

def build_runtime_env(cpus_per_task: int) -> Dict[str, str]:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(cpus_per_task)
    env.setdefault("OMP_PROC_BIND", "close")
    env.setdefault("OMP_PLACES", "cores")
    return env


def run_command(
    exe_path: Path,
    *,
    use_srun: bool,
    nodes: int,
    tasks_per_node: int,
    cpus_per_task: int,
    capture_output: bool = True,
    cwd: Optional[Path] = None,
) -> subprocess.CompletedProcess:
    env = build_runtime_env(cpus_per_task)
    cwd = cwd or exe_path.parent

    if use_srun:
        # For standard PolyBench kernels, default should be one process.
        total_tasks = nodes * tasks_per_node
        cmd = [
            "srun",
            f"--nodes={nodes}",
            f"--ntasks={total_tasks}",
            f"--ntasks-per-node={tasks_per_node}",
            f"--cpus-per-task={cpus_per_task}",
            str(exe_path),
        ]
    else:
        cmd = [str(exe_path)]

    return subprocess.run(
        cmd,
        capture_output=capture_output,
        text=True,
        cwd=str(cwd),
        env=env,
    )


# ---------------------------------------------------------------------------
# BENCHMARKING
# ---------------------------------------------------------------------------

def run_benchmark(
    *,
    kernel_name: str,
    file_path: Path,
    build_dir: Path,
    dataset_size: str,
    nodes: int,
    tasks_per_node: int,
    cpus_per_task: int,
    runs: int,
    use_srun: bool,
    baseline_output_path: Optional[Path] = None,
    kernel_include_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Compiles and benchmarks a PolyBench kernel.

    Returns a dict with median, mean, stdev, all_times, verified, output_file.
    """
    verify_binary = build_dir / f"{kernel_name}_verify.exe"
    timing_binary = build_dir / f"{kernel_name}_timing.exe"
    output_file = build_dir / f"{kernel_name}_output.txt"

    # Step 1: Compile and run verification binary
    compile_kernel(
        file_path=file_path,
        output_binary=verify_binary,
        dataset_size=dataset_size,
        extra_flags=["-DPOLYBENCH_DUMP_ARRAYS"],
        extra_include_dirs=[kernel_include_dir] if kernel_include_dir is not None else None,
    )

    result = run_command(
        verify_binary,
        use_srun=use_srun,
        nodes=nodes,
        tasks_per_node=tasks_per_node,
        cpus_per_task=cpus_per_task,
        capture_output=True,
        cwd=build_dir,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Verification binary failed for {kernel_name}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

    # PolyBench dump arrays typically go to stderr
    output_file.write_text(result.stderr, encoding="utf-8")

    # Step 2: Correctness check against baseline
    if baseline_output_path is not None:
        verified = verify_output(baseline_output_path, output_file)
        if not verified:
            print(f"  WARNING: Output mismatch detected for {kernel_name}")
    else:
        verified = True

    # Step 3: Compile for timing
    compile_kernel(
        file_path=file_path,
        output_binary=timing_binary,
        dataset_size=dataset_size,
        extra_flags=["-DPOLYBENCH_TIME"],
        extra_include_dirs=[kernel_include_dir] if kernel_include_dir is not None else None,
    )

    # Step 4: Timed runs
    times: List[float] = []
    for i in range(runs):
        result = run_command(
            timing_binary,
            use_srun=use_srun,
            nodes=nodes,
            tasks_per_node=tasks_per_node,
            cpus_per_task=cpus_per_task,
            capture_output=True,
            cwd=build_dir,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Run failed on iteration {i + 1} for {kernel_name}\n"
                f"STDOUT:\n{result.stdout}\n"
                f"STDERR:\n{result.stderr}"
            )

        stdout_lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if not stdout_lines:
            raise RuntimeError(
                f"No timing output produced on run {i + 1} for {kernel_name}\n"
                f"STDERR:\n{result.stderr}"
            )

        raw_output = stdout_lines[-1]
        try:
            times.append(float(raw_output))
        except ValueError:
            raise RuntimeError(
                f"Could not parse timing output on run {i + 1} for {kernel_name}\n"
                f"Last line: '{raw_output}'\n"
                f"Full STDOUT:\n{result.stdout}\n"
                f"STDERR:\n{result.stderr}"
            )

    sorted_times = sorted(times)
    n = len(sorted_times)
    median = (
        (sorted_times[n // 2 - 1] + sorted_times[n // 2]) / 2.0
        if n % 2 == 0
        else sorted_times[n // 2]
    )

    return {
        "median": median,
        "mean": statistics.mean(times),
        "stdev": statistics.stdev(times) if len(times) > 1 else 0.0,
        "all_times": times,
        "verified": verified,
        "output_file": output_file,
    }


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="OptimizeHPC orchestrator for PolyBench and managed external benchmarks"
    )
    parser.add_argument("--nodes", type=int, default=1, help="Number of nodes to use at runtime")
    parser.add_argument("--tasks-per-node", type=int, default=1, help="MPI tasks per node")
    parser.add_argument("--cpus", type=int, default=40, help="OpenMP threads per task")
    parser.add_argument("--dataset", type=str, default="EXTRALARGE", help="PolyBench dataset size")
    parser.add_argument("--runs", type=int, default=5, help="Number of timed runs per kernel")
    parser.add_argument("--advisor-model", type=str, default="gpt-5", help="Advisor model")
    parser.add_argument("--generator-model", type=str, default="gpt-5", help="Generator model")
    parser.add_argument(
        "--kernels",
        nargs="*",
        default=["all"],
        help=(
            "PolyBench kernel paths relative to repo root, without .c suffix; "
            "'all' for PolyBench auto-discovery; "
            "'parallel-shortlist' or 'all-external' for the managed parallel benchmark bundle; "
            "or explicit external ids such as external/osu_latency."
        ),
    )
    parser.add_argument(
        "--build-dir",
        type=str,
        default="build/polybench_runs",
        help="Directory for generated sources, binaries, and outputs",
    )
    parser.add_argument(
        "--use-srun",
        action="store_true",
        help="Launch binaries via srun instead of direct execution",
    )
    parser.add_argument(
        "--rebuild-external",
        action="store_true",
        help="Force rebuilding the managed external benchmark bundle before execution",
    )
    args = parser.parse_args()

    dataset_size = f"{args.dataset}_DATASET"
    build_dir = (REPO_ROOT / args.build_dir).resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    args.kernels = expand_requested_benchmarks(args.kernels)

    print(
        f"Configuration: nodes={args.nodes} | tasks/node={args.tasks_per_node} | "
        f"cpus/task={args.cpus} | dataset={dataset_size} | runs={args.runs}"
    )
    print(f"Repo root: {REPO_ROOT}")
    print(f"Build dir: {build_dir}")
    print(f"Advisor model: {args.advisor_model} | Generator model: {args.generator_model}")
    print(f"Use srun: {args.use_srun}")
    if args.nodes > 1 and not args.use_srun:
        print("WARNING: nodes > 1 but --use-srun is disabled, so binaries will run as direct local processes.")
    if args.runs < 5:
        print("WARNING: runs < 5 may produce noisy speedup measurements.")

    external_specs = [
        EXTERNAL_BENCHMARKS[kernel_name]
        for kernel_name in args.kernels
        if kernel_name in EXTERNAL_BENCHMARKS
    ]
    if external_specs:
        print(f"Selected {len(external_specs)} managed external benchmark(s).")
        ensure_external_benchmarks_built(
            external_specs,
            rebuild=args.rebuild_external,
        )

    advisor: Optional[MoEAdvisor] = None
    generator: Optional[CodeGenerator] = None
    has_polybench_work = any(kernel_name not in EXTERNAL_BENCHMARKS for kernel_name in args.kernels)
    if has_polybench_work:
        from implementation.advisor import MoEAdvisor
        from implementation.generator import CodeGenerator
        from implementation.kb import KnowledgeBase

        kb = KnowledgeBase.from_csv(REPO_ROOT / "updated_optimization_catalog.csv")

        advisor = MoEAdvisor(
            llm=OpenAIExpertLLM(model_name=args.advisor_model),
            kb=kb,
            prompts_dir=REPO_ROOT / "implementation" / "prompts",
        )

        generator = CodeGenerator(
            backend=OpenAIGeneratorBackend(
                prompts_dir=REPO_ROOT / "implementation" / "generator" / "prompts",
                model_name=args.generator_model,
            )
        )

    summary_counts: Dict[str, int] = {
        "baseline_ok": 0,
        "optimized_ok": 0,
        "performance_regressed": 0,
        "correctness_failed": 0,
        "generation_failed": 0,
        "compile_failed": 0,
        "no_rewrite_candidate": 0,
        "external_only_ok": 0,
        "external_only_failed": 0,
        "missing_source": 0,
        "unexpected_error": 0,
    }

    for kernel_rel in args.kernels:
        try:
            if kernel_rel in EXTERNAL_BENCHMARKS:
                spec = EXTERNAL_BENCHMARKS[kernel_rel]
                kernel_label = kernel_rel.replace("/", "_")

                print(f"\n{'=' * 70}")
                print(f"Benchmark: {kernel_rel}")
                print(f"{'=' * 70}")
                print("  [external] Running managed external benchmark...")

                ext_results = run_external_benchmark(
                    spec=spec,
                    build_dir=build_dir,
                    nodes=args.nodes,
                    tasks_per_node=args.tasks_per_node,
                    cpus_per_task=args.cpus,
                    runs=args.runs,
                    use_srun=args.use_srun,
                )

                for warning in ext_results["warnings"]:
                    print(f"        WARNING: {warning}")

                print(f"        Family: {spec.family}")
                print(f"        Launch command: {' '.join(ext_results['command'])}")
                print(f"        Effective MPI tasks: {ext_results['total_tasks']}")
                print(
                    f"        Median wall time: {ext_results['median']:.6f}s "
                    f"(mean: {ext_results['mean']:.6f}s, stdev: {ext_results['stdev']:.6f}s)"
                )
                print(f"        Log file: {ext_results['output_file']}")
                print("        Status: external_only_ok")
                summary_counts["external_only_ok"] += 1
                continue

            kernel_base = (REPO_ROOT / kernel_rel).resolve()
            source_path = kernel_base.with_suffix(".c")
            kernel_stem = kernel_base.name
            kernel_label = kernel_rel.replace("/", "_")

            print(f"\n{'=' * 70}")
            print(f"Kernel: {kernel_rel}")
            print(f"{'=' * 70}")

            if not source_path.exists():
                print(f"  ERROR: Missing source file {source_path}")
                summary_counts["missing_source"] += 1
                continue

            # 1. Baseline
            print("  [1/3] Running baseline...")
            base_results = run_benchmark(
                kernel_name=f"{kernel_label}_base",
                file_path=source_path,
                build_dir=build_dir,
                dataset_size=dataset_size,
                nodes=args.nodes,
                tasks_per_node=args.tasks_per_node,
                cpus_per_task=args.cpus,
                runs=args.runs,
                use_srun=args.use_srun,
                baseline_output_path=None,
                kernel_include_dir=source_path.parent,
            )
            print(
                f"        Baseline median: {base_results['median']:.6f}s "
                f"(mean: {base_results['mean']:.6f}s, stdev: {base_results['stdev']:.6f}s)"
            )
            print("        Status: baseline_ok")
            summary_counts["baseline_ok"] += 1

            # 2. Optimize
            print("  [2/3] Applying catalog transformations via MoE pipeline...")
            source_code = source_path.read_text(encoding="utf-8")
            if advisor is None or generator is None:
                raise RuntimeError(
                    "The MoE pipeline was not initialized for source-based benchmarks."
                )
            transformation = apply_catalog_transformations(
                source_code=source_code,
                kernel_stem=kernel_stem,
                kernel_label=kernel_label,
                advisor=advisor,
                generator=generator,
                build_dir=build_dir,
                dataset_size=dataset_size,
                kernel_include_dir=source_path.parent,
            )

            if not transformation.accepted:
                print(f"        Status: {transformation.status}")
                print("        Optimization rejected after compile validation; skipping optimized run.")
                if transformation.rejection_reason:
                    print(transformation.rejection_reason)
                summary_counts[transformation.status] = summary_counts.get(transformation.status, 0) + 1
                continue

            opt_source_path = transformation.opt_source_path
            if transformation.used_retry:
                print("        Compile validation failed once; retry_with_feedback() produced a compilable rewrite.")

            # 3. Run optimized
            print("  [3/3] Running optimized...")
            opt_results = run_benchmark(
                kernel_name=f"{kernel_label}_opt",
                file_path=opt_source_path,
                build_dir=build_dir,
                dataset_size=dataset_size,
                nodes=args.nodes,
                tasks_per_node=args.tasks_per_node,
                cpus_per_task=args.cpus,
                runs=args.runs,
                use_srun=args.use_srun,
                baseline_output_path=base_results["output_file"],
                kernel_include_dir=source_path.parent,
            )
            print(
                f"        Optimized median: {opt_results['median']:.6f}s "
                f"(mean: {opt_results['mean']:.6f}s, stdev: {opt_results['stdev']:.6f}s)"
            )
            print(f"        Correctness verified: {opt_results['verified']}")

            if opt_results["verified"]:
                speedup = base_results["median"] / opt_results["median"]
                status = "optimized_ok" if speedup >= 1.0 else "performance_regressed"
                print(f"        Status: {status}")
                summary_counts[status] += 1
                print(f"\n  >>> Speedup: {speedup:.4f}x")
            else:
                print("        Status: correctness_failed")
                summary_counts["correctness_failed"] += 1
                print("\n  >>> Speedup not reported — output verification FAILED.")

        except Exception as exc:
            print(f"  ERROR while processing {kernel_rel}: {exc}")
            if kernel_rel in EXTERNAL_BENCHMARKS:
                summary_counts["external_only_failed"] += 1
            else:
                summary_counts["unexpected_error"] += 1
            continue

    print(f"\n{'=' * 70}")
    print("Run Summary")
    print(f"{'=' * 70}")
    print(f"Total kernels considered: {len(args.kernels)}")
    print(f"Baseline OK: {summary_counts['baseline_ok']}")
    print(f"Optimized OK: {summary_counts['optimized_ok']}")
    print(f"Performance regressed: {summary_counts['performance_regressed']}")
    print(f"Correctness failed: {summary_counts['correctness_failed']}")
    print(f"Generation failed: {summary_counts['generation_failed']}")
    print(f"Compile failed: {summary_counts['compile_failed']}")
    print(f"No rewrite candidate: {summary_counts['no_rewrite_candidate']}")
    print(f"External-only OK: {summary_counts['external_only_ok']}")
    print(f"External-only failed: {summary_counts['external_only_failed']}")
    print(f"Missing source: {summary_counts['missing_source']}")
    print(f"Unexpected errors: {summary_counts['unexpected_error']}")


if __name__ == "__main__":
    main()
