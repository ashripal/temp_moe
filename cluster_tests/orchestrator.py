from __future__ import annotations

import argparse
import filecmp
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

# --- REPO ROOT SETUP ---
REPO_ROOT = Path(os.environ.get("AUTOUP_REPO_ROOT", Path(__file__).resolve().parent.parent))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# --- MoE PIPELINE IMPORTS ---
from implementation.advisor import MoEAdvisor
from implementation.generator import CodeGenerator
from implementation.kb import KnowledgeBase
from implementation.llm import LLMClient, LLMMessage
from implementation.generator.generator_schema import GeneratorInput, GeneratorResult
from openai import OpenAI


# ---------------------------------------------------------------------------
# OPENAI BACKENDS
# ---------------------------------------------------------------------------

class OpenAIExpertLLM(LLMClient):
    """OpenAI-backed LLM client for the advisor/expert stage."""

    def __init__(self, model_name: str = "gpt-4o", api_key: Optional[str] = None) -> None:
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
        model_name: str = "gpt-4o",
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

def apply_catalog_transformations(
    source_code: str,
    kernel_stem: str,
    advisor: MoEAdvisor,
    generator: CodeGenerator,
) -> str:
    bench_meta = benchmark_inputs(kernel_stem)

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

    generator_result = generator.generate(generator_input)

    if not generator_result.generation_succeeded or not generator_result.final_code.strip():
        raise RuntimeError(
            f"Code generation failed for {kernel_stem}.\n"
            f"Result: {generator_result.to_dict()}"
        )

    return generator_result.final_code


# ---------------------------------------------------------------------------
# COMPILATION
# ---------------------------------------------------------------------------

def compile_kernel(
    file_path: Path,
    output_binary: Path,
    dataset_size: str,
    extra_flags: Optional[List[str]] = None,
) -> None:
    """
    Compiles a PolyBench kernel using mpicc with OpenMP support.
    Raises RuntimeError if compilation fails.
    """
    flags = extra_flags or []
    polybench_c = REPO_ROOT / "kernels" / "utilities" / "polybench.c"
    include_dir = REPO_ROOT / "kernels" / "utilities"

    cmd = [
        "mpicc",
        "-O3",
        "-march=native",
        "-fopenmp",
        "-I", str(include_dir),
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
    parser = argparse.ArgumentParser(description="OptimizeHPC PolyBench Orchestrator")
    parser.add_argument("--nodes", type=int, default=1, help="Number of nodes to use at runtime")
    parser.add_argument("--tasks-per-node", type=int, default=1, help="MPI tasks per node")
    parser.add_argument("--cpus", type=int, default=40, help="OpenMP threads per task")
    parser.add_argument("--dataset", type=str, default="EXTRALARGE", help="PolyBench dataset size")
    parser.add_argument("--runs", type=int, default=5, help="Number of timed runs per kernel")
    parser.add_argument("--advisor-model", type=str, default="gpt-4o", help="Advisor model")
    parser.add_argument("--generator-model", type=str, default="gpt-4o", help="Generator model")
    parser.add_argument(
        "--kernels",
        nargs="*",
        default=[
            "kernels/linear-algebra/blas/gemm/gemm",
            "kernels/stencils/jacobi-2d/jacobi-2d",
        ],
        help="Kernel paths relative to repo root, without .c suffix",
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
    args = parser.parse_args()

    dataset_size = f"{args.dataset}_DATASET"
    build_dir = (REPO_ROOT / args.build_dir).resolve()
    build_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Configuration: nodes={args.nodes} | tasks/node={args.tasks_per_node} | "
        f"cpus/task={args.cpus} | dataset={dataset_size} | runs={args.runs}"
    )
    print(f"Repo root: {REPO_ROOT}")
    print(f"Build dir: {build_dir}")
    print(f"Advisor model: {args.advisor_model} | Generator model: {args.generator_model}")
    print(f"Use srun: {args.use_srun}")

    # Initialize MoE pipeline once
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

    for kernel_rel in args.kernels:
        kernel_base = (REPO_ROOT / kernel_rel).resolve()
        source_path = kernel_base.with_suffix(".c")
        kernel_stem = kernel_base.name
        kernel_label = kernel_rel.replace("/", "_")

        print(f"\n{'=' * 70}")
        print(f"Kernel: {kernel_rel}")
        print(f"{'=' * 70}")

        if not source_path.exists():
            print(f"  ERROR: Missing source file {source_path}")
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
        )
        print(
            f"        Baseline median: {base_results['median']:.6f}s "
            f"(mean: {base_results['mean']:.6f}s, stdev: {base_results['stdev']:.6f}s)"
        )

        # 2. Optimize
        print("  [2/3] Applying catalog transformations via MoE pipeline...")
        source_code = source_path.read_text(encoding="utf-8")
        optimized_code = apply_catalog_transformations(
            source_code=source_code,
            kernel_stem=kernel_stem,
            advisor=advisor,
            generator=generator,
        )

        opt_source_path = build_dir / f"{kernel_label}_opt.c"
        opt_source_path.write_text(optimized_code, encoding="utf-8")

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
        )
        print(
            f"        Optimized median: {opt_results['median']:.6f}s "
            f"(mean: {opt_results['mean']:.6f}s, stdev: {opt_results['stdev']:.6f}s)"
        )
        print(f"        Correctness verified: {opt_results['verified']}")

        if opt_results["verified"]:
            speedup = base_results["median"] / opt_results["median"]
            print(f"\n  >>> Speedup: {speedup:.4f}x")
        else:
            print("\n  >>> Speedup not reported — output verification FAILED.")


if __name__ == "__main__":
    main()