from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import sys
from dataclasses import asdict, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

# Ensure the repository root is on sys.path even when this script is run
# from inside the tests/ directory.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from implementation.advisor import MoEAdvisor
from implementation.generator import CodeGenerator
from implementation.generator.generator_schema import GeneratorInput, GeneratorResult
from implementation.kb import KnowledgeBase
from implementation.llm import LLMClient, LLMMessage

from compare_c_versions import compare_versions


class OpenAIExpertLLM(LLMClient):
    """
    OpenAI API-backed LLM client for the advisor/expert stage.

    It returns raw text because the expert layer already parses and validates
    the JSON candidate list.
    """

    def __init__(
        self,
        model_name: str = "gpt-5.2",
        api_key: Optional[str] = None,
    ) -> None:
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
    """
    OpenAI API-backed generator backend.

    This implements the same generate(...) contract that CodeGenerator expects.
    """

    def __init__(
        self,
        prompts_dir: str | Path,
        model_name: str = "gpt-5.2",
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
        from jinja2 import TemplateNotFound

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
            template = self.env.get_template(template_name)
            prompt = template.render(**payload)
        except TemplateNotFound as exc:
            return build_failure_result(
                original_code=generator_input.original_code,
                selected_candidate=selected_candidate,
                failure_reason=f"Prompt rendering failed: missing template {exc}",
                used_feedback=bool(generator_input.resolved_feedback_text()),
            )
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


def benchmark_inputs(benchmark_name: str) -> Dict[str, Any]:
    """
    Provide stable benchmark-specific profiling/telemetry inputs.

    These are simple seed inputs so the advisor routes each benchmark toward the
    intended expert family.
    """
    if benchmark_name == "mpi_pingpong":
        telemetry_struct = {
            "mpi_wait_pct": 40.0,
            "omp_barrier_pct": 0.0,
            "omp_imbalance_ratio": 1.0,
            "memory_bound_score": 0.1,
        }
        profiling_summary = (
            "Hotspots: MPI_Waitall 40%, message exchange dominates runtime, "
            "communication latency is the primary bottleneck."
        )
    elif benchmark_name == "omp_imbalance":
        telemetry_struct = {
            "mpi_wait_pct": 0.0,
            "omp_barrier_pct": 22.0,
            "omp_imbalance_ratio": 1.8,
            "memory_bound_score": 0.2,
        }
        profiling_summary = (
            "Hotspots: OpenMP barrier 22%, parallel region shows load imbalance, "
            "threads spend significant time waiting at synchronization points."
        )
    elif benchmark_name == "mem_saxpy":
        telemetry_struct = {
            "mpi_wait_pct": 0.0,
            "omp_barrier_pct": 0.0,
            "omp_imbalance_ratio": 1.0,
            "memory_bound_score": 0.85,
        }
        profiling_summary = (
            "Hotspots: SAXPY kernel is memory bandwidth bound, low arithmetic intensity, "
            "performance appears limited by cache/memory behavior."
        )
    else:
        telemetry_struct = {
            "mpi_wait_pct": 0.0,
            "omp_barrier_pct": 0.0,
            "omp_imbalance_ratio": 1.0,
            "memory_bound_score": 0.2,
        }
        profiling_summary = "Generic benchmark profile."

    telemetry_summary = ", ".join(f"{k}={v}" for k, v in telemetry_struct.items())

    return {
        "profiling_summary": profiling_summary,
        "telemetry_summary": telemetry_summary,
        "telemetry_struct": telemetry_struct,
    }


def benchmark_compare_config(benchmark_name: str) -> Dict[str, Any]:
    """
    Benchmark-specific compile/run config for deterministic evaluation.
    """
    if benchmark_name == "mpi_pingpong":
        return {
            "compiler": "mpicc",
            "cflags": ["-O3"],
            "program_args": [],
            "timeout_seconds": 20.0,
            "trials": 10,
            "warmup_trials": 2,
            "run_prefix": ["mpirun", "-np", "2"],
        }
    if benchmark_name == "omp_imbalance":
        return {
            "compiler": "cc",
            "cflags": ["-O3", "-fopenmp"],
            "program_args": [],
            "timeout_seconds": 20.0,
            "trials": 10,
            "warmup_trials": 2,
            "run_prefix": [],
        }
    return {
        "compiler": "cc",
        "cflags": ["-O3"],
        "program_args": [],
        "timeout_seconds": 20.0,
        "trials": 20,
        "warmup_trials": 2,
        "run_prefix": [],
    }


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def extract_numeric_signal(stdout_text: str) -> Optional[float]:
    """
    Prefer CHECKSUM=... but also allow a generic single numeric token fallback.
    """
    if not stdout_text:
        return None

    checksum_match = re.search(r"CHECKSUM=([-+]?\d+(?:\.\d+)?)", stdout_text)
    if checksum_match:
        return float(checksum_match.group(1))

    numeric_tokens = re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", stdout_text)
    if len(numeric_tokens) == 1:
        return float(numeric_tokens[0])

    return None


def relative_error(a: float, b: float) -> float:
    denom = max(abs(a), 1e-12)
    return abs(a - b) / denom


def compute_numeric_correctness(
    benchmark_name: str,
    comparison_report: Dict[str, Any],
) -> Dict[str, Any]:
    """
    HPC-aware correctness:
    - exact match passes
    - otherwise, if representative stdouts contain comparable numeric signals,
      accept within tolerance
    """
    output_cmp = comparison_report.get("output_comparison") or {}
    original_run = comparison_report.get("original_run") or {}
    optimized_run = comparison_report.get("optimized_run") or {}

    original_stdout = original_run.get("representative_stdout", "")
    optimized_stdout = optimized_run.get("representative_stdout", "")

    original_value = extract_numeric_signal(original_stdout)
    optimized_value = extract_numeric_signal(optimized_stdout)

    tolerance = 0.0
    if benchmark_name == "mem_saxpy":
        tolerance = 1e-5
    elif benchmark_name == "omp_imbalance":
        tolerance = 1e-7
    elif benchmark_name == "mpi_pingpong":
        tolerance = 0.0

    rel_err = None
    within_tolerance = False

    if original_value is not None and optimized_value is not None:
        rel_err = relative_error(original_value, optimized_value)
        within_tolerance = rel_err <= tolerance

    exact_pass = bool(output_cmp.get("stdout_match", False))
    final_pass = bool(
        output_cmp.get("exit_code_match", False)
        and output_cmp.get("stderr_match", False)
        and (exact_pass or within_tolerance)
    )

    return {
        "original_value": original_value,
        "optimized_value": optimized_value,
        "relative_error": rel_err,
        "tolerance": tolerance,
        "within_tolerance": within_tolerance,
        "exact_stdout_match": exact_pass,
        "final_pass": final_pass,
    }


def evaluate_with_rubric(
    benchmark_name: str,
    generator_result: GeneratorResult,
    comparison_report: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Deterministic rubric:
    - hard gates first
    - then weighted scoring
    - repair_once at most once
    """
    optimized_build = comparison_report.get("optimized_build") or {}
    optimized_run = comparison_report.get("optimized_run") or {}
    diff_metrics = comparison_report.get("diff_metrics") or {}
    output_cmp = comparison_report.get("output_comparison") or {}

    numeric = compute_numeric_correctness(benchmark_name, comparison_report)

    hard_gates = {
        "generation_succeeded": bool(generator_result.generation_succeeded),
        "compile_pass": bool(optimized_build.get("compile_success", False)),
        "run_pass": bool(optimized_run.get("run_success", False)),
        "timeout_free": int(optimized_run.get("timeout_count", 0)) == 0,
        "crash_free": int(optimized_run.get("crash_count", 0)) == 0,
        "exit_code_match": bool(output_cmp.get("exit_code_match", False)),
        "stderr_match": bool(output_cmp.get("stderr_match", False)),
    }

    correctness_safety = 0
    correctness_safety += 15 if hard_gates["compile_pass"] else 0

    if numeric["final_pass"]:
        correctness_safety += 15 if numeric["exact_stdout_match"] else 12
    else:
        correctness_safety += 0

    percent_file_changed = float(diff_metrics.get("percent_file_changed", 100.0))
    if percent_file_changed <= 35.0:
        correctness_safety += 10
    elif percent_file_changed <= 70.0:
        correctness_safety += 5

    optimization_faithfulness = 0
    if getattr(generator_result, "selected_candidate_pattern", None):
        optimization_faithfulness += 10
    if getattr(generator_result, "selected_candidate_target", None):
        optimization_faithfulness += 10
    if percent_file_changed <= 35.0:
        optimization_faithfulness += 5
    elif percent_file_changed <= 70.0:
        optimization_faithfulness += 2

    hpc_applicability = 0
    selected_pattern = (getattr(generator_result, "selected_candidate_pattern", "") or "").lower()
    expected_improvement = comparison_report.get("percent_improvement")

    if benchmark_name == "mem_saxpy":
        if any(k in selected_pattern for k in ["smaller data", "precision", "type", "memory", "locality"]):
            hpc_applicability += 10
        else:
            hpc_applicability += 5
    elif benchmark_name == "omp_imbalance":
        if any(k in selected_pattern for k in ["schedule", "load", "imbalance", "openmp", "parallel"]):
            hpc_applicability += 10
        else:
            hpc_applicability += 5
    elif benchmark_name == "mpi_pingpong":
        if any(k in selected_pattern for k in ["mpi", "communication", "async", "message"]):
            hpc_applicability += 10
        else:
            hpc_applicability += 5
    else:
        hpc_applicability += 5

    if hard_gates["run_pass"] and hard_gates["timeout_free"] and hard_gates["crash_free"]:
        hpc_applicability += 10

    if expected_improvement is not None:
        if expected_improvement >= 5.0:
            hpc_applicability += 5
        elif expected_improvement > 0.0:
            hpc_applicability += 2

    repairability = 0
    if hard_gates["compile_pass"] and not numeric["final_pass"]:
        repairability += 5
    if percent_file_changed <= 70.0:
        repairability += 5

    total = correctness_safety + optimization_faithfulness + hpc_applicability + repairability

    hard_fail = not all(
        [
            hard_gates["generation_succeeded"],
            hard_gates["compile_pass"],
            hard_gates["run_pass"],
            hard_gates["timeout_free"],
            hard_gates["crash_free"],
            hard_gates["exit_code_match"],
            hard_gates["stderr_match"],
        ]
    )

    if hard_fail:
        action = "repair_once"
        reason = "Hard gate failed but may be repairable."
    elif numeric["final_pass"] and total >= 75:
        action = "accept"
        reason = "Passed deterministic checks with strong rubric score."
    elif total >= 45:
        action = "repair_once"
        reason = "Localized or moderate-quality result; one repair attempt justified."
    else:
        action = "reject"
        reason = "Low rubric score or non-repairable result."

    return {
        "hard_gates": hard_gates,
        "numeric_correctness": numeric,
        "rubric": {
            "correctness_safety": correctness_safety,
            "optimization_faithfulness": optimization_faithfulness,
            "hpc_applicability": hpc_applicability,
            "repairability": repairability,
            "total": total,
        },
        "decision": {
            "action": action,
            "reason": reason,
        },
    }


def build_repair_feedback(
    benchmark_name: str,
    generator_result: GeneratorResult,
    comparison_report: Dict[str, Any],
    evaluation: Dict[str, Any],
) -> str:
    numeric = evaluation["numeric_correctness"]
    diff_metrics = comparison_report.get("diff_metrics") or {}
    optimized_build = comparison_report.get("optimized_build") or {}
    optimized_run = comparison_report.get("optimized_run") or {}
    selected_pattern = getattr(generator_result, "selected_candidate_pattern", None)
    selected_target = getattr(generator_result, "selected_candidate_target", None)

    lines: List[str] = []
    lines.append("Repair the generated code conservatively.")
    if selected_pattern:
        lines.append(f"Preserve the selected optimization pattern: {selected_pattern}.")
    if selected_target:
        lines.append(f"Keep the optimization scoped to the selected target: {selected_target}.")
    lines.append(f"Benchmark: {benchmark_name}.")

    if not optimized_build.get("compile_success", False):
        stderr = (optimized_build.get("stderr") or "").strip()
        lines.append("The optimized code did not compile.")
        if stderr:
            lines.append(f"Compiler stderr:\n{stderr}")

    if optimized_run and not optimized_run.get("run_success", False):
        lines.append("The optimized code did not run successfully.")
        rep_stderr = (optimized_run.get("representative_stderr") or "").strip()
        if rep_stderr:
            lines.append(f"Runtime stderr:\n{rep_stderr}")

    if numeric["relative_error"] is not None and not numeric["within_tolerance"]:
        lines.append(
            f"Output deviated beyond tolerance: relative_error={numeric['relative_error']}, "
            f"tolerance={numeric['tolerance']}."
        )

    if float(diff_metrics.get("percent_file_changed", 0.0)) > 70.0:
        lines.append(
            f"The patch changed too much of the file ({diff_metrics.get('percent_file_changed')}%). "
            "Make the edit more localized."
        )

    lines.append("Return a single corrected optimized file only in the required schema.")
    return "\n".join(lines)


def maybe_add_feedback(generator_input: GeneratorInput, feedback_text: str) -> Optional[GeneratorInput]:
    """
    Best-effort feedback injection without assuming a specific GeneratorInput field name.
    """
    if not is_dataclass(generator_input):
        return None

    valid_field_names = {f.name for f in fields(generator_input)}
    for candidate_name in ("feedback_text", "feedback", "review_feedback", "verifier_feedback"):
        if candidate_name in valid_field_names:
            return replace(generator_input, **{candidate_name: feedback_text})

    return None


def advisor_result_to_json(advisor_result: Any) -> Dict[str, Any]:
    return {
        "routing": asdict(advisor_result.routing),
        "expert_outputs": [eo.to_dict() for eo in advisor_result.expert_outputs],
        "final_ranked_candidates": advisor_result.final_ranked_candidates,
    }


def run_single_generation_attempt(
    benchmark_name: str,
    bench_dir: Path,
    output_dir: Path,
    generator: CodeGenerator,
    generator_input: GeneratorInput,
    compare_cfg: Dict[str, Any],
    file_stem: str,
) -> Dict[str, Any]:
    generator_result = generator.generate(generator_input)

    optimized_code_path = output_dir / f"{file_stem}.c"
    optimized_code_path.write_text(generator_result.final_code, encoding="utf-8")

    comparison_report = compare_versions(
        original_source=bench_dir / "main.c",
        optimized_source=optimized_code_path,
        compiler=compare_cfg["compiler"],
        cflags=compare_cfg["cflags"],
        program_args=compare_cfg["program_args"],
        timeout_seconds=compare_cfg["timeout_seconds"],
        trials=compare_cfg["trials"],
        warmup_trials=compare_cfg["warmup_trials"],
        run_prefix=compare_cfg["run_prefix"],
    )

    comparison_report_dict = asdict(comparison_report)
    evaluation = evaluate_with_rubric(
        benchmark_name=benchmark_name,
        generator_result=generator_result,
        comparison_report=comparison_report_dict,
    )

    return {
        "generator_result": generator_result,
        "comparison_report": comparison_report_dict,
        "evaluation": evaluation,
        "optimized_code_path": optimized_code_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate optimized code for local benchmarks using the MoE advisor + generator pipeline."
    )
    parser.add_argument(
        "--model",
        default="gpt-5.2",
        help="Compatibility fallback model name if stage-specific model flags are not set.",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="*",
        default=["mem_saxpy", "mpi_pingpong", "omp_imbalance"],
        help="Benchmark folder names under benchmarks/.",
    )
    parser.add_argument(
        "--output-dir",
        default="generated_optimizations",
        help="Directory where generated artifacts will be written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing optimized_main.c files if they already exist.",
    )
    parser.add_argument(
        "--provider",
        choices=["openai"],
        default="openai",
        help="LLM provider to use for advisor and generator.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Optional API key override. If omitted, OPENAI_API_KEY is used.",
    )
    parser.add_argument(
        "--advisor-model",
        default="gpt-5.2",
        help="Model for the advisor/expert stage.",
    )
    parser.add_argument(
        "--generator-model",
        default="gpt-5.2",
        help="Model for the generator stage.",
    )
    parser.add_argument(
        "--max-repair-attempts",
        type=int,
        default=1,
        choices=[0, 1],
        help="Bounded repair loop. Use 0 to disable repair, 1 for at most one repair.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    benchmarks_root = repo_root / "benchmarks"
    prompts_dir = repo_root / "implementation" / "prompts"
    generator_prompts_dir = repo_root / "implementation" / "generator" / "prompts"
    catalog_path = repo_root / "updated_optimization_catalog.csv"
    output_root = repo_root / args.output_dir

    advisor_model = args.advisor_model or args.model
    generator_model = args.generator_model or args.model

    kb = KnowledgeBase.from_csv(catalog_path)
    print(f"Using OpenAI models: advisor={advisor_model}, generator={generator_model}")

    advisor_llm = OpenAIExpertLLM(
        model_name=advisor_model,
        api_key=args.api_key,
    )
    advisor = MoEAdvisor(
        llm=advisor_llm,
        kb=kb,
        prompts_dir=prompts_dir,
    )

    generator_backend = OpenAIGeneratorBackend(
        prompts_dir=generator_prompts_dir,
        model_name=generator_model,
        api_key=args.api_key,
    )
    generator = CodeGenerator(backend=generator_backend)

    output_root.mkdir(parents=True, exist_ok=True)

    for bench_name in args.benchmarks:
        try:
            bench_dir = benchmarks_root / bench_name
            code_path = bench_dir / "main.c"

            if not code_path.exists():
                print(f"[skip] {bench_name}: missing {code_path}")
                continue

            bench_output_dir = output_root / bench_name
            bench_output_dir.mkdir(parents=True, exist_ok=True)

            optimized_code_path = bench_output_dir / "optimized_main.c"
            if optimized_code_path.exists() and not args.overwrite:
                print(f"[skip] {bench_name}: {optimized_code_path} exists (use --overwrite)")
                continue

            code_text = code_path.read_text(encoding="utf-8")
            bench_meta = benchmark_inputs(bench_name)
            compare_cfg = benchmark_compare_config(bench_name)

            print(f"[run] {bench_name}")

            advisor_result = advisor.run(
                code_snippets=code_text,
                profiling_summary=bench_meta["profiling_summary"],
                telemetry_summary=bench_meta["telemetry_summary"],
                telemetry_struct=bench_meta["telemetry_struct"],
            )

            generator_input = generator.from_advisor_result(
                advisor_result=advisor_result,
                original_code=code_text,
                profiling_summary=bench_meta["profiling_summary"],
                telemetry_summary=bench_meta["telemetry_summary"],
                telemetry_struct=bench_meta["telemetry_struct"],
                ast=None,
                flame_report=None,
            )

            first_attempt = run_single_generation_attempt(
                benchmark_name=bench_name,
                bench_dir=bench_dir,
                output_dir=bench_output_dir,
                generator=generator,
                generator_input=generator_input,
                compare_cfg=compare_cfg,
                file_stem="optimized_main_initial",
            )

            final_attempt = first_attempt
            repair_attempt = None

            first_decision = first_attempt["evaluation"]["decision"]["action"]

            if first_decision == "repair_once" and args.max_repair_attempts == 1:
                feedback_text = build_repair_feedback(
                    benchmark_name=bench_name,
                    generator_result=first_attempt["generator_result"],
                    comparison_report=first_attempt["comparison_report"],
                    evaluation=first_attempt["evaluation"],
                )

                repaired_input = maybe_add_feedback(generator_input, feedback_text)

                if repaired_input is not None:
                    repair_attempt = run_single_generation_attempt(
                        benchmark_name=bench_name,
                        bench_dir=bench_dir,
                        output_dir=bench_output_dir,
                        generator=generator,
                        generator_input=repaired_input,
                        compare_cfg=compare_cfg,
                        file_stem="optimized_main_repair",
                    )

                    repaired_action = repair_attempt["evaluation"]["decision"]["action"]
                    repaired_total = repair_attempt["evaluation"]["rubric"]["total"]
                    first_total = first_attempt["evaluation"]["rubric"]["total"]

                    if repaired_action == "accept" or repaired_total >= first_total:
                        final_attempt = repair_attempt

                    write_json(bench_output_dir / "repair_input_feedback.json", {"feedback_text": feedback_text})
                    write_json(bench_output_dir / "generator_result_repair.json", repair_attempt["generator_result"].to_dict())
                    write_json(bench_output_dir / "tester_result_repair.json", repair_attempt["comparison_report"])
                    write_json(bench_output_dir / "rubric_result_repair.json", repair_attempt["evaluation"])
                else:
                    write_json(
                        bench_output_dir / "repair_input_feedback.json",
                        {
                            "feedback_text": feedback_text,
                            "note": "Repair was requested by rubric, but no recognized feedback field was found on GeneratorInput.",
                        },
                    )

            final_generator_result: GeneratorResult = final_attempt["generator_result"]
            final_code_path: Path = final_attempt["optimized_code_path"]

            optimized_code_path.write_text(final_code_path.read_text(encoding="utf-8"), encoding="utf-8")

            write_json(bench_output_dir / "advisor_result.json", advisor_result_to_json(advisor_result))
            write_json(bench_output_dir / "generator_input.json", generator_input.to_dict())
            write_json(bench_output_dir / "generator_result.json", final_generator_result.to_dict())
            write_json(bench_output_dir / "tester_result.json", final_attempt["comparison_report"])
            write_json(bench_output_dir / "rubric_result.json", final_attempt["evaluation"])

            print(
                f"[done] {bench_name}: "
                f"generation_succeeded={final_generator_result.generation_succeeded} "
                f"decision={final_attempt['evaluation']['decision']['action']} "
                f"score={final_attempt['evaluation']['rubric']['total']} "
                f"output={optimized_code_path}"
            )

        except Exception as exc:
            print(f"[error] {bench_name}: {exc}")
            continue


if __name__ == "__main__":
    main()