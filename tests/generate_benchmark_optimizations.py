from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
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
from implementation.generator.generator_schema import (
    EvaluationFeedback,
    GeneratorInput,
    GeneratorResult,
)
from implementation.kb import KnowledgeBase
from implementation.llm import LLMClient, LLMMessage

from compare_c_versions import compare_versions
from rubric_evaluator import evaluate_with_rubric


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
                    "Return exactly one valid JSON object and nothing else. "
                    "Do not output markdown fences, prose outside the JSON object, or diffs. "
                    "The JSON must contain complete source code in the final_code field."
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
        libomp_prefix = Path(os.popen("brew --prefix libomp 2>/dev/null").read().strip())
        cflags = ["-O3", "-Xpreprocessor", "-fopenmp"]
        if str(libomp_prefix):
            cflags += [
                f"-I{libomp_prefix / 'include'}",
                f"-L{libomp_prefix / 'lib'}",
                "-lomp",
            ]
        else:
            cflags += ["-fopenmp"]

        return {
            "compiler": "cc",
            "cflags": cflags,
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


def build_repair_feedback(
    benchmark_name: str,
    generator_result: GeneratorResult,
    comparison_report: Dict[str, Any],
    evaluation: Dict[str, Any],
) -> str:
    numeric = evaluation.get("numeric_correctness") or {}
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

    if numeric.get("relative_error") is not None and not numeric.get("within_tolerance", False):
        lines.append(
            f"Output deviated beyond tolerance: relative_error={numeric.get('relative_error')}, "
            f"tolerance={numeric.get('numeric_tolerance')}."
        )

    if float(diff_metrics.get("percent_file_changed", 0.0)) > 70.0:
        lines.append(
            f"The patch changed too much of the file ({diff_metrics.get('percent_file_changed')}%). "
            "Make the edit more localized."
        )

    lines.append("Return a single corrected optimized file only in the required schema.")
    return "\n".join(lines)


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
        generator_result=generator_result,
        comparison_report=comparison_report_dict,
    )

    return {
        "generator_result": generator_result,
        "comparison_report": comparison_report_dict,
        "evaluation": evaluation,
        "optimized_code_path": optimized_code_path,
    }


def should_promote_attempt(
    current_best: Dict[str, Any],
    candidate_attempt: Dict[str, Any],
) -> bool:
    current_eval = current_best["evaluation"]
    candidate_eval = candidate_attempt["evaluation"]

    current_action = current_eval["decision"]["action"]
    candidate_action = candidate_eval["decision"]["action"]

    priority = {
        "accept": 5,
        "accept_with_warning": 4,
        "repair_once": 3,
        "reconsider_advisor": 2,
        "reject": 1,
        "environment_failure": 0,
    }

    current_score = int(current_eval.get("rubric", {}).get("total", 0))
    candidate_score = int(candidate_eval.get("rubric", {}).get("total", 0))

    if priority.get(candidate_action, -1) > priority.get(current_action, -1):
        return True
    if priority.get(candidate_action, -1) < priority.get(current_action, -1):
        return False

    return candidate_score >= current_score


def choose_next_candidate_input(
    generator: CodeGenerator,
    advisor_result: Any,
    original_code: str,
    bench_meta: Dict[str, Any],
    current_input: GeneratorInput,
) -> Optional[GeneratorInput]:
    ranked = advisor_result.final_ranked_candidates or []
    current_selected = current_input.selected_candidate or {}
    current_pattern = str(current_selected.get("pattern", ""))
    current_target = str(current_selected.get("target", ""))

    for idx, candidate in enumerate(ranked):
        if not isinstance(candidate, dict):
            continue
        if (
            str(candidate.get("pattern", "")) == current_pattern
            and str(candidate.get("target", "")) == current_target
        ):
            continue

        return generator.from_advisor_result(
            advisor_result=advisor_result,
            original_code=original_code,
            profiling_summary=bench_meta["profiling_summary"],
            telemetry_summary=bench_meta["telemetry_summary"],
            telemetry_struct=bench_meta["telemetry_struct"],
            ast=None,
            flame_report=None,
            selected_candidate_index=idx,
        )

    return None


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
        help="Bounded generator repair loop. Use 0 to disable repair, 1 for at most one repair.",
    )
    parser.add_argument(
        "--max-advisor-reconsiderations",
        type=int,
        default=1,
        choices=[0, 1],
        help="Bounded advisor reconsideration loop. Use 0 to disable, 1 to try at most one alternate ranked candidate.",
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
            advisor_reconsider_attempt = None

            first_decision = first_attempt["evaluation"]["decision"]["action"]

            if first_decision == "repair_once" and args.max_repair_attempts == 1:
                feedback_text = build_repair_feedback(
                    benchmark_name=bench_name,
                    generator_result=first_attempt["generator_result"],
                    comparison_report=first_attempt["comparison_report"],
                    evaluation=first_attempt["evaluation"],
                )
                structured_feedback = EvaluationFeedback.from_rubric_result(first_attempt["evaluation"])

                repair_attempt_result = generator.retry_with_feedback(
                    previous_input=generator_input,
                    evaluator_feedback=feedback_text,
                    evaluation_feedback=structured_feedback,
                )

                repair_attempt = {
                    "generator_result": repair_attempt_result,
                    "optimized_code_path": bench_output_dir / "optimized_main_repair.c",
                }
                repair_attempt["optimized_code_path"].write_text(
                    repair_attempt_result.final_code,
                    encoding="utf-8",
                )

                comparison_report = compare_versions(
                    original_source=bench_dir / "main.c",
                    optimized_source=repair_attempt["optimized_code_path"],
                    compiler=compare_cfg["compiler"],
                    cflags=compare_cfg["cflags"],
                    program_args=compare_cfg["program_args"],
                    timeout_seconds=compare_cfg["timeout_seconds"],
                    trials=compare_cfg["trials"],
                    warmup_trials=compare_cfg["warmup_trials"],
                    run_prefix=compare_cfg["run_prefix"],
                )
                repair_attempt["comparison_report"] = asdict(comparison_report)
                repair_attempt["evaluation"] = evaluate_with_rubric(
                    generator_result=repair_attempt_result,
                    comparison_report=repair_attempt["comparison_report"],
                )

                if should_promote_attempt(final_attempt, repair_attempt):
                    final_attempt = repair_attempt

                write_json(bench_output_dir / "repair_input_feedback.json", {"feedback_text": feedback_text})
                write_json(bench_output_dir / "generator_result_repair.json", repair_attempt_result.to_dict())
                write_json(bench_output_dir / "tester_result_repair.json", repair_attempt["comparison_report"])
                write_json(bench_output_dir / "rubric_result_repair.json", repair_attempt["evaluation"])

            current_decision = final_attempt["evaluation"]["decision"]["action"]

            if current_decision == "reconsider_advisor" and args.max_advisor_reconsiderations == 1:
                alternate_input = choose_next_candidate_input(
                    generator=generator,
                    advisor_result=advisor_result,
                    original_code=code_text,
                    bench_meta=bench_meta,
                    current_input=generator_input,
                )

                if alternate_input is not None:
                    advisor_reconsider_attempt = run_single_generation_attempt(
                        benchmark_name=bench_name,
                        bench_dir=bench_dir,
                        output_dir=bench_output_dir,
                        generator=generator,
                        generator_input=alternate_input,
                        compare_cfg=compare_cfg,
                        file_stem="optimized_main_advisor_retry",
                    )

                    if should_promote_attempt(final_attempt, advisor_reconsider_attempt):
                        final_attempt = advisor_reconsider_attempt

                    write_json(bench_output_dir / "generator_input_advisor_retry.json", alternate_input.to_dict())
                    write_json(
                        bench_output_dir / "generator_result_advisor_retry.json",
                        advisor_reconsider_attempt["generator_result"].to_dict(),
                    )
                    write_json(
                        bench_output_dir / "tester_result_advisor_retry.json",
                        advisor_reconsider_attempt["comparison_report"],
                    )
                    write_json(
                        bench_output_dir / "rubric_result_advisor_retry.json",
                        advisor_reconsider_attempt["evaluation"],
                    )
                else:
                    write_json(
                        bench_output_dir / "advisor_reconsideration_note.json",
                        {"note": "Evaluator requested advisor reconsideration, but no alternate ranked candidate was available."},
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