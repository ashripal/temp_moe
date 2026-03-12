from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from implementation.generator.generator_schema import (
    EvaluationFeedback,
    GeneratorInput,
)
from implementation.generator.generator_utils import build_prompt_payload


# Build one valid advisor candidate that matches the expected schema.
# This is reused across tests so each test only focuses on prompt rendering behavior.
def _valid_candidate() -> dict:
    return {
        "pattern": "MPI_WAIT_REDUCTION",
        "target": "main timestep loop",
        "rationale": "High MPI_Waitall time suggests excessive synchronization overhead.",
        "action_sketch": "Fuse communication phases and reduce unnecessary waits.",
        "preconditions": ["Communication pattern is safe to reorder locally."],
        "parameters_to_sweep": ["batch_size", "message_grouping"],
        "correctness_checks": ["Results match baseline output."],
        "performance_metrics": ["mpi_wait_pct", "runtime_seconds"],
        "risk_level": "medium",
        "rollback_criteria": "Rollback if runtime increases or outputs diverge.",
    }


# Helper to locate the generator prompt directory from the repository root.
# Since this test file lives in tests/, we walk up one level and then into implementation/.
def _prompt_env() -> Environment:
    repo_root = Path(__file__).resolve().parent.parent
    prompts_dir = repo_root / "implementation" / "generator" / "prompts"
    return Environment(loader=FileSystemLoader(str(prompts_dir)))


# Helper to build a minimal valid GeneratorInput.
# Individual tests can override fields as needed.
def _generator_input(
    *,
    evaluator_feedback: str | None = None,
    evaluation_feedback: EvaluationFeedback | None = None,
    ast: str | None = "FunctionDecl(main)",
    flame_report: str | None = "main 80%",
) -> GeneratorInput:
    candidate = _valid_candidate()
    return GeneratorInput(
        original_code="int main() { return 0; }",
        profiling_summary="MPI_Waitall dominates runtime.",
        telemetry_summary="mpi_wait_pct=42",
        telemetry_struct={"mpi_wait_pct": 42.0},
        ranked_candidates=[candidate],
        selected_candidate=candidate,
        ast=ast,
        flame_report=flame_report,
        evaluator_feedback=evaluator_feedback,
        evaluation_feedback=evaluation_feedback,
    )


def test_generator_prompt_renders_selected_candidate_fields() -> None:
    # Render the first-pass prompt and verify that the selected candidate's
    # key fields are actually injected into the template output.
    env = _prompt_env()
    template = env.get_template("generator_prompt.jinja")

    generator_input = _generator_input()
    payload = build_prompt_payload(generator_input)
    rendered = template.render(**payload)

    assert "Selected Candidate:" in rendered
    assert "MPI_WAIT_REDUCTION" in rendered
    assert "main timestep loop" in rendered
    assert "Fuse communication phases and reduce unnecessary waits." in rendered
    assert "Communication pattern is safe to reorder locally." in rendered
    assert "Results match baseline output." in rendered
    assert "mpi_wait_pct" in rendered


def test_generator_prompt_renders_original_code_and_context_sections() -> None:
    # The prompt should include the original code and the supporting profiling,
    # telemetry, AST, and flame report context because the model needs all of it
    # to produce a grounded rewrite.
    env = _prompt_env()
    template = env.get_template("generator_prompt.jinja")

    generator_input = _generator_input()
    payload = build_prompt_payload(generator_input)
    rendered = template.render(**payload)

    assert "Original source code:" in rendered
    assert "int main() { return 0; }" in rendered
    assert "Profiling summary:" in rendered
    assert "MPI_Waitall dominates runtime." in rendered
    assert "Telemetry summary:" in rendered
    assert "mpi_wait_pct=42" in rendered
    assert "Structured telemetry:" in rendered
    assert "AST context:" in rendered
    assert "FunctionDecl(main)" in rendered
    assert "Flame report:" in rendered
    assert "main 80%" in rendered


def test_generator_prompt_renders_ranked_candidates_block() -> None:
    # The payload formatter includes ranked candidates text for context.
    # This test checks that the prompt includes that block, even though v1
    # only allows the selected candidate to be implemented.
    env = _prompt_env()
    template = env.get_template("generator_prompt.jinja")

    first = _valid_candidate()
    second = _valid_candidate()
    second["pattern"] = "OPENMP_SCHEDULE_TUNING"
    second["target"] = "parallel solver loop"
    second["rationale"] = "Observed thread imbalance in the parallel region."
    second["action_sketch"] = "Change loop schedule to static and tune chunk size."

    generator_input = GeneratorInput(
        original_code="int main() { return 0; }",
        profiling_summary="Mixed MPI/OpenMP bottlenecks.",
        telemetry_summary="mpi_wait_pct=42, omp_imbalance_pct=30",
        telemetry_struct={"mpi_wait_pct": 42.0, "omp_imbalance_pct": 30.0},
        ranked_candidates=[first, second],
        selected_candidate=first,
        ast="FunctionDecl(main)",
        flame_report="main 80%",
    )

    payload = build_prompt_payload(generator_input)
    rendered = template.render(**payload)

    assert "Other ranked candidates for context:" in rendered
    assert "Candidate Rank 1" in rendered
    assert "Candidate Rank 2" in rendered
    assert "OPENMP_SCHEDULE_TUNING" in rendered
    assert "parallel solver loop" in rendered


def test_generator_prompt_omits_ast_section_when_empty() -> None:
    # If AST is not provided, the conditional Jinja section should not render.
    env = _prompt_env()
    template = env.get_template("generator_prompt.jinja")

    generator_input = _generator_input(ast=None)
    payload = build_prompt_payload(generator_input)
    rendered = template.render(**payload)

    assert "AST context:" not in rendered
    assert "FunctionDecl(main)" not in rendered


def test_generator_prompt_omits_flame_report_section_when_empty() -> None:
    # If flame report data is not available, the prompt should stay clean and
    # omit that section rather than rendering an empty block.
    env = _prompt_env()
    template = env.get_template("generator_prompt.jinja")

    generator_input = _generator_input(flame_report=None)
    payload = build_prompt_payload(generator_input)
    rendered = template.render(**payload)

    assert "Flame report:" not in rendered
    assert "main 80%" not in rendered


def test_feedback_prompt_renders_freeform_evaluator_feedback() -> None:
    # When plain evaluator feedback is supplied, the backend should choose the
    # feedback prompt and include the feedback text verbatim.
    env = _prompt_env()
    template = env.get_template("generator_feedback_prompt.jinja")

    generator_input = _generator_input(
        evaluator_feedback="Compilation failed due to undeclared variable tmp."
    )
    payload = build_prompt_payload(generator_input)
    rendered = template.render(**payload)

    assert payload["has_feedback"] is True
    assert "Evaluator feedback:" in rendered
    assert "Compilation failed due to undeclared variable tmp." in rendered
    assert "second-pass repair/refinement" in rendered


def test_feedback_prompt_renders_structured_feedback_text() -> None:
    # Structured feedback should be converted to prompt text through
    # resolved_feedback_text() and then rendered into the feedback template.
    env = _prompt_env()
    template = env.get_template("generator_feedback_prompt.jinja")

    structured_feedback = EvaluationFeedback(
        summary="Compilation failed on retry.",
        compile_succeeded=False,
        correctness_succeeded=True,
        performance_improved=False,
        compile_errors=["unknown pragma omp simd"],
        performance_notes=["speedup below threshold"],
    )

    generator_input = _generator_input(
        evaluator_feedback="fallback text that should not be used",
        evaluation_feedback=structured_feedback,
    )
    payload = build_prompt_payload(generator_input)
    rendered = template.render(**payload)

    assert payload["has_feedback"] is True
    assert "Summary: Compilation failed on retry." in rendered
    assert "Compile succeeded: False" in rendered
    assert "Correctness succeeded: True" in rendered
    assert "Performance improved: False" in rendered
    assert "- unknown pragma omp simd" in rendered
    assert "- speedup below threshold" in rendered
    assert "fallback text that should not be used" not in rendered


def test_build_prompt_payload_sets_has_feedback_false_without_feedback() -> None:
    # First-pass generation should not accidentally enable the feedback path.
    generator_input = _generator_input(
        evaluator_feedback=None,
        evaluation_feedback=None,
    )

    payload = build_prompt_payload(generator_input)

    assert payload["has_feedback"] is False
    assert payload["evaluator_feedback"] == ""


def test_build_prompt_payload_sets_has_feedback_true_with_feedback() -> None:
    # Feedback presence drives prompt selection in generator_llm.py, so this
    # boolean needs to be correct.
    generator_input = _generator_input(
        evaluator_feedback="Correctness regression detected."
    )

    payload = build_prompt_payload(generator_input)

    assert payload["has_feedback"] is True
    assert payload["evaluator_feedback"] == "Correctness regression detected."


def test_build_prompt_payload_includes_selected_candidate_text_block() -> None:
    # The utility should generate a human-readable selected candidate block that
    # the prompt template can inject directly.
    generator_input = _generator_input()

    payload = build_prompt_payload(generator_input)
    selected_candidate_text = payload["selected_candidate_text"]

    assert "Selected Candidate:" in selected_candidate_text
    assert "- Pattern: MPI_WAIT_REDUCTION" in selected_candidate_text
    assert "- Target: main timestep loop" in selected_candidate_text
    assert "- Risk Level: medium" in selected_candidate_text


def test_build_prompt_payload_includes_ranked_candidates_text_block() -> None:
    # The utility should also produce a rendered ranked candidates context block
    # for the prompt template.
    generator_input = _generator_input()

    payload = build_prompt_payload(generator_input)
    ranked_candidates_text = payload["ranked_candidates_text"]

    assert "Ranked Candidates:" in ranked_candidates_text
    assert "Candidate Rank 1" in ranked_candidates_text
    assert "MPI_WAIT_REDUCTION" in ranked_candidates_text


def test_generator_prompt_contains_required_json_schema_fields() -> None:
    # This test checks for the key structured output fields that the model is
    # instructed to return. If these disappear from the prompt, parsing will
    # likely become unstable.
    env = _prompt_env()
    template = env.get_template("generator_prompt.jinja")

    generator_input = _generator_input()
    payload = build_prompt_payload(generator_input)
    rendered = template.render(**payload)

    assert '"analysis"' in rendered
    assert '"selected_candidate_pattern"' in rendered
    assert '"selected_candidate_target"' in rendered
    assert '"applied_changes_summary"' in rendered
    assert '"correctness_risks"' in rendered
    assert '"expected_metrics"' in rendered
    assert '"compile_ready"' in rendered
    assert '"final_code"' in rendered


def test_feedback_prompt_contains_required_json_schema_fields() -> None:
    # The feedback template must enforce the same output contract as the first-pass
    # prompt so downstream parsing works the same way.
    env = _prompt_env()
    template = env.get_template("generator_feedback_prompt.jinja")

    generator_input = _generator_input(
        evaluator_feedback="Compilation failed due to invalid pragma."
    )
    payload = build_prompt_payload(generator_input)
    rendered = template.render(**payload)

    assert '"analysis"' in rendered
    assert '"selected_candidate_pattern"' in rendered
    assert '"selected_candidate_target"' in rendered
    assert '"applied_changes_summary"' in rendered
    assert '"correctness_risks"' in rendered
    assert '"expected_metrics"' in rendered
    assert '"compile_ready"' in rendered
    assert '"final_code"' in rendered