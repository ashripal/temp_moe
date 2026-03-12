from __future__ import annotations

from implementation.generator.generator_schema import (
    EvaluationFeedback,
    GeneratorInput,
    GeneratorResult,
    validate_evaluation_feedback,
    validate_generator_input,
    validate_generator_result,
)


# Build one valid candidate that matches the real schema validator used by the
# generator package. The important fields are:
# - parameters_to_sweep: dict[str, list]
# - rollback_criteria: list[str]
def _valid_candidate() -> dict:
    return {
        "pattern": "MPI_WAIT_REDUCTION",
        "target": "main timestep loop",
        "rationale": "High MPI_Waitall time suggests excessive synchronization overhead.",
        "action_sketch": "Fuse communication phases and reduce unnecessary waits.",
        "preconditions": ["Communication pattern is safe to reorder locally."],
        "parameters_to_sweep": {
            "batch_size": [16, 32, 64],
            "message_grouping": [1, 2, 4],
        },
        "correctness_checks": ["Results match baseline output."],
        "performance_metrics": ["mpi_wait_pct", "runtime_seconds"],
        "risk_level": "medium",
        "rollback_criteria": [
            "Rollback if runtime increases.",
            "Rollback if outputs diverge from baseline.",
        ],
    }


def test_evaluation_feedback_to_dict_contains_expected_fields() -> None:
    feedback = EvaluationFeedback(
        summary="Compilation failed due to undeclared variable.",
        compile_succeeded=False,
        correctness_succeeded=None,
        performance_improved=None,
        compile_errors=["undeclared identifier 'tmp'"],
        correctness_failures=[],
        performance_notes=[],
        formatting_issues=["code fence present"],
    )

    result = feedback.to_dict()

    assert result["summary"] == "Compilation failed due to undeclared variable."
    assert result["compile_succeeded"] is False
    assert result["compile_errors"] == ["undeclared identifier 'tmp'"]
    assert result["formatting_issues"] == ["code fence present"]


def test_evaluation_feedback_to_prompt_string_includes_all_sections() -> None:
    feedback = EvaluationFeedback(
        summary="Second pass needed.",
        compile_succeeded=False,
        correctness_succeeded=True,
        performance_improved=False,
        compile_errors=["missing semicolon"],
        correctness_failures=["output mismatch on test 3"],
        performance_notes=["speedup below threshold"],
        formatting_issues=["response included markdown fences"],
    )

    text = feedback.to_prompt_string()

    assert "Summary: Second pass needed." in text
    assert "Compile succeeded: False" in text
    assert "Correctness succeeded: True" in text
    assert "Performance improved: False" in text
    assert "Compile errors:" in text
    assert "- missing semicolon" in text
    assert "Correctness failures:" in text
    assert "- output mismatch on test 3" in text
    assert "Performance notes:" in text
    assert "- speedup below threshold" in text
    assert "Formatting / patch issues:" in text
    assert "- response included markdown fences" in text


def test_validate_evaluation_feedback_accepts_valid_feedback() -> None:
    feedback = EvaluationFeedback(
        summary="Looks good.",
        compile_succeeded=True,
        correctness_succeeded=True,
        performance_improved=True,
    )

    errors = validate_evaluation_feedback(feedback)

    assert errors == []


def test_validate_evaluation_feedback_rejects_empty_summary() -> None:
    feedback = EvaluationFeedback(summary="")

    errors = validate_evaluation_feedback(feedback)

    assert any("summary" in error for error in errors)


def test_validate_evaluation_feedback_rejects_non_string_list_entries() -> None:
    feedback = EvaluationFeedback(summary="Bad list type.")
    feedback.compile_errors = ["valid", 123]  # type: ignore[list-item]

    errors = validate_evaluation_feedback(feedback)

    assert any("compile_errors" in error for error in errors)


def test_generator_input_to_dict_serializes_structured_feedback() -> None:
    structured_feedback = EvaluationFeedback(
        summary="Compilation failed.",
        compile_succeeded=False,
        compile_errors=["unknown pragma"],
    )

    generator_input = GeneratorInput(
        original_code="int main() { return 0; }",
        profiling_summary="hotspot in main loop",
        telemetry_summary="mpi_wait_pct=42",
        telemetry_struct={"mpi_wait_pct": 42.0},
        ranked_candidates=[_valid_candidate()],
        selected_candidate=_valid_candidate(),
        ast="FunctionDecl(main)",
        flame_report="main 80%",
        evaluator_feedback="freeform feedback",
        evaluation_feedback=structured_feedback,
    )

    result = generator_input.to_dict()

    assert result["original_code"] == "int main() { return 0; }"
    assert result["telemetry_struct"] == {"mpi_wait_pct": 42.0}
    assert result["evaluation_feedback"]["summary"] == "Compilation failed."
    assert result["evaluation_feedback"]["compile_errors"] == ["unknown pragma"]


def test_generator_input_resolved_feedback_text_prefers_structured_feedback() -> None:
    structured_feedback = EvaluationFeedback(
        summary="Use structured feedback.",
        compile_succeeded=False,
        compile_errors=["bad symbol"],
    )

    generator_input = GeneratorInput(
        original_code="code",
        profiling_summary="profile",
        telemetry_summary="telemetry",
        telemetry_struct={"x": 1.0},
        ranked_candidates=[_valid_candidate()],
        evaluator_feedback="plain text fallback",
        evaluation_feedback=structured_feedback,
    )

    resolved = generator_input.resolved_feedback_text()

    assert resolved is not None
    assert "Summary: Use structured feedback." in resolved
    assert "plain text fallback" not in resolved


def test_generator_input_resolved_feedback_text_falls_back_to_freeform_text() -> None:
    generator_input = GeneratorInput(
        original_code="code",
        profiling_summary="profile",
        telemetry_summary="telemetry",
        telemetry_struct={"x": 1.0},
        ranked_candidates=[_valid_candidate()],
        evaluator_feedback="plain evaluator feedback",
        evaluation_feedback=None,
    )

    assert generator_input.resolved_feedback_text() == "plain evaluator feedback"


def test_validate_generator_input_accepts_valid_input() -> None:
    generator_input = GeneratorInput(
        original_code="int main() { return 0; }",
        profiling_summary="Loop dominates runtime.",
        telemetry_summary="omp_imbalance_pct=30",
        telemetry_struct={"omp_imbalance_pct": 30.0},
        ranked_candidates=[_valid_candidate()],
        selected_candidate=_valid_candidate(),
        ast="FunctionDecl(main)",
        flame_report="main -> loop",
    )

    errors = validate_generator_input(generator_input)

    assert errors == []


def test_validate_generator_input_rejects_empty_required_strings() -> None:
    generator_input = GeneratorInput(
        original_code="",
        profiling_summary="",
        telemetry_summary="",
        telemetry_struct={"x": 1.0},
        ranked_candidates=[_valid_candidate()],
    )

    errors = validate_generator_input(generator_input)

    assert any("original_code" in error for error in errors)
    assert any("profiling_summary" in error for error in errors)
    assert any("telemetry_summary" in error for error in errors)


def test_validate_generator_input_rejects_non_numeric_telemetry_values() -> None:
    generator_input = GeneratorInput(
        original_code="code",
        profiling_summary="profile",
        telemetry_summary="telemetry",
        telemetry_struct={"mpi_wait_pct": "high"},  # type: ignore[dict-item]
        ranked_candidates=[_valid_candidate()],
    )

    errors = validate_generator_input(generator_input)

    assert any("telemetry_struct" in error for error in errors)


def test_validate_generator_input_rejects_empty_ranked_candidates() -> None:
    generator_input = GeneratorInput(
        original_code="code",
        profiling_summary="profile",
        telemetry_summary="telemetry",
        telemetry_struct={"x": 1.0},
        ranked_candidates=[],
    )

    errors = validate_generator_input(generator_input)

    assert any("ranked_candidates" in error for error in errors)


def test_validate_generator_input_rejects_invalid_ranked_candidate() -> None:
    bad_candidate = _valid_candidate()
    del bad_candidate["pattern"]

    generator_input = GeneratorInput(
        original_code="code",
        profiling_summary="profile",
        telemetry_summary="telemetry",
        telemetry_struct={"x": 1.0},
        ranked_candidates=[bad_candidate],
    )

    errors = validate_generator_input(generator_input)

    assert any("Ranked candidate 0 failed validation" in error for error in errors)


def test_validate_generator_input_rejects_invalid_selected_candidate() -> None:
    bad_candidate = _valid_candidate()
    del bad_candidate["target"]

    generator_input = GeneratorInput(
        original_code="code",
        profiling_summary="profile",
        telemetry_summary="telemetry",
        telemetry_struct={"x": 1.0},
        ranked_candidates=[_valid_candidate()],
        selected_candidate=bad_candidate,
    )

    errors = validate_generator_input(generator_input)

    assert any("selected_candidate" in error for error in errors)


def test_validate_generator_input_rejects_invalid_structured_feedback() -> None:
    feedback = EvaluationFeedback(summary="bad")
    feedback.performance_notes = [123]  # type: ignore[list-item]

    generator_input = GeneratorInput(
        original_code="code",
        profiling_summary="profile",
        telemetry_summary="telemetry",
        telemetry_struct={"x": 1.0},
        ranked_candidates=[_valid_candidate()],
        evaluation_feedback=feedback,
    )

    errors = validate_generator_input(generator_input)

    assert any("performance_notes" in error for error in errors)


def test_generator_result_to_dict_contains_expected_fields() -> None:
    result = GeneratorResult(
        analysis="Applied loop scheduling change.",
        selected_candidate_pattern="OPENMP_SCHEDULE_TUNING",
        selected_candidate_target="parallel loop in solver",
        applied_changes_summary="Changed schedule from dynamic to static.",
        final_code="int main() { return 0; }",
        correctness_risks=["Potential load imbalance on skewed inputs."],
        expected_metrics=["runtime_seconds", "omp_imbalance_pct"],
        compile_ready=True,
        used_feedback=False,
        generation_succeeded=True,
        failure_reason=None,
    )

    result_dict = result.to_dict()

    assert result_dict["analysis"] == "Applied loop scheduling change."
    assert result_dict["selected_candidate_pattern"] == "OPENMP_SCHEDULE_TUNING"
    assert result_dict["compile_ready"] is True
    assert result_dict["generation_succeeded"] is True
    assert result_dict["failure_reason"] is None


def test_validate_generator_result_accepts_valid_result() -> None:
    result = GeneratorResult(
        analysis="Safe rewrite applied.",
        selected_candidate_pattern="MPI_WAIT_REDUCTION",
        selected_candidate_target="main loop",
        applied_changes_summary="Grouped waits conservatively.",
        final_code="int main() { return 0; }",
        correctness_risks=["Ordering assumptions must be preserved."],
        expected_metrics=["mpi_wait_pct"],
        compile_ready=True,
        used_feedback=True,
        generation_succeeded=True,
        failure_reason=None,
    )

    errors = validate_generator_result(result)

    assert errors == []


def test_validate_generator_result_rejects_bad_list_and_bool_fields() -> None:
    result = GeneratorResult(
        analysis="analysis",
        selected_candidate_pattern="pattern",
        selected_candidate_target="target",
        applied_changes_summary="summary",
        final_code="code",
        correctness_risks=["risk"],
        expected_metrics=["metric"],
        compile_ready=True,
        used_feedback=False,
        generation_succeeded=True,
        failure_reason=None,
    )

    result.correctness_risks = "not-a-list"  # type: ignore[assignment]
    result.compile_ready = "yes"  # type: ignore[assignment]

    errors = validate_generator_result(result)

    assert any("correctness_risks" in error for error in errors)
    assert any("compile_ready" in error for error in errors)


def test_validate_generator_result_rejects_non_string_failure_reason() -> None:
    result = GeneratorResult(
        analysis="analysis",
        selected_candidate_pattern="pattern",
        selected_candidate_target="target",
        applied_changes_summary="summary",
        final_code="code",
        compile_ready=False,
        used_feedback=False,
        generation_succeeded=False,
        failure_reason=None,
    )

    result.failure_reason = 123  # type: ignore[assignment]

    errors = validate_generator_result(result)

    assert any("failure_reason" in error for error in errors)