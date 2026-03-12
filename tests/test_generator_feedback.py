from __future__ import annotations

from implementation.generator.generator import CodeGenerator
from implementation.generator.generator_schema import GeneratorInput, GeneratorResult


# Build one valid candidate matching the real advisor/generator schema validator.
# Important: parameters_to_sweep must be dict[str, list], and rollback_criteria
# must be list[str], otherwise generation will fail before the backend is called.
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


# Build a minimal valid GeneratorInput used across tests.
def _generator_input(
    *,
    evaluator_feedback: str | None = None,
) -> GeneratorInput:
    candidate = _valid_candidate()
    return GeneratorInput(
        original_code="int main() { return 0; }",
        profiling_summary="MPI_Waitall dominates runtime.",
        telemetry_summary="mpi_wait_pct=42",
        telemetry_struct={"mpi_wait_pct": 42.0},
        ranked_candidates=[candidate],
        selected_candidate=candidate,
        ast="FunctionDecl(main)",
        flame_report="main 80%",
        evaluator_feedback=evaluator_feedback,
        evaluation_feedback=None,
    )


class RecordingBackend:
    """
    Backend stub that records the GeneratorInput it receives and returns a
    successful GeneratorResult.

    This is not acting as an LLM. It is a protocol-conforming backend used
    to test generator orchestration and retry behavior in isolation.
    """

    def __init__(self) -> None:
        self.calls: list[GeneratorInput] = []

    def generate(self, generator_input: GeneratorInput) -> GeneratorResult:
        self.calls.append(generator_input)
        return GeneratorResult(
            analysis="Applied retry-aware rewrite.",
            selected_candidate_pattern=str(
                generator_input.selected_candidate.get("pattern", "UNKNOWN_PATTERN")
            ),
            selected_candidate_target=str(
                generator_input.selected_candidate.get("target", "UNKNOWN_TARGET")
            ),
            applied_changes_summary="Adjusted code in response to evaluator feedback.",
            final_code="int main() { /* optimized */ return 0; }",
            correctness_risks=["Recheck communication ordering semantics."],
            expected_metrics=["mpi_wait_pct", "runtime_seconds"],
            compile_ready=True,
            used_feedback=bool(generator_input.resolved_feedback_text()),
            generation_succeeded=True,
            failure_reason=None,
        )


class EmptyCodeBackend:
    """
    Backend stub that returns an empty final_code string so we can verify that
    the orchestration layer safely falls back to unchanged code on retry.
    """

    def generate(self, generator_input: GeneratorInput) -> GeneratorResult:
        return GeneratorResult(
            analysis="Generation failed.",
            selected_candidate_pattern="MPI_WAIT_REDUCTION",
            selected_candidate_target="main timestep loop",
            applied_changes_summary="No valid rewrite produced.",
            final_code="",
            correctness_risks=[],
            expected_metrics=[],
            compile_ready=False,
            used_feedback=bool(generator_input.resolved_feedback_text()),
            generation_succeeded=False,
            failure_reason="Backend returned empty code.",
        )


class RaisingBackend:
    """
    Backend stub that raises an exception so we can verify that retry logic
    still returns a safe fallback result.
    """

    def generate(self, generator_input: GeneratorInput) -> GeneratorResult:
        raise RuntimeError("Simulated backend failure during retry.")


def test_retry_with_feedback_preserves_selected_candidate() -> None:
    # The retry path should preserve the original selected candidate rather than
    # silently changing the rewrite plan.
    backend = RecordingBackend()
    generator = CodeGenerator(backend=backend)

    original_input = _generator_input()
    original_selected = original_input.selected_candidate

    result = generator.retry_with_feedback(
        previous_input=original_input,
        evaluator_feedback="Compilation failed due to undeclared variable tmp.",
    )

    assert result.generation_succeeded is True
    assert len(backend.calls) == 1
    assert backend.calls[0].selected_candidate == original_selected
    assert backend.calls[0].selected_candidate["pattern"] == "MPI_WAIT_REDUCTION"


def test_retry_with_feedback_updates_feedback_text() -> None:
    # The new evaluator feedback passed into retry should replace any prior
    # feedback text on the retried input.
    backend = RecordingBackend()
    generator = CodeGenerator(backend=backend)

    original_input = _generator_input(evaluator_feedback="Old feedback")
    generator.retry_with_feedback(
        previous_input=original_input,
        evaluator_feedback="Correctness regression detected on test case 3.",
    )

    assert len(backend.calls) == 1
    assert (
        backend.calls[0].evaluator_feedback
        == "Correctness regression detected on test case 3."
    )


def test_retry_with_feedback_preserves_original_context_fields() -> None:
    # The retry input should carry over all original generation context:
    # code, summaries, telemetry, AST, and flame report.
    backend = RecordingBackend()
    generator = CodeGenerator(backend=backend)

    original_input = _generator_input()
    generator.retry_with_feedback(
        previous_input=original_input,
        evaluator_feedback="Formatting issue detected.",
    )

    assert len(backend.calls) == 1
    retried_input = backend.calls[0]

    assert retried_input.original_code == original_input.original_code
    assert retried_input.profiling_summary == original_input.profiling_summary
    assert retried_input.telemetry_summary == original_input.telemetry_summary
    assert retried_input.telemetry_struct == original_input.telemetry_struct
    assert retried_input.ast == original_input.ast
    assert retried_input.flame_report == original_input.flame_report


def test_retry_with_feedback_marks_result_as_using_feedback() -> None:
    # Successful retry results should reflect that feedback was used.
    backend = RecordingBackend()
    generator = CodeGenerator(backend=backend)

    result = generator.retry_with_feedback(
        previous_input=_generator_input(),
        evaluator_feedback="Performance gain insufficient.",
    )

    assert result.generation_succeeded is True
    assert result.used_feedback is True


def test_retry_with_feedback_returns_safe_fallback_on_empty_code() -> None:
    # If the backend returns empty final_code during retry, the orchestration
    # layer should return unchanged code rather than passing through bad output.
    backend = EmptyCodeBackend()
    generator = CodeGenerator(backend=backend)

    previous_input = _generator_input()
    result = generator.retry_with_feedback(
        previous_input=previous_input,
        evaluator_feedback="Compilation failed.",
    )

    assert result.generation_succeeded is False
    assert result.final_code == previous_input.original_code
    assert result.failure_reason is not None
    assert "empty final_code" in result.failure_reason


def test_retry_with_feedback_returns_safe_fallback_on_backend_exception() -> None:
    # Exceptions raised during retry should not crash the pipeline.
    # The generator should return unchanged code with a structured failure reason.
    backend = RaisingBackend()
    generator = CodeGenerator(backend=backend)

    previous_input = _generator_input()
    result = generator.retry_with_feedback(
        previous_input=previous_input,
        evaluator_feedback="Compile error in optimized version.",
    )

    assert result.generation_succeeded is False
    assert result.final_code == previous_input.original_code
    assert result.failure_reason is not None
    assert "raised an exception" in result.failure_reason


def test_retry_with_feedback_keeps_candidate_pattern_and_target_in_success_result() -> None:
    # A successful retry result should remain traceable to the originally
    # selected candidate pattern and target.
    backend = RecordingBackend()
    generator = CodeGenerator(backend=backend)

    result = generator.retry_with_feedback(
        previous_input=_generator_input(),
        evaluator_feedback="Patch formatting issue detected.",
    )

    assert result.selected_candidate_pattern == "MPI_WAIT_REDUCTION"
    assert result.selected_candidate_target == "main timestep loop"


def test_retry_with_feedback_does_not_mutate_original_input_candidate() -> None:
    # The retry path should not mutate the original selected candidate dict in a
    # way that changes its meaning for future operations.
    backend = RecordingBackend()
    generator = CodeGenerator(backend=backend)

    previous_input = _generator_input()
    original_candidate_before = dict(previous_input.selected_candidate)

    generator.retry_with_feedback(
        previous_input=previous_input,
        evaluator_feedback="Need second pass.",
    )

    assert previous_input.selected_candidate == original_candidate_before


def test_retry_with_feedback_uses_new_feedback_even_if_previous_input_had_none() -> None:
    # Retry should work even if the first-pass input had no feedback at all.
    backend = RecordingBackend()
    generator = CodeGenerator(backend=backend)

    previous_input = _generator_input(evaluator_feedback=None)

    result = generator.retry_with_feedback(
        previous_input=previous_input,
        evaluator_feedback="Correctness regression after optimization.",
    )

    assert result.generation_succeeded is True
    assert len(backend.calls) == 1
    assert (
        backend.calls[0].resolved_feedback_text()
        == "Correctness regression after optimization."
    )