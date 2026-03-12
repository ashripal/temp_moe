from __future__ import annotations

from implementation.advisor import AdvisorResult
from implementation.generator.generator import CodeGenerator
from implementation.generator.generator_schema import GeneratorInput, GeneratorResult
from implementation.router import RoutingDecision


# Build a valid ranked candidate matching the real advisor/generator schema.
# Important: parameters_to_sweep must be dict[str, list], and rollback_criteria
# must be list[str], otherwise the generator will reject the candidate before
# any backend call is made.
def _valid_candidate(
    *,
    pattern: str = "MPI_WAIT_REDUCTION",
    target: str = "main timestep loop",
) -> dict:
    return {
        "pattern": pattern,
        "target": target,
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


# Build a minimal RoutingDecision matching the current router.py schema.
def _routing_decision() -> RoutingDecision:
    return RoutingDecision(
        selected_experts=["Communication & Resilience Expert"],
        reason="MPI wait high (mpi_wait_pct=42.0).",
    )


# Backend stub used to test integration between the generator orchestration
# layer and the advisor handoff.
class RecordingBackend:
    def __init__(self) -> None:
        self.calls: list[GeneratorInput] = []

    def generate(self, generator_input: GeneratorInput) -> GeneratorResult:
        self.calls.append(generator_input)

        selected = generator_input.selected_candidate or {}

        return GeneratorResult(
            analysis="Applied selected candidate to the source code.",
            selected_candidate_pattern=str(
                selected.get("pattern", "UNKNOWN_PATTERN")
            ),
            selected_candidate_target=str(
                selected.get("target", "UNKNOWN_TARGET")
            ),
            applied_changes_summary="Applied a conservative rewrite.",
            final_code="int main() { /* optimized */ return 0; }",
            correctness_risks=["Recheck synchronization semantics."],
            expected_metrics=["mpi_wait_pct", "runtime_seconds"],
            compile_ready=True,
            used_feedback=bool(generator_input.resolved_feedback_text()),
            generation_succeeded=True,
            failure_reason=None,
        )


# Backend stub that returns empty code so we can verify unchanged-code fallback
# across the full advisor -> generator handoff.
class EmptyCodeBackend:
    def generate(self, generator_input: GeneratorInput) -> GeneratorResult:
        selected = generator_input.selected_candidate or {}
        return GeneratorResult(
            analysis="No rewrite produced.",
            selected_candidate_pattern=str(
                selected.get("pattern", "UNKNOWN_PATTERN")
            ),
            selected_candidate_target=str(
                selected.get("target", "UNKNOWN_TARGET")
            ),
            applied_changes_summary="No valid code emitted.",
            final_code="",
            correctness_risks=[],
            expected_metrics=[],
            compile_ready=False,
            used_feedback=bool(generator_input.resolved_feedback_text()),
            generation_succeeded=False,
            failure_reason="Backend produced empty code.",
        )


def _advisor_result_with_candidates() -> AdvisorResult:
    # Build a minimal AdvisorResult using the current handoff contract:
    # final_ranked_candidates is the generator-facing output.
    return AdvisorResult(
        routing=_routing_decision(),
        expert_outputs=[],
        final_ranked_candidates=[
            _valid_candidate(pattern="MPI_WAIT_REDUCTION", target="main timestep loop"),
            _valid_candidate(
                pattern="OPENMP_SCHEDULE_TUNING",
                target="parallel solver loop",
            ),
        ],
    )


def test_from_advisor_result_builds_generator_input_with_top_candidate() -> None:
    # The generator should convert AdvisorResult into GeneratorInput and pick
    # the top-ranked candidate by default.
    backend = RecordingBackend()
    generator = CodeGenerator(backend=backend)

    advisor_result = _advisor_result_with_candidates()

    generator_input = generator.from_advisor_result(
        advisor_result=advisor_result,
        original_code="int main() { return 0; }",
        profiling_summary="MPI_Waitall dominates runtime.",
        telemetry_summary="mpi_wait_pct=42",
        telemetry_struct={"mpi_wait_pct": 42.0},
        ast="FunctionDecl(main)",
        flame_report="main 80%",
    )

    assert generator_input.original_code == "int main() { return 0; }"
    assert generator_input.profiling_summary == "MPI_Waitall dominates runtime."
    assert generator_input.telemetry_summary == "mpi_wait_pct=42"
    assert generator_input.telemetry_struct == {"mpi_wait_pct": 42.0}
    assert len(generator_input.ranked_candidates) == 2
    assert generator_input.selected_candidate is not None
    assert generator_input.selected_candidate["pattern"] == "MPI_WAIT_REDUCTION"
    assert generator_input.selected_candidate["target"] == "main timestep loop"


def test_from_advisor_result_respects_selected_candidate_index() -> None:
    # When an alternate candidate index is requested, the generator should use
    # that candidate as the selected rewrite plan.
    backend = RecordingBackend()
    generator = CodeGenerator(backend=backend)

    advisor_result = _advisor_result_with_candidates()

    generator_input = generator.from_advisor_result(
        advisor_result=advisor_result,
        original_code="int main() { return 0; }",
        profiling_summary="Mixed MPI/OpenMP bottlenecks.",
        telemetry_summary="mpi_wait_pct=42, omp_imbalance_pct=30",
        telemetry_struct={"mpi_wait_pct": 42.0, "omp_imbalance_pct": 30.0},
        selected_candidate_index=1,
    )

    assert generator_input.selected_candidate is not None
    assert generator_input.selected_candidate["pattern"] == "OPENMP_SCHEDULE_TUNING"
    assert generator_input.selected_candidate["target"] == "parallel solver loop"


def test_generate_accepts_advisor_handoff_and_returns_structured_result() -> None:
    # End-to-end within the generator stage:
    # AdvisorResult -> GeneratorInput -> backend.generate() -> GeneratorResult.
    backend = RecordingBackend()
    generator = CodeGenerator(backend=backend)

    advisor_result = _advisor_result_with_candidates()
    generator_input = generator.from_advisor_result(
        advisor_result=advisor_result,
        original_code="int main() { return 0; }",
        profiling_summary="MPI_Waitall dominates runtime.",
        telemetry_summary="mpi_wait_pct=42",
        telemetry_struct={"mpi_wait_pct": 42.0},
    )

    result = generator.generate(generator_input)

    assert len(backend.calls) == 1
    assert result.generation_succeeded is True
    assert result.final_code == "int main() { /* optimized */ return 0; }"
    assert result.selected_candidate_pattern == "MPI_WAIT_REDUCTION"
    assert result.selected_candidate_target == "main timestep loop"
    assert result.compile_ready is True


def test_generate_passes_selected_candidate_and_context_to_backend() -> None:
    # Verify that the backend actually receives the advisor-derived selected
    # candidate and surrounding context fields.
    backend = RecordingBackend()
    generator = CodeGenerator(backend=backend)

    advisor_result = _advisor_result_with_candidates()
    generator_input = generator.from_advisor_result(
        advisor_result=advisor_result,
        original_code="int main() { return 0; }",
        profiling_summary="Profile text",
        telemetry_summary="Telemetry text",
        telemetry_struct={"mpi_wait_pct": 42.0},
        ast="FunctionDecl(main)",
        flame_report="main 80%",
    )

    generator.generate(generator_input)

    assert len(backend.calls) == 1
    call = backend.calls[0]
    assert call.selected_candidate is not None
    assert call.selected_candidate["pattern"] == "MPI_WAIT_REDUCTION"
    assert call.original_code == "int main() { return 0; }"
    assert call.profiling_summary == "Profile text"
    assert call.telemetry_summary == "Telemetry text"
    assert call.telemetry_struct == {"mpi_wait_pct": 42.0}
    assert call.ast == "FunctionDecl(main)"
    assert call.flame_report == "main 80%"


def test_generate_fills_missing_expected_metrics_from_selected_candidate() -> None:
    # If the backend omits expected_metrics, the orchestration layer should fill
    # them from the selected candidate's performance_metrics.
    class MissingMetricsBackend:
        def generate(self, generator_input: GeneratorInput) -> GeneratorResult:
            selected = generator_input.selected_candidate or {}
            return GeneratorResult(
                analysis="Applied rewrite.",
                selected_candidate_pattern=str(
                    selected.get("pattern", "UNKNOWN_PATTERN")
                ),
                selected_candidate_target=str(
                    selected.get("target", "UNKNOWN_TARGET")
                ),
                applied_changes_summary="Applied selected optimization.",
                final_code="int main() { /* optimized */ return 0; }",
                correctness_risks=[],
                expected_metrics=[],
                compile_ready=True,
                used_feedback=False,
                generation_succeeded=True,
                failure_reason=None,
            )

    generator = CodeGenerator(backend=MissingMetricsBackend())
    advisor_result = _advisor_result_with_candidates()

    generator_input = generator.from_advisor_result(
        advisor_result=advisor_result,
        original_code="int main() { return 0; }",
        profiling_summary="Profile",
        telemetry_summary="Telemetry",
        telemetry_struct={"mpi_wait_pct": 42.0},
    )

    result = generator.generate(generator_input)

    assert result.generation_succeeded is True
    assert result.expected_metrics == ["mpi_wait_pct", "runtime_seconds"]


def test_generate_returns_safe_fallback_when_backend_returns_empty_code() -> None:
    # Even if the backend runs, the orchestration layer must reject empty code
    # and return the unchanged original source safely.
    generator = CodeGenerator(backend=EmptyCodeBackend())
    advisor_result = _advisor_result_with_candidates()

    generator_input = generator.from_advisor_result(
        advisor_result=advisor_result,
        original_code="int main() { return 0; }",
        profiling_summary="Profile",
        telemetry_summary="Telemetry",
        telemetry_struct={"mpi_wait_pct": 42.0},
    )

    result = generator.generate(generator_input)

    assert result.generation_succeeded is False
    assert result.final_code == "int main() { return 0; }"
    assert result.failure_reason is not None
    assert "empty final_code" in result.failure_reason


def test_generate_returns_safe_fallback_when_ranked_candidates_are_invalid() -> None:
    # Invalid advisor handoff data should be caught before backend invocation.
    backend = RecordingBackend()
    generator = CodeGenerator(backend=backend)

    advisor_result = AdvisorResult(
        routing=_routing_decision(),
        expert_outputs=[],
        final_ranked_candidates=[
            {
                # Missing required "pattern" field on purpose.
                "target": "main timestep loop",
                "rationale": "bad candidate",
                "action_sketch": "bad action",
                "preconditions": [],
                "parameters_to_sweep": {},
                "correctness_checks": [],
                "performance_metrics": [],
                "risk_level": "medium",
                "rollback_criteria": [],
            }
        ],
    )

    generator_input = generator.from_advisor_result(
        advisor_result=advisor_result,
        original_code="int main() { return 0; }",
        profiling_summary="Profile",
        telemetry_summary="Telemetry",
        telemetry_struct={"mpi_wait_pct": 42.0},
    )

    result = generator.generate(generator_input)

    assert len(backend.calls) == 0
    assert result.generation_succeeded is False
    assert result.final_code == "int main() { return 0; }"
    assert result.failure_reason is not None


def test_retry_with_feedback_works_after_advisor_handoff() -> None:
    # This checks the full integration path for a second-pass run:
    # AdvisorResult -> GeneratorInput -> retry_with_feedback().
    backend = RecordingBackend()
    generator = CodeGenerator(backend=backend)
    advisor_result = _advisor_result_with_candidates()

    initial_input = generator.from_advisor_result(
        advisor_result=advisor_result,
        original_code="int main() { return 0; }",
        profiling_summary="MPI_Waitall dominates runtime.",
        telemetry_summary="mpi_wait_pct=42",
        telemetry_struct={"mpi_wait_pct": 42.0},
    )

    result = generator.retry_with_feedback(
        previous_input=initial_input,
        evaluator_feedback="Compilation failed due to undeclared temporary buffer.",
    )

    assert len(backend.calls) == 1
    assert backend.calls[0].resolved_feedback_text() == (
        "Compilation failed due to undeclared temporary buffer."
    )
    assert result.generation_succeeded is True
    assert result.used_feedback is True
    assert result.selected_candidate_pattern == "MPI_WAIT_REDUCTION"


def test_from_advisor_result_with_no_candidates_leads_to_safe_failure() -> None:
    # If the advisor returns no ranked candidates, the generator should fail
    # safely and return the original code unchanged.
    backend = RecordingBackend()
    generator = CodeGenerator(backend=backend)

    advisor_result = AdvisorResult(
        routing=_routing_decision(),
        expert_outputs=[],
        final_ranked_candidates=[],
    )

    generator_input = generator.from_advisor_result(
        advisor_result=advisor_result,
        original_code="int main() { return 0; }",
        profiling_summary="Profile",
        telemetry_summary="Telemetry",
        telemetry_struct={"mpi_wait_pct": 42.0},
    )

    result = generator.generate(generator_input)

    assert len(backend.calls) == 0
    assert result.generation_succeeded is False
    assert result.final_code == "int main() { return 0; }"
    assert result.failure_reason is not None