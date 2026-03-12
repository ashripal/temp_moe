from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from ..advisor import AdvisorResult
from .generator_schema import GeneratorInput, GeneratorResult
from .generator_utils import (
    build_failure_result,
    prepare_retry_input,
    select_candidate,
    validate_ranked_candidates,
)


@runtime_checkable
class GeneratorBackend(Protocol):
    """
    Backend protocol for the generator LLM layer.

    Concrete implementations should live in generator_llm.py and use a real
    model backend, such as a Hugging Face Transformers model.
    """

    def generate(self, generator_input: GeneratorInput) -> GeneratorResult:
        ...


class CodeGenerator:
    """
    Orchestrates the Generator stage for the MoE HPC Optimization Advisor.

    Responsibilities:
      - validate advisor-ranked candidates
      - choose one candidate for v1 generation
      - construct GeneratorInput
      - invoke the backend generator
      - provide safe unchanged-code fallback on failure
      - support feedback-driven retry
    """

    def __init__(
        self,
        backend: GeneratorBackend,
        selected_candidate_index: int = 0,
    ) -> None:
        self.backend = backend
        self.selected_candidate_index = selected_candidate_index

    def from_advisor_result(
        self,
        advisor_result: AdvisorResult,
        original_code: str,
        profiling_summary: str,
        telemetry_summary: str,
        telemetry_struct: dict[str, float],
        ast: Optional[str] = None,
        flame_report: Optional[str] = None,
        evaluator_feedback: Optional[str] = None,
        selected_candidate_index: Optional[int] = None,
    ) -> GeneratorInput:
        """
        Build a GeneratorInput object directly from AdvisorResult.

        Primary handoff path:
            MoEAdvisor.run(...) -> CodeGenerator.generate(...)
        """
        ranked_candidates = advisor_result.final_ranked_candidates or []
        index = (
            self.selected_candidate_index
            if selected_candidate_index is None
            else selected_candidate_index
        )

        selected_candidate = select_candidate(ranked_candidates, index=index)

        return GeneratorInput(
            original_code=original_code,
            profiling_summary=profiling_summary,
            telemetry_summary=telemetry_summary,
            telemetry_struct=telemetry_struct,
            ranked_candidates=ranked_candidates,
            selected_candidate=selected_candidate,
            ast=ast,
            flame_report=flame_report,
            evaluator_feedback=evaluator_feedback,
            evaluation_feedback=None,
        )

    def generate(
        self,
        generator_input: GeneratorInput,
    ) -> GeneratorResult:
        """
        Run the first-pass generator.

        Behavior:
          1. validate ranked candidates
          2. ensure a selected candidate exists
          3. invoke backend generator
          4. return safe fallback if generation fails
        """
        candidate_errors = validate_ranked_candidates(
            generator_input.ranked_candidates
        )
        if candidate_errors:
            return build_failure_result(
                original_code=generator_input.original_code,
                selected_candidate=generator_input.selected_candidate,
                failure_reason="; ".join(candidate_errors),
                used_feedback=bool(generator_input.resolved_feedback_text()),
            )

        if generator_input.selected_candidate is None:
            generator_input.selected_candidate = select_candidate(
                generator_input.ranked_candidates,
                index=self.selected_candidate_index,
            )

        if generator_input.selected_candidate is None:
            return build_failure_result(
                original_code=generator_input.original_code,
                selected_candidate=None,
                failure_reason="No valid ranked candidate was available for generation.",
                used_feedback=bool(generator_input.resolved_feedback_text()),
            )

        try:
            result = self.backend.generate(generator_input)
        except Exception as exc:
            return build_failure_result(
                original_code=generator_input.original_code,
                selected_candidate=generator_input.selected_candidate,
                failure_reason=f"Generator backend raised an exception: {exc}",
                used_feedback=bool(generator_input.resolved_feedback_text()),
            )

        if not result.final_code.strip():
            return build_failure_result(
                original_code=generator_input.original_code,
                selected_candidate=generator_input.selected_candidate,
                failure_reason="Generator backend returned empty final_code.",
                used_feedback=bool(generator_input.resolved_feedback_text()),
            )

        if not result.selected_candidate_pattern:
            result.selected_candidate_pattern = str(
                generator_input.selected_candidate.get("pattern", "UNKNOWN_PATTERN")
            )

        if not result.selected_candidate_target:
            result.selected_candidate_target = str(
                generator_input.selected_candidate.get("target", "UNKNOWN_TARGET")
            )

        if not result.expected_metrics:
            raw_metrics = generator_input.selected_candidate.get(
                "performance_metrics", []
            )
            if isinstance(raw_metrics, list):
                result.expected_metrics = [str(x) for x in raw_metrics]

        result.used_feedback = bool(generator_input.resolved_feedback_text())
        result.generation_succeeded = True
        return result

    def retry_with_feedback(
        self,
        previous_input: GeneratorInput,
        evaluator_feedback: str,
    ) -> GeneratorResult:
        """
        Re-run generation using evaluator feedback.

        The selected candidate is preserved unless explicitly changed before
        retry. This matches the v1 single-candidate policy.
        """
        retry_input = prepare_retry_input(
            previous_input=previous_input,
            evaluator_feedback=evaluator_feedback,
        )
        return self.generate(retry_input)