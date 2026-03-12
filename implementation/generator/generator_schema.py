from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from ..schema import validate_candidate_dict


@dataclass
class EvaluationFeedback:
    """
    Structured evaluator feedback for generator retry.

    This is intentionally lightweight for v1. It can represent both:
      - coarse text feedback from an evaluator
      - more structured compile/correctness/performance signals

    The generator can still consume a plain string if needed, but this schema
    gives us a stable format for future evaluator integration.
    """
    summary: str
    compile_succeeded: Optional[bool] = None
    correctness_succeeded: Optional[bool] = None
    performance_improved: Optional[bool] = None
    compile_errors: List[str] = field(default_factory=list)
    correctness_failures: List[str] = field(default_factory=list)
    performance_notes: List[str] = field(default_factory=list)
    formatting_issues: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_prompt_string(self) -> str:
        """
        Render feedback into a compact text block suitable for prompt injection.
        """
        lines: List[str] = [f"Summary: {self.summary}"]

        if self.compile_succeeded is not None:
            lines.append(f"Compile succeeded: {self.compile_succeeded}")
        if self.correctness_succeeded is not None:
            lines.append(f"Correctness succeeded: {self.correctness_succeeded}")
        if self.performance_improved is not None:
            lines.append(f"Performance improved: {self.performance_improved}")

        if self.compile_errors:
            lines.append("Compile errors:")
            lines.extend(f"- {item}" for item in self.compile_errors)

        if self.correctness_failures:
            lines.append("Correctness failures:")
            lines.extend(f"- {item}" for item in self.correctness_failures)

        if self.performance_notes:
            lines.append("Performance notes:")
            lines.extend(f"- {item}" for item in self.performance_notes)

        if self.formatting_issues:
            lines.append("Formatting / patch issues:")
            lines.extend(f"- {item}" for item in self.formatting_issues)

        return "\n".join(lines)


@dataclass
class GeneratorInput:
    """
    Structured handoff into the generator stage.

    ranked_candidates:
        Full advisor-ranked candidate list.

    selected_candidate:
        The single candidate chosen for v1 generation. If None, the generator
        orchestration layer should select one from ranked_candidates.

    evaluator_feedback:
        Freeform string form retained for compatibility with simple retry flows.

    evaluation_feedback:
        Structured evaluator feedback form for more robust retry handling.
    """
    original_code: str
    profiling_summary: str
    telemetry_summary: str
    telemetry_struct: Dict[str, float]
    ranked_candidates: List[Dict[str, Any]]
    selected_candidate: Optional[Dict[str, Any]] = None
    ast: Optional[str] = None
    flame_report: Optional[str] = None
    evaluator_feedback: Optional[str] = None
    evaluation_feedback: Optional[EvaluationFeedback] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.evaluation_feedback is not None:
            data["evaluation_feedback"] = self.evaluation_feedback.to_dict()
        return data

    def resolved_feedback_text(self) -> Optional[str]:
        """
        Return the most useful textual feedback representation for prompting.
        """
        if self.evaluation_feedback is not None:
            return self.evaluation_feedback.to_prompt_string()
        return self.evaluator_feedback


@dataclass
class GeneratorResult:
    """
    Structured output from the generator stage.

    This result is returned even on failure so the pipeline always has a stable
    object to inspect, log, evaluate, or pass to retry logic.
    """
    analysis: str
    selected_candidate_pattern: str
    selected_candidate_target: str
    applied_changes_summary: str
    final_code: str
    correctness_risks: List[str] = field(default_factory=list)
    expected_metrics: List[str] = field(default_factory=list)
    compile_ready: bool = False
    used_feedback: bool = False
    generation_succeeded: bool = False
    failure_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def validate_evaluation_feedback(feedback: EvaluationFeedback) -> List[str]:
    """
    Lightweight validation for EvaluationFeedback.
    """
    errors: List[str] = []

    if not isinstance(feedback, EvaluationFeedback):
        return ["Evaluation feedback must be an EvaluationFeedback instance."]

    if not isinstance(feedback.summary, str) or not feedback.summary.strip():
        errors.append("Evaluation feedback 'summary' must be a non-empty string.")

    list_fields = [
        ("compile_errors", feedback.compile_errors),
        ("correctness_failures", feedback.correctness_failures),
        ("performance_notes", feedback.performance_notes),
        ("formatting_issues", feedback.formatting_issues),
    ]
    for field_name, value in list_fields:
        if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
            errors.append(f"Evaluation feedback '{field_name}' must be list[str].")

    bool_fields = [
        ("compile_succeeded", feedback.compile_succeeded),
        ("correctness_succeeded", feedback.correctness_succeeded),
        ("performance_improved", feedback.performance_improved),
    ]
    for field_name, value in bool_fields:
        if value is not None and not isinstance(value, bool):
            errors.append(f"Evaluation feedback '{field_name}' must be bool | None.")

    return errors


def validate_generator_input(obj: GeneratorInput) -> List[str]:
    """
    Lightweight validation for GeneratorInput.
    """
    errors: List[str] = []

    if not isinstance(obj, GeneratorInput):
        return ["Generator input must be a GeneratorInput instance."]

    required_text_fields = [
        ("original_code", obj.original_code),
        ("profiling_summary", obj.profiling_summary),
        ("telemetry_summary", obj.telemetry_summary),
    ]
    for field_name, value in required_text_fields:
        if not isinstance(value, str) or not value.strip():
            errors.append(f"'{field_name}' must be a non-empty string.")

    if not isinstance(obj.telemetry_struct, dict):
        errors.append("'telemetry_struct' must be dict[str, float].")
    else:
        for key, value in obj.telemetry_struct.items():
            if not isinstance(key, str):
                errors.append("'telemetry_struct' keys must be strings.")
                break
            if not isinstance(value, (int, float)):
                errors.append("'telemetry_struct' values must be numeric.")
                break

    if not isinstance(obj.ranked_candidates, list):
        errors.append("'ranked_candidates' must be a list of candidate dictionaries.")
    elif not obj.ranked_candidates:
        errors.append("'ranked_candidates' must not be empty.")
    else:
        for idx, candidate in enumerate(obj.ranked_candidates):
            if not isinstance(candidate, dict):
                errors.append(f"Ranked candidate {idx} must be a dictionary.")
                continue
            candidate_errors = validate_candidate_dict(candidate)
            if candidate_errors:
                errors.append(
                    f"Ranked candidate {idx} failed validation: {'; '.join(candidate_errors)}"
                )

    if obj.selected_candidate is not None:
        if not isinstance(obj.selected_candidate, dict):
            errors.append("'selected_candidate' must be a candidate dictionary or None.")
        else:
            candidate_errors = validate_candidate_dict(obj.selected_candidate)
            if candidate_errors:
                errors.append(
                    "'selected_candidate' failed validation: "
                    + "; ".join(candidate_errors)
                )

    if obj.ast is not None and not isinstance(obj.ast, str):
        errors.append("'ast' must be str | None.")
    if obj.flame_report is not None and not isinstance(obj.flame_report, str):
        errors.append("'flame_report' must be str | None.")
    if obj.evaluator_feedback is not None and not isinstance(obj.evaluator_feedback, str):
        errors.append("'evaluator_feedback' must be str | None.")

    if obj.evaluation_feedback is not None:
        errors.extend(validate_evaluation_feedback(obj.evaluation_feedback))

    return errors


def validate_generator_result(obj: GeneratorResult) -> List[str]:
    """
    Lightweight validation for GeneratorResult.
    """
    errors: List[str] = []

    if not isinstance(obj, GeneratorResult):
        return ["Generator result must be a GeneratorResult instance."]

    required_text_fields = [
        ("analysis", obj.analysis),
        ("selected_candidate_pattern", obj.selected_candidate_pattern),
        ("selected_candidate_target", obj.selected_candidate_target),
        ("applied_changes_summary", obj.applied_changes_summary),
        ("final_code", obj.final_code),
    ]
    for field_name, value in required_text_fields:
        if not isinstance(value, str):
            errors.append(f"'{field_name}' must be a string.")

    list_fields = [
        ("correctness_risks", obj.correctness_risks),
        ("expected_metrics", obj.expected_metrics),
    ]
    for field_name, value in list_fields:
        if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
            errors.append(f"'{field_name}' must be list[str].")

    if not isinstance(obj.compile_ready, bool):
        errors.append("'compile_ready' must be bool.")
    if not isinstance(obj.used_feedback, bool):
        errors.append("'used_feedback' must be bool.")
    if not isinstance(obj.generation_succeeded, bool):
        errors.append("'generation_succeeded' must be bool.")

    if obj.failure_reason is not None and not isinstance(obj.failure_reason, str):
        errors.append("'failure_reason' must be str | None.")

    return errors