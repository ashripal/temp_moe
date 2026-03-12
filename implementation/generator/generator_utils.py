from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from ..schema import validate_candidate_dict
from .generator_schema import GeneratorInput, GeneratorResult


def select_candidate(
    ranked_candidates: List[Dict[str, Any]],
    index: int = 0,
) -> Optional[Dict[str, Any]]:
    """
    Select a single valid candidate from the advisor-ranked list.

    v1 policy:
      - use exactly one candidate
      - default to top-ranked candidate
      - if the requested index is invalid, fall back to index 0
      - if the chosen candidate is invalid, try the first valid candidate
    """
    if not ranked_candidates:
        return None

    if index < 0 or index >= len(ranked_candidates):
        index = 0

    preferred = ranked_candidates[index]
    if isinstance(preferred, dict) and not validate_candidate_dict(preferred):
        return preferred

    for candidate in ranked_candidates:
        if isinstance(candidate, dict) and not validate_candidate_dict(candidate):
            return candidate

    return None


def candidate_brief(candidate: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Produce a compact prompt-friendly representation of a candidate.
    """
    if not candidate:
        return {}

    return {
        "pattern": str(candidate.get("pattern", "")),
        "target": str(candidate.get("target", "")),
        "rationale": str(candidate.get("rationale", "")),
        "action_sketch": str(candidate.get("action_sketch", "")),
        "preconditions": _stringify_list(candidate.get("preconditions", [])),
        "parameters_to_sweep": _stringify_list(candidate.get("parameters_to_sweep", [])),
        "correctness_checks": _stringify_list(candidate.get("correctness_checks", [])),
        "performance_metrics": _stringify_list(candidate.get("performance_metrics", [])),
        "risk_level": str(candidate.get("risk_level", "")),
        "rollback_criteria": str(candidate.get("rollback_criteria", "")),
    }


def format_candidate_for_prompt(
    candidate: Optional[Dict[str, Any]],
    heading: str = "Selected Candidate",
) -> str:
    """
    Render one candidate into a readable prompt block.
    """
    if not candidate:
        return f"{heading}:\n- None"

    brief = candidate_brief(candidate)

    sections = [
        f"{heading}:",
        f"- Pattern: {brief['pattern']}",
        f"- Target: {brief['target']}",
        f"- Rationale: {brief['rationale']}",
        f"- Action Sketch: {brief['action_sketch']}",
        f"- Preconditions: {format_string_list(brief['preconditions'])}",
        f"- Parameters to Sweep: {format_string_list(brief['parameters_to_sweep'])}",
        f"- Correctness Checks: {format_string_list(brief['correctness_checks'])}",
        f"- Performance Metrics: {format_string_list(brief['performance_metrics'])}",
        f"- Risk Level: {brief['risk_level']}",
        f"- Rollback Criteria: {brief['rollback_criteria']}",
    ]
    return "\n".join(sections)


def format_ranked_candidates_for_prompt(
    ranked_candidates: List[Dict[str, Any]],
    max_candidates: int = 3,
) -> str:
    """
    Render the top-N ranked candidates into a readable prompt block.
    """
    if not ranked_candidates:
        return "Ranked Candidates:\n- None"

    rendered: List[str] = ["Ranked Candidates:"]
    count = 0

    for idx, candidate in enumerate(ranked_candidates):
        if count >= max_candidates:
            break
        if not isinstance(candidate, dict):
            continue

        rendered.append(
            format_candidate_for_prompt(
                candidate,
                heading=f"Candidate Rank {idx + 1}",
            )
        )
        count += 1
        if count < max_candidates and idx < len(ranked_candidates) - 1:
            rendered.append("")

    if count == 0:
        return "Ranked Candidates:\n- None"

    return "\n".join(rendered)


def build_prompt_payload(generator_input: GeneratorInput) -> Dict[str, Any]:
    """
    Convert GeneratorInput into a prompt-rendering payload for Jinja templates.
    """
    selected_candidate = (
        generator_input.selected_candidate
        if generator_input.selected_candidate is not None
        else select_candidate(generator_input.ranked_candidates, index=0)
    )

    feedback_text = generator_input.resolved_feedback_text()

    return {
        "original_code": generator_input.original_code,
        "profiling_summary": generator_input.profiling_summary,
        "telemetry_summary": generator_input.telemetry_summary,
        "telemetry_struct": generator_input.telemetry_struct,
        "selected_candidate": selected_candidate,
        "selected_candidate_text": format_candidate_for_prompt(
            selected_candidate,
            heading="Selected Candidate",
        ),
        "ranked_candidates": generator_input.ranked_candidates,
        "ranked_candidates_text": format_ranked_candidates_for_prompt(
            generator_input.ranked_candidates
        ),
        "ast": generator_input.ast or "",
        "flame_report": generator_input.flame_report or "",
        "evaluator_feedback": feedback_text or "",
        "has_feedback": bool(feedback_text and feedback_text.strip()),
    }


def sanitize_generated_text(text: str) -> str:
    """
    Normalize raw model text before parsing.

    This does not remove content aggressively; it only cleans trivial wrapping.
    """
    if not text:
        return ""

    cleaned = text.strip()
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    return cleaned


def strip_code_fences(text: str) -> str:
    """
    Remove surrounding markdown code fences if the model wrapped the output.
    """
    if not text:
        return ""

    stripped = text.strip()

    fenced_match = re.match(r"^```[a-zA-Z0-9_-]*\n(.*)\n```$", stripped, flags=re.DOTALL)
    if fenced_match:
        return fenced_match.group(1).strip()

    return stripped


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Best-effort extraction of the first JSON object from raw model output.

    Strategy:
      1. try parsing the whole text as JSON
      2. search for the first balanced {...} region and parse it
    """
    if not text or not text.strip():
        return None

    stripped = strip_code_fences(sanitize_generated_text(text))

    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    candidate = _find_first_balanced_json_object(stripped)
    if candidate is None:
        return None

    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        return None

    return None


def extract_final_code(text: str) -> str:
    """
    Best-effort extraction of final_code from raw model output.

    Preferred path:
      - parse JSON and return `final_code`

    Fallback paths:
      - detect sections like FINAL_CODE:
      - return stripped raw text if nothing else works
    """
    if not text:
        return ""

    parsed = extract_json_object(text)
    if parsed and isinstance(parsed.get("final_code"), str):
        return strip_code_fences(parsed["final_code"]).strip()

    patterns = [
        r"final_code\s*[:=]\s*(.*)",
        r"FINAL_CODE\s*[:=]\s*(.*)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.DOTALL)
        if match:
            return strip_code_fences(match.group(1)).strip()

    return strip_code_fences(text).strip()


def build_failure_result(
    original_code: str,
    selected_candidate: Optional[Dict[str, Any]],
    failure_reason: str,
    used_feedback: bool = False,
) -> GeneratorResult:
    """
    Create a safe unchanged-code fallback result.
    """
    pattern = "UNKNOWN_PATTERN"
    target = "UNKNOWN_TARGET"
    expected_metrics: List[str] = []
    correctness_risks = ["No transformation applied; original code preserved."]

    if selected_candidate:
        pattern = str(selected_candidate.get("pattern", pattern))
        target = str(selected_candidate.get("target", target))

        raw_metrics = selected_candidate.get("performance_metrics", [])
        if isinstance(raw_metrics, list):
            expected_metrics = [str(x) for x in raw_metrics]

        raw_checks = selected_candidate.get("correctness_checks", [])
        if isinstance(raw_checks, list) and raw_checks:
            correctness_risks = [
                "Generator returned unchanged code due to failure or unsafe rewrite."
            ] + [str(x) for x in raw_checks]

    return GeneratorResult(
        analysis=(
            "Generator did not apply a rewrite and returned the original code "
            "to preserve correctness."
        ),
        selected_candidate_pattern=pattern,
        selected_candidate_target=target,
        applied_changes_summary="No changes applied; original code returned unchanged.",
        final_code=original_code,
        correctness_risks=correctness_risks,
        expected_metrics=expected_metrics,
        compile_ready=False,
        used_feedback=used_feedback,
        generation_succeeded=False,
        failure_reason=failure_reason,
    )


def normalize_generator_response(
    response_dict: Dict[str, Any],
    selected_candidate: Optional[Dict[str, Any]],
    used_feedback: bool,
) -> GeneratorResult:
    """
    Convert a parsed model response dictionary into GeneratorResult.

    Missing fields are filled conservatively from the selected candidate when
    possible.
    """
    fallback_pattern = "UNKNOWN_PATTERN"
    fallback_target = "UNKNOWN_TARGET"
    fallback_metrics: List[str] = []

    if selected_candidate:
        fallback_pattern = str(selected_candidate.get("pattern", fallback_pattern))
        fallback_target = str(selected_candidate.get("target", fallback_target))
        raw_metrics = selected_candidate.get("performance_metrics", [])
        if isinstance(raw_metrics, list):
            fallback_metrics = [str(x) for x in raw_metrics]

    correctness_risks = response_dict.get("correctness_risks", [])
    if not isinstance(correctness_risks, list):
        correctness_risks = [str(correctness_risks)]

    expected_metrics = response_dict.get("expected_metrics", fallback_metrics)
    if not isinstance(expected_metrics, list):
        expected_metrics = [str(expected_metrics)]

    final_code = response_dict.get("final_code", "")
    if not isinstance(final_code, str):
        final_code = str(final_code)

    compile_ready = response_dict.get("compile_ready", True)
    if not isinstance(compile_ready, bool):
        compile_ready = True

    return GeneratorResult(
        analysis=str(response_dict.get("analysis", "")),
        selected_candidate_pattern=str(
            response_dict.get(
                "selected_candidate_pattern",
                fallback_pattern,
            )
        ),
        selected_candidate_target=str(
            response_dict.get(
                "selected_candidate_target",
                fallback_target,
            )
        ),
        applied_changes_summary=str(
            response_dict.get("applied_changes_summary", "")
        ),
        final_code=strip_code_fences(final_code).strip(),
        correctness_risks=[str(x) for x in correctness_risks],
        expected_metrics=[str(x) for x in expected_metrics],
        compile_ready=compile_ready,
        used_feedback=used_feedback,
        generation_succeeded=bool(strip_code_fences(final_code).strip()),
        failure_reason=None,
    )


def format_string_list(items: Any) -> str:
    """
    Format a list of strings compactly for prompts.
    """
    if items is None:
        return "[]"

    if isinstance(items, list):
        return "[" + ", ".join(str(x) for x in items) + "]"

    return str(items)


def validate_ranked_candidates(
    ranked_candidates: List[Dict[str, Any]],
) -> List[str]:
    """
    Validate the advisor handoff candidate list.
    """
    errors: List[str] = []

    if not isinstance(ranked_candidates, list):
        return ["Ranked candidates must be a list of candidate dictionaries."]

    if not ranked_candidates:
        return ["Ranked candidates list is empty."]

    for idx, candidate in enumerate(ranked_candidates):
        if not isinstance(candidate, dict):
            errors.append(f"Candidate at index {idx} is not a dictionary.")
            continue

        candidate_errors = validate_candidate_dict(candidate)
        if candidate_errors:
            errors.append(
                f"Candidate {idx} failed validation: {'; '.join(candidate_errors)}"
            )

    return errors


def prepare_retry_input(
    previous_input: GeneratorInput,
    evaluator_feedback: str,
) -> GeneratorInput:
    """
    Build a retry input while preserving the original selected candidate.
    """
    return GeneratorInput(
        original_code=previous_input.original_code,
        profiling_summary=previous_input.profiling_summary,
        telemetry_summary=previous_input.telemetry_summary,
        telemetry_struct=previous_input.telemetry_struct,
        ranked_candidates=previous_input.ranked_candidates,
        selected_candidate=previous_input.selected_candidate,
        ast=previous_input.ast,
        flame_report=previous_input.flame_report,
        evaluator_feedback=evaluator_feedback,
        evaluation_feedback=None,
    )


def split_model_response(
    text: str,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Convenience helper that returns both parsed JSON (if any) and extracted code.
    """
    parsed = extract_json_object(text)
    code = extract_final_code(text)
    return parsed, code


def _stringify_list(value: Any) -> List[str]:
    """
    Normalize value into list[str] for candidate prompt summaries.
    """
    if value is None:
        return []

    if isinstance(value, list):
        return [str(x) for x in value]

    return [str(value)]


def _find_first_balanced_json_object(text: str) -> Optional[str]:
    """
    Find the first balanced JSON object in text, ignoring braces inside strings.
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for idx in range(start, len(text)):
        ch = text[idx]

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:idx + 1]

    return None