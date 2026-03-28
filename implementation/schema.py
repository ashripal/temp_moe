from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal


RiskLevel = Literal["low", "medium", "high"]


@dataclass
class CandidateAction:
    pattern: str
    target: str
    rationale: str
    action_sketch: str
    preconditions: List[str]
    parameters_to_sweep: Dict[str, List[Any]]
    correctness_checks: List[str]
    performance_metrics: List[str]
    risk_level: RiskLevel
    rollback_criteria: List[str]

    # Optional metadata added later in the advisor/ranking stage.
    # Defaults keep expert outputs backward-compatible.
    catalog_score: int = 0
    telemetry_alignment: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExpertOutput:
    expert_name: str
    candidates: List[CandidateAction]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expert_name": self.expert_name,
            "candidates": [c.to_dict() for c in self.candidates],
        }


def validate_candidate_dict(d: Dict[str, Any]) -> List[str]:
    """
    Lightweight schema validation.
    Returns a list of validation errors; empty list means valid.

    The core expert output fields are required.
    Advisor-added enrichment fields such as catalog_score and
    telemetry_alignment are optional.
    """
    errors: List[str] = []

    required_str_fields = [
        "pattern",
        "target",
        "rationale",
        "action_sketch",
        "risk_level",
    ]
    for field_name in required_str_fields:
        if field_name not in d or not isinstance(d[field_name], str) or not d[field_name].strip():
            errors.append(f"Missing/invalid '{field_name}' (must be non-empty string).")

    if (
        "preconditions" not in d
        or not isinstance(d["preconditions"], list)
        or not all(isinstance(x, str) for x in d["preconditions"])
    ):
        errors.append("Missing/invalid 'preconditions' (must be list[str]).")

    if "parameters_to_sweep" not in d or not isinstance(d["parameters_to_sweep"], dict):
        errors.append("Missing/invalid 'parameters_to_sweep' (must be dict[str, list]).")
    else:
        for key, value in d["parameters_to_sweep"].items():
            if not isinstance(key, str) or not isinstance(value, list):
                errors.append("Invalid 'parameters_to_sweep' entry (must be dict[str, list]).")
                break

    for list_field in ["correctness_checks", "performance_metrics", "rollback_criteria"]:
        if (
            list_field not in d
            or not isinstance(d[list_field], list)
            or not all(isinstance(x, str) for x in d[list_field])
        ):
            errors.append(f"Missing/invalid '{list_field}' (must be list[str]).")

    if d.get("risk_level") not in ("low", "medium", "high"):
        errors.append("Invalid 'risk_level' (must be 'low'|'medium'|'high').")

    # Optional advisor-added fields
    if "catalog_score" in d and not isinstance(d["catalog_score"], int):
        errors.append("Invalid 'catalog_score' (must be int when present).")

    if "telemetry_alignment" in d:
        if not isinstance(d["telemetry_alignment"], list) or not all(
            isinstance(x, str) for x in d["telemetry_alignment"]
        ):
            errors.append("Invalid 'telemetry_alignment' (must be list[str] when present).")

    return errors