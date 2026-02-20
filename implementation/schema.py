from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Literal, Optional


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
    Lightweight schema validation (kept dependency-free).
    Returns list of errors; empty => valid.
    """
    errors: List[str] = []

    required_str_fields = ["pattern", "target", "rationale", "action_sketch", "risk_level"]
    for f in required_str_fields:
        if f not in d or not isinstance(d[f], str) or not d[f].strip():
            errors.append(f"Missing/invalid '{f}' (must be non-empty string).")

    if "preconditions" not in d or not isinstance(d["preconditions"], list) or not all(isinstance(x, str) for x in d["preconditions"]):
        errors.append("Missing/invalid 'preconditions' (must be list[str]).")

    if "parameters_to_sweep" not in d or not isinstance(d["parameters_to_sweep"], dict):
        errors.append("Missing/invalid 'parameters_to_sweep' (must be dict[str, list]).")
    else:
        for k, v in d["parameters_to_sweep"].items():
            if not isinstance(k, str) or not isinstance(v, list):
                errors.append("Invalid 'parameters_to_sweep' entry (must be dict[str, list]).")
                break

    for lf in ["correctness_checks", "performance_metrics", "rollback_criteria"]:
        if lf not in d or not isinstance(d[lf], list) or not all(isinstance(x, str) for x in d[lf]):
            errors.append(f"Missing/invalid '{lf}' (must be list[str]).")

    if d.get("risk_level") not in ("low", "medium", "high"):
        errors.append("Invalid 'risk_level' (must be 'low'|'medium'|'high').")

    return errors