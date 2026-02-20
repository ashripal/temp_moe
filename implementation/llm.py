from __future__ import annotations
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Protocol


@dataclass
class LLMMessage:
    role: str
    content: str


class LLMClient(Protocol):
    def complete(self, messages: List[LLMMessage]) -> str:
        ...


def _extract_allowed_patterns(prompt: str) -> List[str]:
    m = re.search(r"ALLOWED_PATTERNS_JSON=(\[[\s\S]*?\])", prompt)
    if not m:
        return []
    try:
        return json.loads(m.group(1))
    except Exception:
        return []


def _pick_pattern(allowed: List[str], prefer_keywords: List[str]) -> str:
    allowed_l = [(p, p.lower()) for p in allowed]
    for kw in prefer_keywords:
        kw = kw.lower()
        for p, pl in allowed_l:
            if kw in pl:
                return p
    return allowed[0] if allowed else "UNKNOWN_PATTERN"


class MockLLM:
    """
    Deterministic offline LLM for testing.
    It always returns patterns that exist in the provided catalog (allowed list in the prompt).
    """
    def complete(self, messages: List[LLMMessage]) -> str:
        prompt = "\n".join(m.content for m in messages)
        allowed = _extract_allowed_patterns(prompt)

        if "Parallelism & Job Expert" in prompt:
            pattern = _pick_pattern(
                allowed,
                prefer_keywords=["openmp", "thread", "schedule", "parallel", "collapse", "reduction", "numa", "rank"]
            )
            return json.dumps([{
                "pattern": pattern,
                "target": "foo.c:bar():loop_12",
                "rationale": "Telemetry suggests parallel configuration is a plausible lever (mock).",
                "action_sketch": "Apply the selected catalog pattern conservatively; tune parameters if applicable.",
                "preconditions": ["No semantic changes", "Preserve correctness checks"],
                "parameters_to_sweep": {"variant": [1, 2]},
                "correctness_checks": ["Run regression tests", "Compare key outputs within tolerance"],
                "performance_metrics": ["runtime", "scaling_efficiency"],
                "risk_level": "low",
                "rollback_criteria": ["Correctness failure", "Runtime regression > 3%"]
            }])

        if "Communication & Resilience Expert" in prompt:
            pattern = _pick_pattern(
                allowed,
                prefer_keywords=["mpi", "message", "async", "communication", "checkpoint", "cache"]
            )
            return json.dumps([{
                "pattern": pattern,
                "target": "mpi_region:exchange_halos",
                "rationale": "MPI / comm-related issues suspected from telemetry (mock).",
                "action_sketch": "Apply the selected catalog pattern conservatively; prefer template-based edits.",
                "preconditions": ["Preserve ordering/semantics", "Validate correctness after change"],
                "parameters_to_sweep": {"variant": [1, 2]},
                "correctness_checks": ["Run regression tests", "Compare outputs within tolerance"],
                "performance_metrics": ["runtime", "mpi_wait_pct", "scaling_efficiency"],
                "risk_level": "medium",
                "rollback_criteria": ["Correctness failure", "Hang/deadlock", "Runtime regression > 3%"]
            }])

        if "Kernel & System Efficiency Expert" in prompt:
            pattern = _pick_pattern(
                allowed,
                prefer_keywords=["loop", "vector", "unroll", "locality", "math", "precision", "frequency", "power"]
            )
            return json.dumps([{
                "pattern": pattern,
                "target": "kernel.c:matmul():loop_3",
                "rationale": "Kernel hotspot suggests local optimization could help (mock).",
                "action_sketch": "Apply the selected catalog pattern conservatively with small parameter sweep.",
                "preconditions": ["No out-of-bounds", "Validate numeric tolerance if FP changes occur"],
                "parameters_to_sweep": {"variant": [1, 2]},
                "correctness_checks": ["Run regression tests", "Numeric tolerance check"],
                "performance_metrics": ["runtime", "kernel_time_share"],
                "risk_level": "low",
                "rollback_criteria": ["Correctness failure", "Runtime regression > 3%"]
            }])

        return "[]"