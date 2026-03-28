from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .kb import KnowledgeBase
from .llm import LLMClient, LLMMessage
from .schema import CandidateAction, ExpertOutput, validate_candidate_dict


@dataclass
class ExpertContext:
    code_snippets: str
    profiling_summary: str
    telemetry_summary: str
    retrieved_patterns: List[Dict[str, Any]]


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    fenced_json = re.match(
        r"^```(?:json)?\s*([\s\S]*?)\s*```$",
        text,
        flags=re.IGNORECASE,
    )
    if fenced_json:
        return fenced_json.group(1).strip()
    return text


def _tokenize_for_match(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _score_pattern_match(candidate_text: str, allowed_pattern: str) -> int:
    """
    Lightweight lexical matcher used to repair model outputs that drift outside
    the retrieved subset. Higher is better.
    """
    cand_tokens = set(_tokenize_for_match(candidate_text))
    patt_tokens = set(_tokenize_for_match(allowed_pattern))

    if not cand_tokens or not patt_tokens:
        return 0

    overlap = len(cand_tokens & patt_tokens)

    # small bonus for substring containment
    cand_lower = candidate_text.lower()
    patt_lower = allowed_pattern.lower()
    containment_bonus = 2 if (cand_lower in patt_lower or patt_lower in cand_lower) else 0

    return overlap + containment_bonus


class BaseExpert:
    name: str
    template_file: str

    def __init__(self, llm: LLMClient, kb: KnowledgeBase, prompts_dir: str | Path):
        self.llm = llm
        self.kb = kb

        env = Environment(
            loader=FileSystemLoader(str(prompts_dir)),
            autoescape=select_autoescape(disabled_extensions=("jinja",)),
        )
        self.template = env.get_template(self.template_file)

    def _build_allowed_patterns(self, ctx: ExpertContext) -> List[str]:
        allowed_patterns = sorted(
            {
                str(p["name"]).strip()
                for p in ctx.retrieved_patterns
                if isinstance(p, dict) and p.get("name")
            }
        )
        if allowed_patterns:
            return allowed_patterns
        return sorted(self.kb.allowed_patterns())

    def _build_prompt(self, ctx: ExpertContext, allowed_patterns: List[str]) -> str:
        return self.template.render(
            expert_name=self.name,
            code_snippets=ctx.code_snippets,
            profiling_summary=ctx.profiling_summary,
            telemetry_summary=ctx.telemetry_summary,
            retrieved_patterns=ctx.retrieved_patterns,
            allowed_patterns=allowed_patterns,
        )

    def _parse_llm_output(self, raw: str) -> List[Dict[str, Any]]:
        cleaned = _strip_code_fences(raw)
        try:
            parsed = json.loads(cleaned)
        except Exception as exc:
            raise ValueError(
                f"{self.name} returned invalid JSON: {exc}\n"
                f"Raw response:\n{raw}\n\n"
                f"Cleaned response:\n{cleaned}"
            ) from exc

        if not isinstance(parsed, list):
            raise ValueError(
                f"{self.name} output must be a JSON list of candidate dicts.\n"
                f"Parsed type: {type(parsed).__name__}\n"
                f"Raw response:\n{raw}\n\n"
                f"Cleaned response:\n{cleaned}"
            )

        normalized: List[Dict[str, Any]] = []
        for i, item in enumerate(parsed):
            if not isinstance(item, dict):
                raise ValueError(
                    f"{self.name} candidate {i} is not a dict. "
                    f"Got: {type(item).__name__} -> {item}"
                )
            normalized.append(item)

        return normalized

    def _repair_pattern_to_allowed_subset(
        self,
        pattern_raw: str,
        candidate_dict: Dict[str, Any],
        allowed_patterns: List[str],
    ) -> Tuple[str, bool]:
        """
        If the model proposes a catalog-valid pattern outside the retrieved subset,
        map it deterministically to the closest allowed pattern instead of failing.

        Returns:
            (resolved_pattern, was_repaired)
        """
        canonical = self.kb.canonical_pattern(pattern_raw)
        if canonical is None:
            return pattern_raw, False

        if canonical in allowed_patterns:
            return canonical, False

        if not allowed_patterns:
            return canonical, False

        repair_text = " ".join(
            [
                canonical,
                str(candidate_dict.get("rationale", "")),
                str(candidate_dict.get("action_sketch", "")),
                " ".join(str(x) for x in candidate_dict.get("performance_metrics", [])),
            ]
        )

        best_pattern = None
        best_score = -1
        for allowed in allowed_patterns:
            score = _score_pattern_match(repair_text, allowed)
            if score > best_score:
                best_score = score
                best_pattern = allowed

        if best_pattern is None:
            return canonical, False

        return best_pattern, True

    def propose(self, ctx: ExpertContext) -> ExpertOutput:
        allowed_patterns = self._build_allowed_patterns(ctx)
        prompt = self._build_prompt(ctx, allowed_patterns)

        raw = self.llm.complete(
            [
                LLMMessage(
                    role="system",
                    content=(
                        f"You are {self.name}. "
                        "Use the retrieved catalog patterns as your primary evidence. "
                        "Choose only from ALLOWED_PATTERNS_JSON whenever it is non-empty. "
                        "Return only a valid JSON array of candidate action objects. "
                        "Do not include markdown fences, explanations, or extra text."
                    ),
                ),
                LLMMessage(role="user", content=prompt),
            ]
        )

        parsed = self._parse_llm_output(raw)

        candidates: List[CandidateAction] = []

        for i, d in enumerate(parsed):
            errors = validate_candidate_dict(d)
            if errors:
                raise ValueError(
                    f"{self.name} candidate {i} schema errors: {errors}\nCandidate:\n{d}"
                )

            pattern_raw = d["pattern"].strip()
            pattern = self.kb.canonical_pattern(pattern_raw)
            if pattern is None:
                raise ValueError(
                    f"{self.name} proposed pattern '{pattern_raw}' not recognized in catalog."
                )

            candidates.append(
                CandidateAction(
                    pattern=pattern,
                    target=d["target"].strip(),
                    rationale=d["rationale"].strip(),
                    action_sketch=d["action_sketch"].strip(),
                    preconditions=[str(x).strip() for x in d["preconditions"]],
                    parameters_to_sweep=d["parameters_to_sweep"],
                    correctness_checks=[str(x).strip() for x in d["correctness_checks"]],
                    performance_metrics=[str(x).strip() for x in d["performance_metrics"]],
                    risk_level=d["risk_level"],
                    rollback_criteria=[str(x).strip() for x in d["rollback_criteria"]],
                )
            )

        return ExpertOutput(expert_name=self.name, candidates=candidates)


class ParallelismJobExpert(BaseExpert):
    name = "Parallelism & Job Expert"
    template_file = "parallelism_job_expert.jinja"


class CommunicationResilienceExpert(BaseExpert):
    name = "Communication & Resilience Expert"
    template_file = "communication_resilience_expert.jinja"


class KernelSystemEfficiencyExpert(BaseExpert):
    name = "Kernel & System Efficiency Expert"
    template_file = "kernel_system_efficiency_expert.jinja"