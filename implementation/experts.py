from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .kb import KnowledgeBase
from .llm import LLMClient, LLMMessage
from .schema import CandidateAction, ExpertOutput, validate_candidate_dict


@dataclass
class ExpertContext:
    code_snippets: str
    profiling_summary: str
    telemetry_summary: str
    retrieved_patterns: List[Dict[str, Any]]  # small excerpts


class BaseExpert:
    name: str
    template_file: str

    def __init__(self, llm: LLMClient, kb: KnowledgeBase, prompts_dir: str | Path):
        self.llm = llm
        self.kb = kb
        env = Environment(
            loader=FileSystemLoader(str(prompts_dir)),
            autoescape=select_autoescape(disabled_extensions=("jinja",))
        )
        self.template = env.get_template(self.template_file)

    def propose(self, ctx: ExpertContext) -> ExpertOutput:
        prompt = self.template.render(
            expert_name=self.name,
            code_snippets=ctx.code_snippets,
            profiling_summary=ctx.profiling_summary,
            telemetry_summary=ctx.telemetry_summary,
            retrieved_patterns=ctx.retrieved_patterns,
            allowed_patterns=sorted(self.kb.allowed_patterns()),
        )

        raw = self.llm.complete([
            LLMMessage(role="system", content=f"You are {self.name}."),
            LLMMessage(role="user", content=prompt),
        ])

        try:
            parsed = json.loads(raw)
            if not isinstance(parsed, list):
                raise ValueError("Expert output must be a JSON list of candidate dicts.")
        except Exception as e:
            raise ValueError(f"{self.name} returned invalid JSON: {e}\nRaw:\n{raw}")

        candidates: List[CandidateAction] = []
        for i, d in enumerate(parsed):
            if not isinstance(d, dict):
                raise ValueError(f"{self.name} candidate {i} is not a dict.")
            errors = validate_candidate_dict(d)
            if errors:
                raise ValueError(f"{self.name} candidate {i} schema errors: {errors}\nCandidate:\n{d}")

            # pattern = d["pattern"].strip()
            # if pattern not in self.kb.allowed_patterns():
            #     raise ValueError(f"{self.name} proposed pattern not in catalog: '{pattern}'")

            pattern_raw = d["pattern"].strip()
            pattern = self.kb.canonical_pattern(pattern_raw)
            if pattern is None:
                raise ValueError(f"{self.name} proposed pattern '{pattern_raw}' not recognized in catalog.")

            candidates.append(
                CandidateAction(
                    pattern=pattern,
                    target=d["target"].strip(),
                    rationale=d["rationale"].strip(),
                    action_sketch=d["action_sketch"].strip(),
                    preconditions=[x.strip() for x in d["preconditions"]],
                    parameters_to_sweep=d["parameters_to_sweep"],
                    correctness_checks=[x.strip() for x in d["correctness_checks"]],
                    performance_metrics=[x.strip() for x in d["performance_metrics"]],
                    risk_level=d["risk_level"],
                    rollback_criteria=[x.strip() for x in d["rollback_criteria"]],
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