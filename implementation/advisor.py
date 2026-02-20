from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .experts import (
    ExpertContext,
    ParallelismJobExpert,
    CommunicationResilienceExpert,
    KernelSystemEfficiencyExpert,
)
from .kb import KnowledgeBase
from .llm import LLMClient
from .router import SimpleTelemetryRouter, RoutingDecision
from .schema import ExpertOutput


@dataclass
class AdvisorResult:
    routing: RoutingDecision
    expert_outputs: List[ExpertOutput]
    final_ranked_candidates: List[Dict[str, Any]]  # simple dict list for Generator stage


class MoEAdvisor:
    def __init__(self, llm: LLMClient, kb: KnowledgeBase, prompts_dir: str | Path):
        self.llm = llm
        self.kb = kb
        self.router = SimpleTelemetryRouter()

        self.expert_map = {
            "Parallelism & Job Expert": ParallelismJobExpert(llm, kb, prompts_dir),
            "Communication & Resilience Expert": CommunicationResilienceExpert(llm, kb, prompts_dir),
            "Kernel & System Efficiency Expert": KernelSystemEfficiencyExpert(llm, kb, prompts_dir),
        }

        env = Environment(
            loader=FileSystemLoader(str(prompts_dir)),
            autoescape=select_autoescape(disabled_extensions=("jinja",))
        )
        self.aggregator_template = env.get_template("aggregator.jinja")

    def run(
        self,
        code_snippets: str,
        profiling_summary: str,
        telemetry_summary: str,
        telemetry_struct: Dict[str, float],
    ) -> AdvisorResult:
        routing = self.router.route(telemetry_struct)

        # Retrieve patterns per expert (simple hints for now)
        retrieved = self.kb.retrieve_by_category_hint(" ".join(routing.selected_experts), limit=10)
        # retrieved_patterns = [
        #     {"name": p.name, "category": p.category, "description": p.description, "preconditions": p.preconditions}
        #     for p in retrieved
        # ]
        retrieved_patterns = [
            {
                "name": p.name,
                "category": p.category,
                "description": p.description,
                "example": getattr(p, "example", None),
                "optimized_metrics": getattr(p, "optimized_metrics", None),
                "detection": getattr(p, "detection", None),
            }
            for p in retrieved
        ]

        ctx = ExpertContext(
            code_snippets=code_snippets,
            profiling_summary=profiling_summary,
            telemetry_summary=telemetry_summary,
            retrieved_patterns=retrieved_patterns,
        )

        expert_outputs: List[ExpertOutput] = []
        for expert_name in routing.selected_experts:
            expert_outputs.append(self.expert_map[expert_name].propose(ctx))

        # Simple “aggregator”: concatenate + keep top-N by low risk first (no extra LLM)
        all_candidates: List[Dict[str, Any]] = []
        for eo in expert_outputs:
            for c in eo.candidates:
                d = c.to_dict()
                d["proposed_by"] = eo.expert_name
                all_candidates.append(d)

        # Sort: low risk first, then keep first 3
        risk_rank = {"low": 0, "medium": 1, "high": 2}
        all_candidates.sort(key=lambda d: risk_rank.get(d.get("risk_level", "high"), 2))

        final_ranked = all_candidates[:3]

        return AdvisorResult(
            routing=routing,
            expert_outputs=expert_outputs,
            final_ranked_candidates=final_ranked,
        )