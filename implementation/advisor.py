from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .experts import (
    CommunicationResilienceExpert,
    ExpertContext,
    KernelSystemEfficiencyExpert,
    ParallelismJobExpert,
)
from .kb import KnowledgeBase, Pattern
from .llm import LLMClient
from .router import RoutingDecision, SimpleTelemetryRouter
from .schema import ExpertOutput


@dataclass
class AdvisorResult:
    routing: RoutingDecision
    expert_outputs: List[ExpertOutput]
    final_ranked_candidates: List[Dict[str, Any]]


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
            autoescape=select_autoescape(disabled_extensions=("jinja",)),
        )
        self.aggregator_template = env.get_template("aggregator.jinja")

    def _pattern_to_prompt_dict(self, pattern: Pattern) -> Dict[str, Any]:
        return {
            "name": pattern.name,
            "category": pattern.category,
            "description": pattern.description,
            "example": pattern.example,
            "optimized_metrics": pattern.optimized_metrics,
            "detection": pattern.detection,
            "expert_family": getattr(pattern, "expert_family", None),
            "metric_tags": list(getattr(pattern, "metric_tags", ()) or ()),
            "detection_tags": list(getattr(pattern, "detection_tags", ()) or ()),
        }

    def _catalog_score_for_candidate(
        self,
        candidate_dict: Dict[str, Any],
        telemetry_struct: Dict[str, float],
    ) -> int:
        """
        Score a proposed candidate based on how well it aligns with:
        1. the catalog metadata for the selected pattern
        2. the current telemetry signature
        """
        pattern_name = candidate_dict.get("pattern", "")
        if not pattern_name:
            return 0

        canonical = self.kb.canonical_pattern(pattern_name)
        if canonical is None:
            return 0

        pattern = self.kb.get(canonical)
        metric_tags = set(getattr(pattern, "metric_tags", ()) or ())
        detection_tags = set(getattr(pattern, "detection_tags", ()) or ())
        perf_metrics = {str(x).lower() for x in candidate_dict.get("performance_metrics", [])}

        score = 0

        # Base grounding score: candidate exists in catalog
        score += 2

        # Detection alignment with telemetry
        if telemetry_struct.get("mpi_wait_pct", 0.0) >= 25.0 and "mpi_wait" in detection_tags:
            score += 5
        if telemetry_struct.get("omp_barrier_pct", 0.0) >= 15.0 and "omp_barrier" in detection_tags:
            score += 5
        if telemetry_struct.get("omp_imbalance_ratio", 1.0) >= 1.5 and "omp_imbalance" in detection_tags:
            score += 5
        if telemetry_struct.get("memory_bound_score", 0.0) >= 0.6 and "memory_bound" in detection_tags:
            score += 5

        # Candidate's stated performance metrics aligned to telemetry
        if telemetry_struct.get("mpi_wait_pct", 0.0) >= 25.0:
            if "mpi_wait_pct" in perf_metrics or "throughput" in perf_metrics:
                score += 2

        if telemetry_struct.get("omp_barrier_pct", 0.0) >= 15.0:
            if "scaling_efficiency" in perf_metrics or "omp_barrier_pct" in perf_metrics:
                score += 2

        if telemetry_struct.get("omp_imbalance_ratio", 1.0) >= 1.5:
            if "scaling_efficiency" in perf_metrics or "omp_imbalance_ratio" in perf_metrics:
                score += 2

        if telemetry_struct.get("memory_bound_score", 0.0) >= 0.6:
            if any(("cache" in m or "memory" in m) for m in perf_metrics):
                score += 2

        # Pattern metadata can still add a small boost even if candidate metrics are generic
        if telemetry_struct.get("memory_bound_score", 0.0) >= 0.6:
            if "cache" in metric_tags or "memory" in metric_tags or "memory_bound_score" in metric_tags:
                score += 1

        if telemetry_struct.get("mpi_wait_pct", 0.0) >= 25.0:
            if "mpi_wait_pct" in metric_tags or "throughput" in metric_tags:
                score += 1

        if telemetry_struct.get("omp_barrier_pct", 0.0) >= 15.0:
            if "omp_barrier_pct" in metric_tags or "scaling_efficiency" in metric_tags:
                score += 1

        if telemetry_struct.get("omp_imbalance_ratio", 1.0) >= 1.5:
            if "omp_imbalance_ratio" in metric_tags or "scaling_efficiency" in metric_tags:
                score += 1

        return score

    def _candidate_sort_key(
        self,
        candidate_dict: Dict[str, Any],
        telemetry_struct: Dict[str, float],
    ) -> tuple:
        """
        Lower tuple sorts earlier.
        Priority:
        1. lower risk
        2. higher catalog_score
        3. higher telemetry bonus from stated performance metrics
        """
        risk_rank = {"low": 0, "medium": 1, "high": 2}
        perf_metrics = [str(m).lower() for m in candidate_dict.get("performance_metrics", [])]

        telemetry_bonus = 0
        if telemetry_struct.get("mpi_wait_pct", 0.0) >= 25.0:
            if "mpi_wait_pct" in perf_metrics or "throughput" in perf_metrics:
                telemetry_bonus += 2

        if telemetry_struct.get("omp_barrier_pct", 0.0) >= 15.0:
            if "omp_barrier_pct" in perf_metrics or "scaling_efficiency" in perf_metrics:
                telemetry_bonus += 2

        if telemetry_struct.get("omp_imbalance_ratio", 1.0) >= 1.5:
            if "omp_imbalance_ratio" in perf_metrics or "scaling_efficiency" in perf_metrics:
                telemetry_bonus += 2

        if telemetry_struct.get("memory_bound_score", 0.0) >= 0.6:
            if any(("cache" in m or "memory" in m) for m in perf_metrics):
                telemetry_bonus += 2

        return (
            risk_rank.get(candidate_dict.get("risk_level", "high"), 2),
            -candidate_dict.get("catalog_score", 0),
            -telemetry_bonus,
            candidate_dict.get("pattern", ""),
        )

    def run(
        self,
        code_snippets: str,
        profiling_summary: str,
        telemetry_summary: str,
        telemetry_struct: Dict[str, float],
    ) -> AdvisorResult:
        routing = self.router.route(telemetry_struct)

        expert_outputs: List[ExpertOutput] = []
        all_candidates: List[Dict[str, Any]] = []

        for expert_name in routing.selected_experts:
            retrieved = self.kb.retrieve_for_expert_and_telemetry(
                expert_name=expert_name,
                telemetry=telemetry_struct,
                limit=8,
            )

            retrieved_patterns = [self._pattern_to_prompt_dict(p) for p in retrieved]

            ctx = ExpertContext(
                code_snippets=code_snippets,
                profiling_summary=profiling_summary,
                telemetry_summary=telemetry_summary,
                retrieved_patterns=retrieved_patterns,
            )

            expert_output = self.expert_map[expert_name].propose(ctx)
            expert_outputs.append(expert_output)

            for candidate in expert_output.candidates:
                d = candidate.to_dict()
                d["proposed_by"] = expert_output.expert_name
                d["catalog_score"] = self._catalog_score_for_candidate(
                    candidate_dict=d,
                    telemetry_struct=telemetry_struct,
                )
                all_candidates.append(d)

        all_candidates.sort(
            key=lambda d: self._candidate_sort_key(
                candidate_dict=d,
                telemetry_struct=telemetry_struct,
            )
        )

        final_ranked = all_candidates[:3]

        return AdvisorResult(
            routing=routing,
            expert_outputs=expert_outputs,
            final_ranked_candidates=final_ranked,
        )