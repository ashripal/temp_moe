from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class RoutingDecision:
    selected_experts: List[str]  # names
    reason: str


class SimpleTelemetryRouter:
    """
    Heuristic router for v1: explainable and reliable.
    Uses telemetry_summary string (can be structured later).
    """
    def route(self, telemetry: Dict[str, float]) -> RoutingDecision:
        mpi_wait = telemetry.get("mpi_wait_pct", 0.0)
        omp_barrier = telemetry.get("omp_barrier_pct", 0.0)
        imbalance = telemetry.get("omp_imbalance_ratio", 1.0)
        memory_bound = telemetry.get("memory_bound_score", 0.0)

        # Choose top-1; easy to expand to top-2.
        if mpi_wait >= 25.0:
            return RoutingDecision(
                selected_experts=["Communication & Resilience Expert"],
                reason=f"MPI wait high (mpi_wait_pct={mpi_wait})."
            )
        if omp_barrier >= 15.0 or imbalance >= 1.5:
            return RoutingDecision(
                selected_experts=["Parallelism & Job Expert"],
                reason=f"OpenMP sync/imbalance high (barrier={omp_barrier}, imbalance={imbalance})."
            )
        if memory_bound >= 0.6:
            return RoutingDecision(
                selected_experts=["Kernel & System Efficiency Expert"],
                reason=f"Memory-bound signature high (memory_bound_score={memory_bound})."
            )

        return RoutingDecision(
            selected_experts=["Parallelism & Job Expert"],
            reason="Default to parallelism/job tuning as safe first action."
        )