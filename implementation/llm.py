from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Protocol


@dataclass
class LLMMessage:
    role: str  # "system" | "user"
    content: str


class LLMClient(Protocol):
    def complete(self, messages: List[LLMMessage]) -> str:
        ...


class MockLLM:
    """
    Deterministic offline LLM for testing.
    Returns a JSON list of candidate dicts.
    The experts decide which mock response to request by passing a marker in the prompt.
    """
    def complete(self, messages: List[LLMMessage]) -> str:
        prompt = "\n".join(m.content for m in messages)
        if "Parallelism & Job Expert" in prompt:
            return json.dumps([
                {
                    "pattern": "OpenMP scheduling + thread tuning",
                    "target": "foo.c:bar():loop_12",
                    "rationale": "High barrier time and imbalance suggest schedule/chunk tuning.",
                    "action_sketch": "Sweep schedule={static,dynamic,guided} and chunk sizes; tune OMP_NUM_THREADS and affinity.",
                    "preconditions": ["No oversubscription", "Same loop bounds; no semantic changes"],
                    "parameters_to_sweep": {"schedule": ["static", "dynamic", "guided"], "chunk": [1, 4, 16, 64], "threads": [1, 2, 4, 8]},
                    "correctness_checks": ["Run regression tests", "Compare key outputs within tolerance"],
                    "performance_metrics": ["runtime", "omp_barrier_time_pct", "imbalance_ratio", "scaling_efficiency"],
                    "risk_level": "low",
                    "rollback_criteria": ["Correctness failure", "Runtime regression > 3%", "Scaling efficiency drop > 5%"]
                }
            ])
        if "Communication & Resilience Expert" in prompt:
            return json.dumps([
                {
                    "pattern": "Async Communication",
                    "target": "mpi_region:exchange_halos",
                    "rationale": "MPI_Wait% grows with nodes; overlap could reduce wait and improve scaling.",
                    "action_sketch": "Replace blocking send/recv with Irecv/Isend; do independent compute; Waitall.",
                    "preconditions": ["Overlapped compute must not read receive buffers", "All requests completed exactly once"],
                    "parameters_to_sweep": {"overlap_window": ["small", "medium"]},
                    "correctness_checks": ["Run regression tests", "Compare outputs within tolerance", "Deadlock-free run on 2/4/8 nodes"],
                    "performance_metrics": ["runtime", "mpi_wait_pct", "scaling_efficiency"],
                    "risk_level": "medium",
                    "rollback_criteria": ["Correctness failure", "Deadlock/hang", "No mpi_wait_pct improvement"]
                }
            ])
        if "Kernel & System Efficiency Expert" in prompt:
            return json.dumps([
                {
                    "pattern": "Loop Unrolling",
                    "target": "kernel.c:matmul():loop_3",
                    "rationale": "Compute-heavy inner loop; unrolling may improve ILP and reduce loop overhead.",
                    "action_sketch": "Unroll inner loop by factor 2 or 4; keep bounds identical; consider simd hint.",
                    "preconditions": ["No loop-carried dependencies", "Indexing remains identical"],
                    "parameters_to_sweep": {"unroll_factor": [2, 4]},
                    "correctness_checks": ["Run regression tests", "Numeric tolerance check on outputs"],
                    "performance_metrics": ["runtime", "kernel_time_share"],
                    "risk_level": "low",
                    "rollback_criteria": ["Correctness failure", "Runtime regression > 3%"]
                }
            ])
        # Aggregator default
        return "[]"


# Later: implement OpenAIClient, AnthropicClient, etc.
class OpenAIClient:
    def __init__(self, *args: Any, **kwargs: Any):
        raise NotImplementedError("Hook in your preferred provider here.")