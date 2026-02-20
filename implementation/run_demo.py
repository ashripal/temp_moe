from __future__ import annotations
from pathlib import Path
import json

from .kb import KnowledgeBase
from .llm import MockLLM
from .advisor import MoEAdvisor


def main() -> None:
    root = Path(__file__).resolve().parent
    prompts_dir = root / "prompts"

    # Point this to your uploaded CSV
    # e.g., "../../updated_optimization_catalog.csv" if running from repo root
    catalog_path = Path("updated_optimization_catalog.csv")

    kb = KnowledgeBase.from_csv(catalog_path)
    llm = MockLLM()

    advisor = MoEAdvisor(llm=llm, kb=kb, prompts_dir=prompts_dir)

    code_snippets = "/* hotspot loop */ for (i=0; i<n; i++) { ... }"
    profiling_summary = "Hotspots: loop_12 35%, MPI_Waitall 22%. OpenMP barrier 18%."
    telemetry_summary = "Nodes:1->8 runtime worsens; mpi_wait_pct=35; omp_barrier_pct=10; memory_bound_score=0.2"
    telemetry_struct = {"mpi_wait_pct": 35.0, "omp_barrier_pct": 10.0, "omp_imbalance_ratio": 1.2, "memory_bound_score": 0.2}

    result = advisor.run(code_snippets, profiling_summary, telemetry_summary, telemetry_struct)

    print("Routing decision:", result.routing)
    print("\nFinal ranked candidates:")
    print(json.dumps(result.final_ranked_candidates, indent=2))


if __name__ == "__main__":
    main()