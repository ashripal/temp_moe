from pathlib import Path

from implementation.kb import KnowledgeBase
from implementation.llm import MockLLM
from implementation.advisor import MoEAdvisor

def test_advisor_end_to_end():
    kb = KnowledgeBase.from_csv(Path("updated_optimization_catalog.csv"))
    advisor = MoEAdvisor(llm=MockLLM(), kb=kb, prompts_dir=Path("implementation/prompts"))

    result = advisor.run(
        code_snippets="MPI_Waitall(...);",
        profiling_summary="MPI_Waitall 40%",
        telemetry_summary="mpi_wait_pct=45",
        telemetry_struct={"mpi_wait_pct": 45.0, "omp_barrier_pct": 2.0, "omp_imbalance_ratio": 1.1, "memory_bound_score": 0.2},
    )

    assert result.final_ranked_candidates, "Should produce final candidates"
    assert result.routing.selected_experts[0] == "Communication & Resilience Expert"