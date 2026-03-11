from pathlib import Path
import os
import time

import pytest

from implementation.kb import KnowledgeBase
from implementation.llm import TransformersLLM
from implementation.advisor import MoEAdvisor


pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.getenv("RUN_HF_INTEGRATION", "0") != "1",
    reason="Set RUN_HF_INTEGRATION=1 to run real Hugging Face integration tests.",
)
def test_advisor_end_to_end_with_real_transformers_llm():
    kb = KnowledgeBase.from_csv(Path("updated_optimization_catalog.csv"))

    model_name = os.getenv("HF_MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct")

    llm = TransformersLLM(
        model_name=model_name,
        max_new_tokens=160,
        temperature=0.0,
    )

    advisor = MoEAdvisor(
        llm=llm,
        kb=kb,
        prompts_dir=Path("implementation/prompts"),
    )

    start = time.perf_counter()

    result = advisor.run(
        code_snippets="MPI_Waitall(...);",
        profiling_summary="MPI_Waitall 40%",
        telemetry_summary="mpi_wait_pct=45",
        telemetry_struct={
            "mpi_wait_pct": 45.0,
            "omp_barrier_pct": 2.0,
            "omp_imbalance_ratio": 1.1,
            "memory_bound_score": 0.2,
        },
    )

    elapsed = time.perf_counter() - start

    assert result.final_ranked_candidates, "Should produce final candidates"
    assert result.routing.selected_experts[0] == "Communication & Resilience Expert"

    allowed = kb.allowed_patterns()
    for candidate in result.final_ranked_candidates:
        assert candidate["pattern"] in allowed, f"Pattern not in catalog: {candidate['pattern']}"

    # Loose latency bound; tune based on your machine/model size
    assert elapsed < 120, f"Inference took too long: {elapsed:.2f}s"