from __future__ import annotations

import json
import os
from pathlib import Path

from .advisor import MoEAdvisor
from .kb import KnowledgeBase
from .llm import MockLLM, TransformersLLM


def main() -> None:
    root = Path(__file__).resolve().parent
    prompts_dir = root / "prompts"

    # Point this to your optimization catalog CSV
    catalog_path = Path("updated_optimization_catalog.csv")

    kb = KnowledgeBase.from_csv(catalog_path)

    use_mock = os.getenv("USE_MOCK_LLM", "0").strip().lower() in {"1", "true", "yes"}

    # Default Llama instruct model; override with env var if desired.
    model_name = os.getenv("HF_MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
    max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", "800"))
    temperature = float(os.getenv("TEMPERATURE", "0.0"))

    if use_mock:
        llm = MockLLM()
        print("Using MockLLM")
    else:
        llm = TransformersLLM(
            model_name=model_name,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        print(f"Using TransformersLLM with model={model_name}")

    advisor = MoEAdvisor(llm=llm, kb=kb, prompts_dir=prompts_dir)

    code_snippets = "/* hotspot loop */ for (i=0; i<n; i++) { ... }"
    profiling_summary = "Hotspots: loop_12 35%, MPI_Waitall 22%. OpenMP barrier 18%."
    telemetry_summary = (
        "Nodes:1->8 runtime worsens; mpi_wait_pct=35; "
        "omp_barrier_pct=10; memory_bound_score=0.2"
    )
    telemetry_struct = {
        "mpi_wait_pct": 35.0,
        "omp_barrier_pct": 10.0,
        "omp_imbalance_ratio": 1.2,
        "memory_bound_score": 0.2,
    }

    result = advisor.run(
        code_snippets=code_snippets,
        profiling_summary=profiling_summary,
        telemetry_summary=telemetry_summary,
        telemetry_struct=telemetry_struct,
    )

    print("Routing decision:", result.routing)
    print("\nExpert outputs:")
    for expert_output in result.expert_outputs:
        print(f"\n[{expert_output.expert_name}]")
        print(json.dumps(expert_output.to_dict(), indent=2))

    print("\nFinal ranked candidates:")
    print(json.dumps(result.final_ranked_candidates, indent=2))


if __name__ == "__main__":
    main()