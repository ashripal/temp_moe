# temp_moe - Mixture of Experts HPC Performance Advisor

A Python implementation of a Mixture of Experts (MoE) system for HPC performance optimization recommendations.

## Overview

This system routes performance telemetry to specialized experts who provide optimization recommendations:

- **Communication & Resilience Expert** - handles MPI communication issues
- **Parallelism & Job Expert** - addresses OpenMP and load balancing problems  
- **Kernel & System Efficiency Expert** - focuses on memory-bound and system-level optimizations

## Project Structure

```
temp_moe/
├── implementation/
│   ├── advisor.py          # Main MoEAdvisor orchestration
│   ├── router.py           # SimpleTelemetryRouter for expert selection
│   ├── kb.py              # KnowledgeBase for optimization catalog
│   ├── llm.py             # MockLLM for testing
│   └── prompts/           # Jinja2 templates for expert prompts
├── tests/
│   ├── test_router.py     # Router logic tests
│   ├── test_experts_mock_llm.py  # Expert prompt/LLM/parsing tests
│   └── test_advisor_end_to_end.py  # Full pipeline integration test
└── updated_optimization_catalog.csv  # Knowledge base data
```

## Quick Start

```bash
# Install dependencies
python3 -m pip install -e .

# Run tests
python3 -m pytest tests/ -v

# Run demo
cd implementation
python3 run_demo.py
```

## Testing

### Router Tests (`tests/test_router.py`)
- Validates routing logic with synthetic telemetry
- Confirms expert selection based on thresholds:
  - High `mpi_wait_pct` → Communication expert
  - High `omp_barrier_pct`/imbalance → Parallelism expert  
  - High `memory_bound_score` → Kernel expert

### Expert Tests (`tests/test_experts_mock_llm.py`)
- Tests each expert's prompt rendering and LLM interaction
- Validates JSON schema and knowledge base constraints
- Proves: prompt → LLM → JSON parsing → validation pipeline

### End-to-End Tests (`tests/test_advisor_end_to_end.py`)
- Full MoEAdvisor pipeline execution
- Confirms router selection and final recommendations
- Proves: complete left-side pipeline works end-to-end

## Usage

```python
from pathlib import Path
from implementation.advisor import MoEAdvisor
from implementation.kb import KnowledgeBase
from implementation.llm import MockLLM

kb = KnowledgeBase.from_csv(Path("updated_optimization_catalog.csv"))
advisor = MoEAdvisor(llm=MockLLM(), kb=kb, prompts_dir=Path("implementation/prompts"))

result = advisor.run(
    code_snippets="MPI_Waitall(...);",
    profiling_summary="MPI_Waitall 40%", 
    telemetry_summary="mpi_wait_pct=35",
    telemetry_struct={"mpi_wait_pct": 35.0, "memory_bound_score": 0.2}