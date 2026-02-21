# temp_moe - Mixture of Experts HPC Performance Advisor

A Python implementation of a Mixture of Experts (MoE) system for HPC performance optimization recommendations.

## Overview

This system routes performance telemetry to specialized experts who provide structured optimization recommendations grounded in an HPC optimization catalog:

- **Communication & Resilience Expert** - handles MPI communication issues, async communication, message aggregation, and checkpointing.
- **Parallelism & Job Expert** - addresses OpenMP scheduling, thread tuning, load balancing, and rank × thread configuration.
- **Kernel & System Efficiency Expert** - focuses on memory-bound kernels, loop geometry optimizations, and system-level efficiency improvements.

Routing decisions are made using a telemetry-driven SimpleTelemetryRouter.

## Project Structure

```
temp_moe/
├── implementation/
│   ├── advisor.py              # Main MoEAdvisor orchestration
│   ├── router.py               # SimpleTelemetryRouter for expert selection
│   ├── experts.py              # Expert implementations
│   ├── kb.py                   # KnowledgeBase for optimization catalog
│   ├── llm.py                  # MockLLM for testing
│   └── prompts/                # Jinja2 templates for expert prompts
├── benchmarks/
│   ├── mem_saxpy/              # Memory-bound kernel benchmark
│   ├── omp_imbalance/          # OpenMP imbalance benchmark
│   └── mpi_pingpong/           # MPI communication benchmark
├── tests/
│   ├── test_router.py                  # Router logic tests
│   ├── test_experts_mock_llm.py        # Expert prompt/LLM/parsing tests
│   ├── test_advisor_end_to_end.py      # Full pipeline integration test
│   ├── test_bench_mem_saxpy.py         # Memory benchmark test
│   ├── test_bench_omp_imbalance.py     # OpenMP benchmark test
│   └── test_bench_mpi_pingpong.py      # MPI benchmark test
├── updated_optimization_catalog.csv    # Knowledge base data
├── pytest.ini                          # Pytest configuration
└── README.md
```

## Quick Start

```bash
# Install Python dependencies
python3 -m pip install pytest jinja2

# Run all tests
python3 -m pytest -v

# Run only benchmark tests
python3 -m pytest -v -m benchmark

# Show skip reasons
python3 -m pytest -v -rs
```

## Testing

### Router Tests (`tests/test_router.py`)
- Validates routing logic with synthetic telemetry
- Confirms expert selection based on thresholds:
  - High `mpi_wait_pct` → Communication expert
  - High `omp_barrier_pct` / imbalance → Parallelism expert
  - High `memory_bound_score` → Kernel expert

### Expert Tests (`tests/test_experts_mock_llm.py`)
- Tests each expert's prompt rendering and LLM interaction
- Validates JSON schema and knowledge base constraints
- Proves: prompt → LLM → JSON parsing → validation → catalog enforcement

### End-to-End Tests (`tests/test_advisor_end_to_end.py`)
- Full MoEAdvisor pipeline execution
- Confirms router selection and final recommendations
- Proves: complete left-side pipeline works end-to-end

### Benchmark Tests

#### Memory Benchmark (`tests/test_bench_mem_saxpy.py`)
- Runs memory-bound SAXPY kernel
- Measures runtime
- Verifies checksum stability

#### OpenMP Benchmark (`tests/test_bench_omp_imbalance.py`)
- Runs OpenMP scheduling experiment
- Verifies correctness and non-regression
- Skips automatically if OpenMP toolchain unavailable

#### MPI Benchmark (`tests/test_bench_mpi_pingpong.py`)
- Runs MPI ping-pong communication test
- Verifies execution and checksum
- Skips automatically if MPI is not configured

## Usage

```python
from pathlib import Path
from implementation.advisor import MoEAdvisor
from implementation.kb import KnowledgeBase
from implementation.llm import MockLLM

kb = KnowledgeBase.from_csv(Path("updated_optimization_catalog.csv"))

advisor = MoEAdvisor(
    llm=MockLLM(),
    kb=kb,
    prompts_dir=Path("implementation/prompts")
)

result = advisor.run(
    code_snippets="MPI_Waitall(...);",
    profiling_summary="MPI_Waitall 40%",
    telemetry_summary="mpi_wait_pct=35",
    telemetry_struct={
        "mpi_wait_pct": 35.0,
        "omp_barrier_pct": 2.0,
        "memory_bound_score": 0.2
    }
)

print(result)
```
