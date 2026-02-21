temp_moe – Mixture of Experts HPC Performance Advisor

A Python implementation of a Mixture of Experts (MoE) system for HPC performance optimization recommendations.

Overview

This system routes performance telemetry to specialized experts who provide structured optimization recommendations grounded in an HPC optimization catalog.

Communication & Resilience Expert – handles MPI wait time analysis, async communication, message aggregation, and checkpointing.

Parallelism & Job Expert – addresses OpenMP scheduling, thread tuning, load balancing, and rank × thread configuration.

Kernel & System Efficiency Expert – focuses on loop geometry, memory-bound kernels, precision tuning, and system-level optimizations.

Routing decisions are made using a telemetry-driven SimpleTelemetryRouter.

Project Structure
temp_moe/
├── implementation/
│   ├── advisor.py
│   ├── router.py
│   ├── experts.py
│   ├── kb.py
│   ├── llm.py
│   └── prompts/
│
├── benchmarks/
│   ├── mem_saxpy/
│   ├── omp_imbalance/
│   └── mpi_pingpong/
│
├── tests/
│   ├── test_router.py
│   ├── test_experts_mock_llm.py
│   ├── test_advisor_end_to_end.py
│   ├── test_bench_mem_saxpy.py
│   ├── test_bench_omp_imbalance.py
│   └── test_bench_mpi_pingpong.py
│
├── updated_optimization_catalog.csv
├── pytest.ini
└── README.md
Requirements
Python Dependencies

Python 3.9+

pytest

jinja2

Install:

python3 -m pip install pytest jinja2
System Dependencies (for benchmarks)
Benchmark	Required
mem_saxpy	C compiler (cc / clang / gcc)
omp_imbalance	C compiler + OpenMP runtime
mpi_pingpong	mpicc + mpirun (OpenMPI or MPICH)

macOS OpenMP:

brew install libomp

macOS MPI:

brew install open-mpi
Running Tests

Run all tests:

python3 -m pytest -v

Run only benchmark tests:

python3 -m pytest -v -m benchmark

Show skip reasons:

python3 -m pytest -v -rs
Test Suite Description
Router Tests (test_router.py)

Validates telemetry-based routing logic:

High mpi_wait_pct → Communication expert

High OpenMP imbalance → Parallelism expert

High memory-bound score → Kernel expert

Expert Tests (test_experts_mock_llm.py)

Validates:

Prompt rendering

JSON parsing

Schema validation

Knowledge base constraints

Pipeline validated:

prompt → LLM → JSON → validation → catalog enforcement
End-to-End Test (test_advisor_end_to_end.py)

Executes the full MoE pipeline:

Router selection

Expert invocation

Aggregation

Confirms the left-side advisor architecture functions correctly.

Benchmark Tests
test_bench_mem_saxpy.py

Memory-bound kernel benchmark

Measures runtime

Verifies checksum stability

test_bench_omp_imbalance.py

OpenMP scheduling benchmark

Checks correctness and non-regression

Skips if OpenMP toolchain unavailable

test_bench_mpi_pingpong.py

MPI communication benchmark

Verifies execution and checksum

Skips if MPI not configured

Example Usage
from pathlib import Path
from implementation.advisor import MoEAdvisor
from implementation.kb import KnowledgeBase
from implementation.llm import MockLLM

kb = KnowledgeBase.from_csv(Path("updated_optimization_catalog.csv"))

advisor = MoEAdvisor(
    llm=MockLLM(),
    kb=kb,
    prompts_dir=Path("implementation/prompts"),
)

result = advisor.run(
    code_snippets="MPI_Waitall(...);",
    profiling_summary="MPI_Waitall 40%",
    telemetry_summary="mpi_wait_pct=35",
    telemetry_struct={
        "mpi_wait_pct": 35.0,
        "omp_barrier_pct": 2.0,
        "memory_bound_score": 0.2
    },
)

print(result)
