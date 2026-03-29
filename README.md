# OptimizeHPC - Mixture of Experts HPC Performance Advisor

A Python implementation of a Mixture of Experts (MoE) system for HPC performance optimization recommendations. The system leverages specialized expert agents to provide targeted optimization suggestions based on performance telemetry.

## Overview

The MoE Advisor routes performance telemetry to specialized experts who provide structured, knowledge-base-grounded optimization recommendations:

- **Communication & Resilience Expert** - Handles MPI communication issues, async communication, message aggregation, and checkpointing strategies
- **Parallelism & Job Expert** - Addresses OpenMP scheduling, thread tuning, load balancing, and rank × thread configuration optimization
- **Kernel & System Efficiency Expert** - Focuses on memory-bound kernels, loop geometry optimizations, and system-level efficiency improvements

The system uses a telemetry-driven **SimpleTelemetryRouter** to intelligently select experts based on performance bottleneck patterns.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Input: Performance Data                  │
│  (code snippet, telemetry metrics, profiling summary)      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
            ┌────────────────────────────┐
            │ SimpleTelemetryRouter      │
            │  (mpi_wait_pct threshold)  │
            │  (omp_barrier_pct)         │
            │  (memory_bound_score)      │
            └────────────┬─────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
   ┌─────────┐    ┌────────────┐    ┌──────────┐
   │  Comm   │    │ Parallelism│    │  Kernel  │
   │ Expert  │    │   Expert   │    │  Expert  │
   └────┬────┘    └──────┬─────┘    └────┬─────┘
        │                │               │
        └────────────────┼───────────────┘
                         │
         ┌───────────────▼───────────────┐
         │  Prompt Rendering (Jinja2)   │
         │  + KB Pattern Injection       │
         └───────────────┬───────────────┘
                         │
         ┌───────────────▼───────────────┐
         │  LLM (TransformersLLM/MockLLM)│
         │  (Model: Llama-3.1-8B)       │
         └───────────────┬───────────────┘
                         │
         ┌───────────────▼───────────────┐
         │  JSON Parsing & Validation   │
         │  + Schema Enforcement         │
         └───────────────┬───────────────┘
                         │
         ┌───────────────▼───────────────┐
         │  Ranked Optimization List     │
         │  (from KB Catalog)            │
         └───────────────────────────────┘
```

## Directory Structure

```
temp_moe/
├── implementation/                      # Core MoE advisor implementation
│   ├── advisor.py                      # Main MoEAdvisor orchestration
│   ├── router.py                       # SimpleTelemetryRouter for expert selection
│   ├── experts.py                      # Expert implementations (3 experts)
│   ├── kb.py                           # KnowledgeBase catalog loader
│   ├── llm.py                          # MockLLM & TransformersLLM backends
│   ├── schema.py                       # Data class schemas
│   ├── run_demo.py                     # Demo entry point
│   ├── analysis/                       # Code analysis module
│   │   ├── code_analyzer.py           # Static code analysis
│   │   ├── profiler_parser.py         # Performance profiler output parsing
│   │   ├── telemetry_extractor.py     # Telemetry metric extraction
│   │   └── analysis_bundle.py         # Bundled analysis results
│   ├── generator/                      # Code generation module (right-side pipeline)
│   │   ├── generator.py               # Core generator
│   │   ├── generator_llm.py           # Generator-specific LLM integration
│   │   ├── generator_schema.py        # Generator schemas
│   │   └── generator_utils.py         # Generator utilities
│   └── prompts/                        # Jinja2 prompt templates
│       ├── communication_expert.jinja2
│       ├── parallelism_expert.jinja2
│       └── kernel_expert.jinja2
│
├── benchmarks/                          # Benchmark applications (C/C++)
│   ├── mem_saxpy/                      # Memory-bound SAXPY kernel
│   │   ├── main.c
│   │   ├── Makefile
│   │   └── mem_saxpy
│   ├── omp_imbalance/                  # OpenMP load imbalance test
│   │   ├── main.c
│   │   ├── Makefile
│   │   └── omp_imbalance
│   └── mpi_pingpong/                   # MPI communication benchmark
│       ├── main.c
│       ├── Makefile
│       └── mpi_pingpong
│
├── tests/                               # Comprehensive test suite
│   ├── test_router.py                  # Router logic unit tests
│   ├── test_experts_mock_llm.py        # Expert prompt/parsing tests
│   ├── test_advisor_end_to_end.py      # Full pipeline integration tests
│   ├── test_bench_mem_saxpy.py         # Memory benchmark tests
│   ├── test_bench_omp_imbalance.py     # OpenMP benchmark tests
│   ├── test_bench_mpi_pingpong.py      # MPI benchmark tests
│   ├── test_code_analyzer.py           # Code analyzer tests
│   ├── test_profiler_parser.py         # Profiler parser tests
│   ├── test_telemetry_extractor.py     # Telemetry extraction tests
│   ├── test_generator_*.py             # Generator pipeline tests
│   ├── test_llm_backends.py            # LLM backend integration tests
│   ├── test_advisor_hf_integration.py  # HuggingFace integration tests
│   ├── bench_utils.py                  # Benchmark utilities
│   ├── run_tests.sh                    # Test runner script
│   └── compare_c_versions.py           # Benchmark comparison utilities
│
├── empirical_study/                     # Empirical research analysis
│   ├── scripts/
│   │   ├── github_repo_search_cli.py   # GitHub code search
│   │   ├── github_repo_prs_cli.py      # PR extraction
│   │   ├── github_pr_optimization_filter_cli.py  # Filter optimization PRs
│   │   └── pr_lines_histogram.py       # Visualization
│   ├── inputs/                          # Input data
│   ├── outputs/                         # Analysis results
│   └── README.md                        # Empirical study docs
│
├── cluster_tests/                       # High-performance cluster tests
│   ├── orchestrator.py                 # Test orchestration
│   └── runner.sbatch                   # SLURM job submission
│
├── generated_optimizations/             # Generated optimization results
│   ├── mem_saxpy/
│   ├── omp_imbalance/
│   └── mpi_pingpong/
│
├── updated_optimization_catalog.csv     # Knowledge base data
├── pytest.ini                           # Pytest configuration
├── requirements.txt                     # Python dependencies
├── setup.py                             # Package setup
└── README.md                            # This file
```

## Installation

### Prerequisites

- Python 3.9+ (tested with 3.14)
- Conda or venv for environment management
- C/C++ compiler (for benchmarks)
- OpenMP (optional, for OpenMP benchmarks)
- MPI (optional, for MPI benchmarks)

### Setup

1. **Create and activate a conda environment:**

```bash
conda create -n temp_moe python=3.9
conda activate temp_moe
```

2. **Install dependencies:**

```bash
# Install core dependencies
conda install pytest jinja2

# Install ML dependencies (for HuggingFace models)
conda install pytorch::pytorch torchvision torchaudio -c pytorch
conda install transformers

# Or install from requirements.txt
pip install -r requirements.txt
```

3. **Install package in development mode:**

```bash
pip install -e .
```

## Quick Start

```bash
# Activate environment
conda activate temp_moe

# Run all tests
python -m pytest -v

# Run specific test category
python -m pytest tests/test_router.py -v

# Run with verbose output and stop on first failure
python -m pytest -v -x

# Generate HTML report
python -m pytest --html=report.html
```

## Testing

### Test Organization

Tests are organized by functional area:

| Category | Tests | Purpose |
|----------|-------|---------|
| **Advisor Core** | `test_router.py`, `test_experts_mock_llm.py`, `test_advisor_end_to_end.py` | Left-side pipeline (advisor stage) |
| **Code Analysis** | `test_code_analyzer.py`, `test_profiler_parser.py`, `test_telemetry_extractor.py` | Input analysis and preprocessing |
| **Code Generation** | `test_generator_*.py` | Right-side pipeline (generation stage) |
| **LLM Integration** | `test_llm_backends.py`, `test_advisor_hf_integration.py` | Backend LLM integration |
| **Benchmarks** | `test_bench_*.py` | Application benchmarks |

### Running Tests

#### All Tests

```bash
# Run all tests with verbose output
python -m pytest -v

# Run all tests and generate HTML report
python -m pytest -v --html=report.html --self-contained-html

# Run with minimal output
python -m pytest
```

#### Advisor Pipeline Tests (Left-Side)

```bash
# Router logic tests
python -m pytest tests/test_router.py -v

# Expert prompt/LLM/parsing tests
python -m pytest tests/test_experts_mock_llm.py -v

# End-to-end advisor pipeline test
python -m pytest tests/test_advisor_end_to_end.py -v

# All advisor tests
python -m pytest tests/test_router.py tests/test_experts_mock_llm.py tests/test_advisor_end_to_end.py -v
```

#### Code Analysis Tests

```bash
# Code analyzer tests
python -m pytest tests/test_code_analyzer.py -v

# Profiler parser tests
python -m pytest tests/test_profiler_parser.py -v

# Telemetry extraction tests
python -m pytest tests/test_telemetry_extractor.py -v

# All analysis tests
python -m pytest tests/test_code_analyzer.py tests/test_profiler_parser.py tests/test_telemetry_extractor.py -v
```

#### Generator Pipeline Tests (Right-Side)

```bash
# Generator schema tests
python -m pytest tests/test_generator_schema.py -v

# Generator prompt rendering tests
python -m pytest tests/test_generator_prompt_rendering.py -v

# Generator LLM integration tests
python -m pytest tests/test_generator_llm.py -v

# All generator tests
python -m pytest tests/test_generator_*.py -v
```

#### Benchmark Tests

```bash
# Memory (SAXPY) benchmark
python -m pytest tests/test_bench_mem_saxpy.py -v -s

# OpenMP imbalance benchmark
python -m pytest tests/test_bench_omp_imbalance.py -v -s

# MPI ping-pong benchmark
python -m pytest tests/test_bench_mpi_pingpong.py -v -s

# All benchmarks (may be skipped if toolchain unavailable)
python -m pytest tests/test_bench_*.py -v -s

# Show skip reasons
python -m pytest tests/test_bench_*.py -v -rs
```

#### LLM Integration Tests

```bash
# LLM backend tests
python -m pytest tests/test_llm_backends.py -v

# HuggingFace integration (requires GPU/model download)
python -m pytest tests/test_advisor_hf_integration.py -v
```

#### Using run_tests.sh Script

```bash
# Run all tests
bash tests/run_tests.sh all

# Run advisor tests only
bash tests/run_tests.sh advisor

# Run analysis tests
bash tests/run_tests.sh analysis

# Run generator tests
bash tests/run_tests.sh generator

# Run benchmark tests
bash tests/run_tests.sh benchmark

# Run specific test file
bash tests/run_tests.sh file tests/test_router.py

# Run specific test node
bash tests/run_tests.sh node tests/test_router.py::test_router_picks_comm_when_mpi_wait_high

# List available test groups
bash tests/run_tests.sh list

# Run with verbose output
VERBOSE=1 bash tests/run_tests.sh analysis

# Stop on first failure
FAILFAST=1 bash tests/run_tests.sh all
```

### Test Details

#### Router Tests (`tests/test_router.py`)
- Validates routing logic with synthetic telemetry
- Confirms expert selection based on thresholds:
  - High `mpi_wait_pct` → Communication expert
  - High `omp_barrier_pct` / imbalance → Parallelism expert
  - High `memory_bound_score` → Kernel expert

#### Expert Tests (`tests/test_experts_mock_llm.py`)
- Tests each expert's prompt rendering and LLM interaction
- Validates JSON schema and knowledge base constraints
- Proves: prompt → LLM → JSON parsing → validation → catalog enforcement

#### End-to-End Tests (`tests/test_advisor_end_to_end.py`)
- Full MoEAdvisor pipeline execution
- Confirms router selection and final recommendations
- Proves: complete left-side pipeline works end-to-end

#### Code Analysis Tests
- **`test_code_analyzer.py`**: Static code feature extraction (functions, loops, MPI calls, OpenMP pragmas)
- **`test_profiler_parser.py`**: Performance profiler output parsing
- **`test_telemetry_extractor.py`**: Performance metrics extraction

#### Generator Tests (`tests/test_generator_*.py`)
- Prompt rendering and template processing
- Schema validation
- LLM-based code generation
- Feedback integration

#### Benchmark Tests

##### Memory Benchmark (`tests/test_bench_mem_saxpy.py`)
- Runs memory-bound SAXPY kernel
- Measures runtime and computes speedup
- Verifies checksum stability
- Verifies performance regression detection

##### OpenMP Benchmark (`tests/test_bench_omp_imbalance.py`)
- Runs OpenMP scheduling experiment
- Compares baseline vs. tuned scheduling
- Verifies correctness and non-regression
- Skips automatically if OpenMP toolchain unavailable

##### MPI Benchmark (`tests/test_bench_mpi_pingpong.py`)
- Runs MPI ping-pong communication test
- Verifies execution and checksum
- Measures communication latency
- Skips automatically if MPI is not configured

## Usage

### Basic Advisor Usage

```python
from pathlib import Path
from implementation.advisor import MoEAdvisor
from implementation.kb import KnowledgeBase
from implementation.llm import MockLLM

# Load knowledge base
kb = KnowledgeBase.from_csv(Path("updated_optimization_catalog.csv"))

# Create advisor with mock LLM
advisor = MoEAdvisor(
    llm=MockLLM(),
    kb=kb,
    prompts_dir=Path("implementation/prompts")
)

# Run advisor with performance telemetry
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

# Print results
print(f"Selected experts: {result.routing.selected_experts}")
print(f"Top recommendations: {result.final_ranked_candidates}")
```

### Using HuggingFace Models

```python
from pathlib import Path
from implementation.advisor import MoEAdvisor
from implementation.kb import KnowledgeBase
from implementation.llm import TransformersLLM

kb = KnowledgeBase.from_csv(Path("updated_optimization_catalog.csv"))

# Create advisor with real LLM (Llama-3.1-8B)
llm = TransformersLLM(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    max_new_tokens=800,
    temperature=0.0
)

advisor = MoEAdvisor(
    llm=llm,
    kb=kb,
    prompts_dir=Path("implementation/prompts")
)

# Run advisor
result = advisor.run(...)
```

### Using Environment Variables

```bash
# Set LLM configuration
export USE_MOCK_LLM=0
export HF_MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export MAX_NEW_TOKENS=800
export TEMPERATURE=0.0

# Run demo
cd implementation
python run_demo.py
```

### Code Analysis

```python
from implementation.analysis.code_analyzer import CodeAnalyzer
from pathlib import Path

analyzer = CodeAnalyzer()
summary = analyzer.analyze_file(Path("benchmark.c"))

print(f"Language: {summary.language}")
print(f"Functions: {summary.function_count}")
print(f"MPI calls: {len(summary.mpi_calls)}")
print(f"OpenMP regions: {len(summary.omp_regions)}")
```

## Running the Demo

```bash
# Activate environment
conda activate temp_moe

# Run demo with MockLLM (no model download needed)
cd implementation
USE_MOCK_LLM=1 python run_demo.py

# Run demo with real HuggingFace model (requires GPU, first run downloads model)
cd implementation
python run_demo.py
```

## Project Components

### Core Modules

#### `implementation/advisor.py`
Main orchestration class that coordinates routing and expert selection.

```python
class MoEAdvisor:
    def __init__(self, llm, kb, prompts_dir)
    def run(self, code_snippets, telemetry_struct, ...) -> AdvisorResult
```

#### `implementation/router.py`
Routes telemetry metrics to appropriate experts based on thresholds.

```python
class SimpleTelemetryRouter:
    def route(self, telemetry: dict) -> RoutingDecision
```

#### `implementation/experts.py`
Expert implementations with prompt rendering and response parsing.

```python
class CommunicationExpert(Expert)
class ParallelismExpert(Expert)
class KernelExpert(Expert)
```

#### `implementation/kb.py`
Knowledge base catalog management.

```python
class KnowledgeBase:
    @classmethod
    def from_csv(cls, path: Path) -> KnowledgeBase
```

#### `implementation/llm.py`
LLM backends: MockLLM for testing and TransformersLLM for HuggingFace models.

```python
class MockLLM(LLMClient)
class TransformersLLM(LLMClient)
```

### Analysis Modules

- **`code_analyzer.py`**: Static code analysis (functions, loops, API calls)
- **`profiler_parser.py`**: Parse performance profiler output
- **`telemetry_extractor.py`**: Extract metrics from telemetry data

### Generator Modules

- **`generator.py`**: Code generation pipeline
- **`generator_llm.py`**: LLM-based code suggestions
- **`generator_schema.py`**: Generation schemas and validation
- **`generator_utils.py`**: Helper utilities

### Benchmark Applications

All benchmarks are in C/C++ and can be compiled and run independently:

```bash
# Build all benchmarks
cd benchmarks
make -C mem_saxpy
make -C omp_imbalance
make -C mpi_pingpong

# Run individual benchmark
./benchmarks/mem_saxpy/mem_saxpy
./benchmarks/omp_imbalance/omp_imbalance
./benchmarks/mpi_pingpong/mpi_pingpong
```

## Advanced Topics

### Custom Telemetry Format

The system accepts arbitrary telemetry dictionaries:

```python
telemetry_struct = {
    "mpi_wait_pct": 45.0,          # MPI blocking percentage
    "omp_barrier_pct": 5.0,        # OpenMP barrier time
    "omp_imbalance_ratio": 1.2,    # Load imbalance ratio
    "memory_bound_score": 0.3,     # Memory-boundedness (0-1)
    "cache_miss_rate": 0.15,       # L3 cache miss rate
    "flops_peak": 100.0            # Peak FLOP rate
}
```

### Extending with Custom Experts

Create a new expert by extending the `Expert` base class:

```python
from implementation.experts import Expert

class CustomExpert(Expert):
    def __init__(self, llm, kb, prompts_dir):
        super().__init__(llm, kb, prompts_dir, expert_name="Custom Expert")
    
    def get_recommendations(self, context: dict) -> list:
        # Custom logic here
        pass
```

### Custom LLM Backends

Implement the `LLMClient` protocol:

```python
from implementation.llm import LLMClient, LLMMessage
from typing import List

class CustomLLM(LLMClient):
    def complete(self, messages: List[LLMMessage]) -> str:
        # Your LLM API call here
        pass
```

## Troubleshooting

### Pytest Not Found

```bash
# Ensure you're in the conda environment
conda activate temp_moe

# Reinstall pytest
conda install pytest

# Or use python -m pytest directly
python -m pytest tests/ -v
```

### HuggingFace Model Download Issues

```bash
# Set cache directory
export HF_HOME=/path/to/cache

# Download model explicitly
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')"
```

### Benchmark Compilation Failures

```bash
# Check if OpenMP is installed
gcc -fopenmp -v

# Check if MPI is installed
mpicc -version

# Install missing tools
conda install -c conda-forge openmpi

# Or use Homebrew on macOS
brew install open-mpi
```

### Out of Memory on GPU

```bash
# Use CPU instead
export CUDA_VISIBLE_DEVICES=""

# Reduce batch size in your code
# Or use a smaller model: "meta-llama/Llama-2-7B-Instruct"
```

## Performance Tips

- Use `MockLLM` for development and testing (instant responses)
- Use smaller models like Llama-2-7B for faster inference
- Run benchmarks on idle systems to avoid noise
- Use `pytest -v -s` to see benchmark output in real-time
- Profile with `pytest --profile` to identify slow tests

## Contributing

1. Create a new branch for your feature
2. Add tests in `tests/`
3. Run full test suite: `python -m pytest -v`
4. Ensure all tests pass before submitting PR

## References

- Architecture diagram: `moe_diagram.jpg`
- Optimization catalog: `updated_optimization_catalog.csv`
- Empirical study: `empirical_study/README.md`
- Cluster testing: `cluster_tests/README.md`

## License

[Add your license here]

## Contact

For questions or issues, please open a GitHub issue.
