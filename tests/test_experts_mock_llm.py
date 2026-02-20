from pathlib import Path
import pytest

from implementation.kb import KnowledgeBase
from implementation.llm import MockLLM
from implementation.experts import ParallelismJobExpert, CommunicationResilienceExpert, KernelSystemEfficiencyExpert, ExpertContext

CATALOG = Path("updated_optimization_catalog.csv")

@pytest.fixture
def kb():
    return KnowledgeBase.from_csv(CATALOG)

@pytest.fixture
def prompts_dir():
    return Path("implementation/prompts")

@pytest.fixture
def ctx():
    return ExpertContext(
        code_snippets="for(i=0;i<n;i++){...}",
        profiling_summary="OpenMP barrier 18%, MPI_Waitall 22%",
        telemetry_summary="mpi_wait_pct=35",
        retrieved_patterns=[]
    )

def test_parallelism_expert_schema_and_catalog(kb, prompts_dir, ctx):
    e = ParallelismJobExpert(MockLLM(), kb, prompts_dir)
    out = e.propose(ctx)
    assert out.candidates, "Expected at least one candidate"
    for c in out.candidates:
        assert c.pattern in kb.allowed_patterns()

def test_comm_expert_schema_and_catalog(kb, prompts_dir, ctx):
    e = CommunicationResilienceExpert(MockLLM(), kb, prompts_dir)
    out = e.propose(ctx)
    assert out.candidates
    for c in out.candidates:
        assert c.pattern in kb.allowed_patterns()

def test_kernel_expert_schema_and_catalog(kb, prompts_dir, ctx):
    e = KernelSystemEfficiencyExpert(MockLLM(), kb, prompts_dir)
    out = e.propose(ctx)
    assert out.candidates
    for c in out.candidates:
        assert c.pattern in kb.allowed_patterns()