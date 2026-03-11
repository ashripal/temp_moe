from pathlib import Path
import json

import torch

from implementation.kb import KnowledgeBase
from implementation.llm import TransformersLLM
from implementation.advisor import MoEAdvisor


class FakeBatch(dict):
    def to(self, device):
        return self


class FakeTokenizer:
    def __init__(self, decoded_text: str):
        self.decoded_text = decoded_text
        self.pad_token = None
        self.eos_token = "<eos>"
        self.pad_token_id = 0
        self.eos_token_id = 1

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return "PROMPT"

    def __call__(self, prompt, return_tensors="pt"):
        return FakeBatch({"input_ids": torch.tensor([[1, 2, 3]])})

    def decode(self, ids, skip_special_tokens=True):
        return self.decoded_text


class FakeModel:
    def __init__(self):
        self.device = "cpu"

    def generate(self, **kwargs):
        input_ids = kwargs["input_ids"]
        suffix = torch.tensor([[10, 11]])
        return torch.cat([input_ids, suffix], dim=1)


def test_advisor_end_to_end_transformers(monkeypatch):
    import implementation.llm as llm_module

    kb = KnowledgeBase.from_csv(Path("updated_optimization_catalog.csv"))

    allowed = sorted(kb.allowed_patterns())
    assert allowed, "Catalog should not be empty."

    valid_pattern = allowed[0]

    fake_response = json.dumps([
        {
            "pattern": valid_pattern,
            "target": "mpi_region:exchange",
            "rationale": "MPI wait is high",
            "action_sketch": "Overlap communication with compute",
            "preconditions": ["MPI semantics preserved"],
            "parameters_to_sweep": {"variant": [1, 2]},
            "correctness_checks": ["Regression tests"],
            "performance_metrics": ["runtime"],
            "risk_level": "medium",
            "rollback_criteria": ["Runtime regression >3%"],
        }
    ])

    fake_tokenizer = FakeTokenizer(fake_response)
    fake_model = FakeModel()

    monkeypatch.setattr(
        llm_module.AutoTokenizer,
        "from_pretrained",
        lambda model_name: fake_tokenizer,
    )

    monkeypatch.setattr(
        llm_module.AutoModelForCausalLM,
        "from_pretrained",
        lambda model_name, torch_dtype="auto", device_map="auto": fake_model,
    )

    llm = TransformersLLM(model_name="fake-llama")

    advisor = MoEAdvisor(
        llm=llm,
        kb=kb,
        prompts_dir=Path("implementation/prompts"),
    )

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

    assert result.final_ranked_candidates, "Should produce final candidates"
    assert result.routing.selected_experts[0] == "Communication & Resilience Expert"
    assert result.final_ranked_candidates[0]["pattern"] == valid_pattern