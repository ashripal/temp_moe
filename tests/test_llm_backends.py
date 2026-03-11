import json

import pytest
import torch

from implementation.llm import (
    LLMMessage,
    MockLLM,
    TransformersLLM,
    _extract_allowed_patterns,
    _pick_pattern,
)


def test_extract_allowed_patterns_parses_json_list() -> None:
    prompt = 'hello\nALLOWED_PATTERNS_JSON=["Async Communication", "Loop Tiling"]\nbye'
    result = _extract_allowed_patterns(prompt)
    assert result == ["Async Communication", "Loop Tiling"]


def test_extract_allowed_patterns_returns_empty_when_missing() -> None:
    prompt = "no allowed patterns here"
    result = _extract_allowed_patterns(prompt)
    assert result == []


def test_extract_allowed_patterns_returns_empty_on_invalid_json() -> None:
    prompt = "ALLOWED_PATTERNS_JSON=[not-valid-json]"
    result = _extract_allowed_patterns(prompt)
    assert result == []


def test_pick_pattern_prefers_keyword_match() -> None:
    allowed = ["Thread Pinning", "Async Communication", "Loop Fusion"]
    result = _pick_pattern(allowed, prefer_keywords=["async", "mpi"])
    assert result == "Async Communication"


def test_pick_pattern_falls_back_to_first_allowed() -> None:
    allowed = ["Thread Pinning", "Loop Fusion"]
    result = _pick_pattern(allowed, prefer_keywords=["checkpoint"])
    assert result == "Thread Pinning"


def test_pick_pattern_returns_unknown_when_empty_allowed() -> None:
    result = _pick_pattern([], prefer_keywords=["anything"])
    assert result == "UNKNOWN_PATTERN"


@pytest.mark.parametrize(
    ("expert_name", "expected_risk"),
    [
        ("Parallelism & Job Expert", "low"),
        ("Communication & Resilience Expert", "medium"),
        ("Kernel & System Efficiency Expert", "low"),
    ],
)
def test_mock_llm_returns_valid_candidate_json_for_each_expert(
    expert_name: str,
    expected_risk: str,
) -> None:
    llm = MockLLM()
    messages = [
        LLMMessage(role="system", content=f"You are {expert_name}."),
        LLMMessage(
            role="user",
            content=(
                'ALLOWED_PATTERNS_JSON=["Async Communication", "Loop Tiling", "Thread Pinning"]\n'
                "Return JSON only."
            ),
        ),
    ]

    raw = llm.complete(messages)
    parsed = json.loads(raw)

    assert isinstance(parsed, list)
    assert len(parsed) == 1

    candidate = parsed[0]
    assert isinstance(candidate, dict)
    assert candidate["pattern"] in ["Async Communication", "Loop Tiling", "Thread Pinning"]
    assert isinstance(candidate["target"], str) and candidate["target"]
    assert isinstance(candidate["rationale"], str) and candidate["rationale"]
    assert isinstance(candidate["action_sketch"], str) and candidate["action_sketch"]
    assert isinstance(candidate["preconditions"], list)
    assert isinstance(candidate["parameters_to_sweep"], dict)
    assert isinstance(candidate["correctness_checks"], list)
    assert isinstance(candidate["performance_metrics"], list)
    assert candidate["risk_level"] == expected_risk
    assert isinstance(candidate["rollback_criteria"], list)


def test_mock_llm_returns_empty_list_for_unknown_expert() -> None:
    llm = MockLLM()
    messages = [
        LLMMessage(role="system", content="You are Some Other Expert."),
        LLMMessage(role="user", content='ALLOWED_PATTERNS_JSON=["A", "B"]'),
    ]

    raw = llm.complete(messages)
    parsed = json.loads(raw)

    assert parsed == []


class FakeBatch(dict):
    def to(self, device):
        return self


class FakeTokenizerWithTemplate:
    def __init__(self, decoded_text: str):
        self.decoded_text = decoded_text
        self.pad_token = None
        self.eos_token = "<eos>"
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.last_chat_messages = None
        self.last_add_generation_prompt = None
        self.last_prompt = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        self.last_chat_messages = messages
        self.last_add_generation_prompt = add_generation_prompt
        return "CHAT_PROMPT"

    def __call__(self, prompt, return_tensors="pt"):
        self.last_prompt = prompt
        return FakeBatch({"input_ids": torch.tensor([[10, 11, 12]])})

    def decode(self, token_ids, skip_special_tokens=True):
        return self.decoded_text


class FakeTokenizerNoTemplate:
    def __init__(self, decoded_text: str):
        self.decoded_text = decoded_text
        self.pad_token = None
        self.eos_token = "<eos>"
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.last_prompt = None

    def __call__(self, prompt, return_tensors="pt"):
        self.last_prompt = prompt
        return FakeBatch({"input_ids": torch.tensor([[20, 21, 22, 23]])})

    def decode(self, token_ids, skip_special_tokens=True):
        return self.decoded_text


class FakeModel:
    def __init__(self):
        self.device = "cpu"
        self.last_generate_kwargs = None

    def generate(self, **kwargs):
        self.last_generate_kwargs = kwargs
        input_ids = kwargs["input_ids"]
        prompt_len = input_ids.shape[1]

        generated_suffix = torch.tensor([[101, 102]])
        return torch.cat([input_ids, generated_suffix], dim=1)


def test_transformers_llm_complete_uses_chat_template(monkeypatch) -> None:
    fake_tokenizer = FakeTokenizerWithTemplate(
        '[{"pattern":"Loop Tiling","target":"loop_1","rationale":"r","action_sketch":"a","preconditions":["p"],"parameters_to_sweep":{"tile":[32,64]},"correctness_checks":["c"],"performance_metrics":["runtime"],"risk_level":"low","rollback_criteria":["rb"]}]'
    )
    fake_model = FakeModel()

    import implementation.llm as llm_module

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
    result = llm.complete(
        [
            LLMMessage(role="system", content="You are an expert."),
            LLMMessage(role="user", content="Return JSON only."),
        ]
    )

    assert result.startswith("[")
    assert fake_tokenizer.last_chat_messages == [
        {"role": "system", "content": "You are an expert."},
        {"role": "user", "content": "Return JSON only."},
    ]
    assert fake_tokenizer.last_add_generation_prompt is True
    assert fake_tokenizer.last_prompt == "CHAT_PROMPT"
    assert fake_model.last_generate_kwargs["max_new_tokens"] == 800
    assert fake_model.last_generate_kwargs["do_sample"] is False
    assert fake_model.last_generate_kwargs["pad_token_id"] == 0
    assert fake_model.last_generate_kwargs["eos_token_id"] == 1


def test_transformers_llm_complete_falls_back_when_no_chat_template(monkeypatch) -> None:
    fake_tokenizer = FakeTokenizerNoTemplate(
        '[{"pattern":"Async Communication","target":"mpi_region","rationale":"r","action_sketch":"a","preconditions":["p"],"parameters_to_sweep":{"variant":[1,2]},"correctness_checks":["c"],"performance_metrics":["runtime"],"risk_level":"medium","rollback_criteria":["rb"]}]'
    )
    fake_model = FakeModel()

    import implementation.llm as llm_module

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
    result = llm.complete(
        [
            LLMMessage(role="system", content="system message"),
            LLMMessage(role="user", content="user message"),
        ]
    )

    assert result.startswith("[")
    assert "SYSTEM:\nsystem message" in fake_tokenizer.last_prompt
    assert "USER:\nuser message" in fake_tokenizer.last_prompt
    assert fake_tokenizer.last_prompt.endswith("ASSISTANT:\n")


def test_transformers_llm_sets_pad_token_to_eos_when_missing(monkeypatch) -> None:
    fake_tokenizer = FakeTokenizerNoTemplate("[]")
    fake_model = FakeModel()

    import implementation.llm as llm_module

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

    _ = TransformersLLM(model_name="fake-llama")
    assert fake_tokenizer.pad_token == fake_tokenizer.eos_token


def test_transformers_llm_complete_raises_on_empty_content(monkeypatch) -> None:
    fake_tokenizer = FakeTokenizerNoTemplate("   ")
    fake_model = FakeModel()

    import implementation.llm as llm_module

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

    with pytest.raises(RuntimeError, match="empty content"):
        llm.complete([LLMMessage(role="user", content="Return JSON only.")])


def test_transformers_llm_complete_enables_sampling_when_temperature_positive(monkeypatch) -> None:
    fake_tokenizer = FakeTokenizerNoTemplate("[]")
    fake_model = FakeModel()

    import implementation.llm as llm_module

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

    llm = TransformersLLM(model_name="fake-llama", temperature=0.7)
    _ = llm.complete([LLMMessage(role="user", content="Return JSON only.")])

    assert fake_model.last_generate_kwargs["do_sample"] is True
    assert fake_model.last_generate_kwargs["temperature"] == 0.7