from __future__ import annotations

from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from implementation.generator.generator_llm import HuggingFaceGeneratorBackend
from implementation.generator.generator_schema import GeneratorInput


# Build one valid advisor candidate that matches the schema expected by the
# generator package. Reusing this helper keeps each test focused on backend
# behavior rather than schema setup.
def _valid_candidate() -> dict:
    return {
        "pattern": "MPI_WAIT_REDUCTION",
        "target": "main timestep loop",
        "rationale": "High MPI_Waitall time suggests excessive synchronization overhead.",
        "action_sketch": "Fuse communication phases and reduce unnecessary waits.",
        "preconditions": ["Communication pattern is safe to reorder locally."],
        "parameters_to_sweep": {
            "batch_size": [16, 32, 64],
            "message_grouping": [1, 2, 4],
        },
        "correctness_checks": ["Results match baseline output."],
        "performance_metrics": ["mpi_wait_pct", "runtime_seconds"],
        "risk_level": "medium",
        "rollback_criteria": [
            "Rollback if runtime increases.",
            "Rollback if outputs diverge from baseline.",
        ],
    }


# Build a minimal valid GeneratorInput object.
def _generator_input(
    *,
    evaluator_feedback: str | None = None,
) -> GeneratorInput:
    candidate = _valid_candidate()
    return GeneratorInput(
        original_code="int main() { return 0; }",
        profiling_summary="MPI_Waitall dominates runtime.",
        telemetry_summary="mpi_wait_pct=42",
        telemetry_struct={"mpi_wait_pct": 42.0},
        ranked_candidates=[candidate],
        selected_candidate=candidate,
        ast="FunctionDecl(main)",
        flame_report="main 80%",
        evaluator_feedback=evaluator_feedback,
        evaluation_feedback=None,
    )


# Create temporary prompt templates used only for these backend tests.
# Keeping the prompts very small makes the tests easier to reason about.
def _write_test_prompts(tmp_path: Path) -> Path:
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    (prompts_dir / "generator_prompt.jinja").write_text(
        (
            "FIRST PASS\n"
            "Code:\n{{ original_code }}\n"
            "Selected:\n{{ selected_candidate_text }}\n"
        ),
        encoding="utf-8",
    )

    (prompts_dir / "generator_feedback_prompt.jinja").write_text(
        (
            "FEEDBACK PASS\n"
            "Code:\n{{ original_code }}\n"
            "Feedback:\n{{ evaluator_feedback }}\n"
            "Selected:\n{{ selected_candidate_text }}\n"
        ),
        encoding="utf-8",
    )

    return prompts_dir


# Create a real, tiny Hugging Face model and tokenizer on disk.
# This is not a mock model: it uses real Transformers classes and a real
# tokenizer implementation, but with a very small random-weight GPT-2 model
# so the tests stay lightweight.
def _write_tiny_hf_model(tmp_path: Path) -> Path:
    model_dir = tmp_path / "tiny_hf_model"
    model_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)

    # Keep the vocab simple and include a real UNK token so unexpected prompt
    # text still maps to a valid token id.
    vocab = {
        "<pad>": 0,
        "<bos>": 1,
        "<eos>": 2,
        "[UNK]": 3,
        "system": 4,
        "user": 5,
        "assistant": 6,
        ":": 7,
        "\n": 8,
        "FIRST": 9,
        "PASS": 10,
        "FEEDBACK": 11,
        "Code": 12,
        "Selected": 13,
        "Feedback": 14,
        "int": 15,
        "main": 16,
        "(": 17,
        ")": 18,
        "{": 19,
        "}": 20,
        "return": 21,
        "0": 22,
        ";": 23,
        "MPI_WAIT_REDUCTION": 24,
    }

    tokenizer_obj = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer_obj.pre_tokenizer = Whitespace()

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_obj,
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="[UNK]",
        pad_token="<pad>",
    )

    # Add a simple chat template so apply_chat_template() works in the backend.
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{ message['role'] }}: {{ message['content'] }}\n"
        "{% endfor %}"
        "{% if add_generation_prompt %}assistant: {% endif %}"
    )

    # Important fix:
    # Use a larger context window so prompt tokenization does not overflow the
    # tiny GPT-2 positional embeddings during generate().
    config = GPT2Config(
        vocab_size=len(vocab),
        n_positions=512,
        n_ctx=512,
        n_embd=32,
        n_layer=1,
        n_head=1,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = GPT2LMHeadModel(config)

    tokenizer.save_pretrained(model_dir)
    model.save_pretrained(model_dir)

    return model_dir


def test_backend_loads_real_transformers_model_and_tokenizer(tmp_path: Path) -> None:
    # This test verifies that the backend really loads a Hugging Face tokenizer
    # and model from disk through AutoTokenizer / AutoModelForCausalLM.
    prompts_dir = _write_test_prompts(tmp_path)
    model_dir = _write_tiny_hf_model(tmp_path)

    backend = HuggingFaceGeneratorBackend(
        prompts_dir=prompts_dir,
        model_name=str(model_dir),
        device_map=None,
        torch_dtype="float32",
        max_new_tokens=8,
        do_sample=False,
    )

    tokenizer = backend.tokenizer
    model = backend.model

    assert tokenizer is not None
    assert model is not None
    assert model.config.vocab_size > 0
    assert model.__class__.__name__ == "GPT2LMHeadModel"


def test_render_prompt_uses_first_pass_template_without_feedback(tmp_path: Path) -> None:
    # If no evaluator feedback is present, the backend should render the
    # first-pass generation template.
    prompts_dir = _write_test_prompts(tmp_path)
    model_dir = _write_tiny_hf_model(tmp_path)

    backend = HuggingFaceGeneratorBackend(
        prompts_dir=prompts_dir,
        model_name=str(model_dir),
        device_map=None,
        torch_dtype="float32",
        max_new_tokens=8,
        do_sample=False,
    )

    generator_input = _generator_input()
    rendered = backend._render_prompt(generator_input)

    assert "FIRST PASS" in rendered
    assert "FEEDBACK PASS" not in rendered
    assert "int main() { return 0; }" in rendered
    assert "MPI_WAIT_REDUCTION" in rendered


def test_render_prompt_uses_feedback_template_when_feedback_exists(tmp_path: Path) -> None:
    # If evaluator feedback is present, the backend should switch to the
    # feedback prompt template.
    prompts_dir = _write_test_prompts(tmp_path)
    model_dir = _write_tiny_hf_model(tmp_path)

    backend = HuggingFaceGeneratorBackend(
        prompts_dir=prompts_dir,
        model_name=str(model_dir),
        device_map=None,
        torch_dtype="float32",
        max_new_tokens=8,
        do_sample=False,
    )

    generator_input = _generator_input(
        evaluator_feedback="Compilation failed due to undeclared variable tmp."
    )
    rendered = backend._render_prompt(generator_input)

    assert "FEEDBACK PASS" in rendered
    assert "FIRST PASS" not in rendered
    assert "Compilation failed due to undeclared variable tmp." in rendered


def test_tokenize_prompt_returns_real_tensors(tmp_path: Path) -> None:
    # This test checks that prompt tokenization runs through the real tokenizer
    # and returns PyTorch tensors that can be passed into model.generate().
    prompts_dir = _write_test_prompts(tmp_path)
    model_dir = _write_tiny_hf_model(tmp_path)

    backend = HuggingFaceGeneratorBackend(
        prompts_dir=prompts_dir,
        model_name=str(model_dir),
        device_map=None,
        torch_dtype="float32",
        max_new_tokens=8,
        do_sample=False,
    )

    model_inputs = backend._tokenize_prompt("Hello world")

    assert "input_ids" in model_inputs
    assert "attention_mask" in model_inputs
    assert model_inputs["input_ids"].ndim == 2
    assert model_inputs["attention_mask"].ndim == 2
    assert model_inputs["input_ids"].shape[0] == 1


def test_generate_returns_safe_failure_when_real_model_output_is_not_json(
    tmp_path: Path,
) -> None:
    # The tiny random model is a real Transformers model, but it will not
    # reliably emit the strict JSON schema the generator expects.
    #
    # Depending on tokenizer/model details, the backend may fail in one of two
    # safe ways:
    #   1. inference succeeds, but output is not parseable JSON
    #   2. inference itself fails and the backend returns a safe fallback
    #
    # Either outcome is acceptable here because this test is checking that the
    # backend fails safely and returns unchanged code.
    prompts_dir = _write_test_prompts(tmp_path)
    model_dir = _write_tiny_hf_model(tmp_path)

    backend = HuggingFaceGeneratorBackend(
        prompts_dir=prompts_dir,
        model_name=str(model_dir),
        device_map=None,
        torch_dtype="float32",
        max_new_tokens=8,
        do_sample=False,
    )

    generator_input = _generator_input()
    result = backend.generate(generator_input)

    assert result.generation_succeeded is False
    assert result.final_code == generator_input.original_code
    assert result.failure_reason is not None
    assert (
        "JSON" in result.failure_reason
        or "parsed" in result.failure_reason
        or "Model inference failed" in result.failure_reason
    )


def test_generate_uses_feedback_flag_in_failure_result(tmp_path: Path) -> None:
    # When generation fails during a feedback pass, the returned failure result
    # should still record that feedback was used.
    prompts_dir = _write_test_prompts(tmp_path)
    model_dir = _write_tiny_hf_model(tmp_path)

    backend = HuggingFaceGeneratorBackend(
        prompts_dir=prompts_dir,
        model_name=str(model_dir),
        device_map=None,
        torch_dtype="float32",
        max_new_tokens=8,
        do_sample=False,
    )

    generator_input = _generator_input(
        evaluator_feedback="Correctness regression detected."
    )
    result = backend.generate(generator_input)

    assert result.generation_succeeded is False
    assert result.used_feedback is True
    assert result.final_code == generator_input.original_code


def test_generate_returns_safe_failure_when_prompt_template_is_missing(
    tmp_path: Path,
) -> None:
    # If the required template file is missing, generate() should not crash the
    # pipeline. It should return unchanged code with a structured failure reason.
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    (prompts_dir / "generator_feedback_prompt.jinja").write_text(
        "FEEDBACK PASS\n{{ evaluator_feedback }}\n",
        encoding="utf-8",
    )

    model_dir = _write_tiny_hf_model(tmp_path)

    backend = HuggingFaceGeneratorBackend(
        prompts_dir=prompts_dir,
        model_name=str(model_dir),
        device_map=None,
        torch_dtype="float32",
        max_new_tokens=8,
        do_sample=False,
    )

    generator_input = _generator_input()
    result = backend.generate(generator_input)

    assert result.generation_succeeded is False
    assert result.final_code == generator_input.original_code
    assert result.failure_reason is not None
    assert "Prompt rendering failed" in result.failure_reason


def test_resolve_torch_dtype_supports_expected_values(tmp_path: Path) -> None:
    # This test verifies the dtype mapping helper used during model loading.
    prompts_dir = _write_test_prompts(tmp_path)
    model_dir = _write_tiny_hf_model(tmp_path)

    backend = HuggingFaceGeneratorBackend(
        prompts_dir=prompts_dir,
        model_name=str(model_dir),
        device_map=None,
        torch_dtype="float32",
        max_new_tokens=8,
        do_sample=False,
    )

    assert backend._resolve_torch_dtype(torch) == torch.float32

    backend.torch_dtype = "float16"
    assert backend._resolve_torch_dtype(torch) == torch.float16

    backend.torch_dtype = "auto"
    assert backend._resolve_torch_dtype(torch) == "auto"


def test_resolve_torch_dtype_rejects_invalid_value(tmp_path: Path) -> None:
    # Invalid dtype names should raise a ValueError instead of silently choosing
    # an unexpected precision mode.
    prompts_dir = _write_test_prompts(tmp_path)
    model_dir = _write_tiny_hf_model(tmp_path)

    backend = HuggingFaceGeneratorBackend(
        prompts_dir=prompts_dir,
        model_name=str(model_dir),
        device_map=None,
        torch_dtype="not_a_real_dtype",
        max_new_tokens=8,
        do_sample=False,
    )

    try:
        backend._resolve_torch_dtype(torch)
        assert False, "Expected ValueError for invalid torch_dtype."
    except ValueError as exc:
        assert "Unsupported torch_dtype" in str(exc)