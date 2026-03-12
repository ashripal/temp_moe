from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from jinja2 import Environment, FileSystemLoader, TemplateNotFound, select_autoescape

from .generator_schema import GeneratorInput, GeneratorResult
from .generator_utils import (
    build_failure_result,
    build_prompt_payload,
    extract_json_object,
    normalize_generator_response,
    sanitize_generated_text,
)

LOGGER = logging.getLogger(__name__)


class HuggingFaceGeneratorBackend:
    """
    Real generator backend using the transformers library and Hugging Face models.

    This backend:
      - renders Jinja prompts
      - loads a chat/instruct model via transformers
      - generates a response with model.generate()
      - parses structured JSON into GeneratorResult

    It is intended for real model inference, not mocking.
    """

    def __init__(
        self,
        prompts_dir: str | Path,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        *,
        device_map: str = "auto",
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
        top_p: float = 0.9,
        do_sample: bool = False,
        trust_remote_code: bool = False,
        torch_dtype: str = "auto",
    ) -> None:
        self.prompts_dir = Path(prompts_dir)
        self.model_name = model_name
        self.device_map = device_map
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        self.trust_remote_code = trust_remote_code
        self.torch_dtype = torch_dtype

        self.env = Environment(
            loader=FileSystemLoader(str(self.prompts_dir)),
            autoescape=select_autoescape(disabled_extensions=("jinja",)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

        self._tokenizer = None
        self._model = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._load_model_and_tokenizer()
        return self._tokenizer

    @property
    def model(self):
        if self._model is None:
            self._load_model_and_tokenizer()
        return self._model

    def generate(self, generator_input: GeneratorInput) -> GeneratorResult:
        """
        Run one generator pass with a real Hugging Face model.

        Uses:
          - generator_prompt.jinja for first pass
          - generator_feedback_prompt.jinja when evaluator feedback exists
        """
        selected_candidate = generator_input.selected_candidate

        try:
            prompt = self._render_prompt(generator_input)
        except Exception as exc:
            LOGGER.exception("Failed to render generator prompt.")
            return build_failure_result(
                original_code=generator_input.original_code,
                selected_candidate=selected_candidate,
                failure_reason=f"Prompt rendering failed: {exc}",
                used_feedback=bool(generator_input.resolved_feedback_text()),
            )

        try:
            raw_text = self._run_model(prompt)
        except Exception as exc:
            LOGGER.exception("Generator model inference failed.")
            return build_failure_result(
                original_code=generator_input.original_code,
                selected_candidate=selected_candidate,
                failure_reason=f"Model inference failed: {exc}",
                used_feedback=bool(generator_input.resolved_feedback_text()),
            )

        cleaned = sanitize_generated_text(raw_text)
        parsed = extract_json_object(cleaned)

        if parsed is None:
            LOGGER.error("Generator output was not valid JSON.")
            return build_failure_result(
                original_code=generator_input.original_code,
                selected_candidate=selected_candidate,
                failure_reason=(
                    "Model output could not be parsed into the required JSON schema."
                ),
                used_feedback=bool(generator_input.resolved_feedback_text()),
            )

        result = normalize_generator_response(
            response_dict=parsed,
            selected_candidate=selected_candidate,
            used_feedback=bool(generator_input.resolved_feedback_text()),
        )

        if not result.final_code.strip():
            LOGGER.error("Generator output contained empty final_code.")
            return build_failure_result(
                original_code=generator_input.original_code,
                selected_candidate=selected_candidate,
                failure_reason="Model returned empty final_code.",
                used_feedback=bool(generator_input.resolved_feedback_text()),
            )

        return result

    def _render_prompt(self, generator_input: GeneratorInput) -> str:
        payload = build_prompt_payload(generator_input)
        template_name = (
            "generator_feedback_prompt.jinja"
            if payload["has_feedback"]
            else "generator_prompt.jinja"
        )

        try:
            template = self.env.get_template(template_name)
        except TemplateNotFound as exc:
            raise FileNotFoundError(
                f"Prompt template '{template_name}' not found in {self.prompts_dir}"
            ) from exc

        return template.render(**payload)

    def _run_model(self, prompt: str) -> str:
        """
        Generate text from the Hugging Face model and return only new tokens.

        Uses tokenizer chat templates when supported, which is the recommended
        approach for chat/instruct models in Transformers. :contentReference[oaicite:0]{index=0}
        """
        tokenizer = self.tokenizer
        model = self.model
        model_inputs = self._tokenize_prompt(prompt)

        generate_kwargs: Dict[str, Any] = {
            **model_inputs,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if self.do_sample:
            generate_kwargs["temperature"] = self.temperature
            generate_kwargs["top_p"] = self.top_p

        output_ids = model.generate(**generate_kwargs)

        input_length = model_inputs["input_ids"].shape[-1]
        generated_only = output_ids[0][input_length:]

        return tokenizer.decode(generated_only, skip_special_tokens=True)

    def _tokenize_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        Build model inputs, preferring apply_chat_template when available.

        Transformers recommends formatting chat-model prompts with the tokenizer's
        chat template and using add_generation_prompt=True for generation. :contentReference[oaicite:1]{index=1}
        """
        tokenizer = self.tokenizer
        model = self.model

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert HPC code optimization generator. "
                    "Return exactly one valid JSON object and nothing else."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        if hasattr(tokenizer, "apply_chat_template"):
            model_inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
        else:
            fallback_prompt = (
                "You are an expert HPC code optimization generator. "
                "Return exactly one valid JSON object and nothing else.\n\n"
                f"{prompt}"
            )
            model_inputs = tokenizer(
                fallback_prompt,
                return_tensors="pt",
                truncation=True,
            )

        return {key: value.to(model.device) for key, value in model_inputs.items()}

    def _load_model_and_tokenizer(self) -> None:
        """
        Lazy-load the tokenizer and model from Hugging Face via transformers.
        """
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "generator_llm.py requires the transformers and torch packages."
            ) from exc

        dtype = self._resolve_torch_dtype(torch)

        LOGGER.info("Loading Hugging Face generator model: %s", self.model_name)

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
            device_map=self.device_map,
            torch_dtype=dtype,
        )

        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token

        self._tokenizer = tokenizer
        self._model = model

    def _resolve_torch_dtype(self, torch_module):
        """
        Convert the configured dtype string into a torch dtype.
        """
        if self.torch_dtype == "auto":
            return "auto"

        mapping = {
            "float16": torch_module.float16,
            "bfloat16": getattr(torch_module, "bfloat16", None),
            "float32": torch_module.float32,
        }

        if self.torch_dtype not in mapping or mapping[self.torch_dtype] is None:
            raise ValueError(
                f"Unsupported torch_dtype '{self.torch_dtype}'. "
                "Use one of: auto, float16, bfloat16, float32."
            )

        return mapping[self.torch_dtype]