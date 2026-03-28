from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List, Protocol

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class LLMMessage:
    role: str
    content: str


class LLMClient(Protocol):
    def complete(self, messages: List[LLMMessage]) -> str:
        ...


def _extract_allowed_patterns(prompt: str) -> List[str]:
    """
    Extract ALLOWED_PATTERNS_JSON=[...] from the rendered expert prompt.

    The prompt templates are expected to serialize a JSON array onto one line or
    across multiple lines. If parsing fails, return an empty list.
    """
    match = re.search(r"ALLOWED_PATTERNS_JSON=(\[[\s\S]*?\])", prompt)
    if not match:
        return []

    try:
        parsed = json.loads(match.group(1))
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except Exception:
        pass

    return []


def _pick_pattern(allowed: List[str], prefer_keywords: List[str]) -> str:
    """
    Deterministically pick the first allowed pattern whose normalized text
    contains one of the preferred keywords. Fallback to the first allowed item.
    """
    allowed_lower = [(pattern, pattern.lower()) for pattern in allowed]

    for keyword in prefer_keywords:
        keyword = keyword.lower()
        for pattern, pattern_lower in allowed_lower:
            if keyword in pattern_lower:
                return pattern

    return allowed[0] if allowed else "UNKNOWN_PATTERN"


class MockLLM:
    """
    Deterministic offline LLM for testing.

    It only chooses from ALLOWED_PATTERNS_JSON, which now should typically be the
    retrieved subset rather than the full catalog. This makes unit tests and dry
    runs better reflect the real MoE flow.
    """

    def complete(self, messages: List[LLMMessage]) -> str:
        prompt = "\n".join(m.content for m in messages)
        allowed = _extract_allowed_patterns(prompt)

        if "Parallelism & Job Expert" in prompt:
            pattern = _pick_pattern(
                allowed,
                prefer_keywords=[
                    "openmp",
                    "parallel",
                    "thread",
                    "barrier",
                    "imbalance",
                    "load balance",
                    "schedule",
                    "scheduling",
                    "collapse",
                    "reduction",
                    "numa",
                    "affinity",
                    "rank placement",
                    "lock",
                    "synchronization",
                ],
            )
            return json.dumps(
                [
                    {
                        "pattern": pattern,
                        "target": "foo.c:bar():loop_12",
                        "rationale": (
                            "Telemetry suggests synchronization overhead or "
                            "parallel imbalance is a plausible optimization lever."
                        ),
                        "action_sketch": (
                            "Apply the selected catalog pattern conservatively and "
                            "tune related scheduling or placement parameters."
                        ),
                        "preconditions": [
                            "No semantic changes",
                            "Preserve correctness checks",
                        ],
                        "parameters_to_sweep": {
                            "variant": [1, 2]
                        },
                        "correctness_checks": [
                            "Run regression tests",
                            "Compare key outputs within tolerance",
                        ],
                        "performance_metrics": [
                            "runtime",
                            "scaling_efficiency",
                            "omp_barrier_pct",
                        ],
                        "risk_level": "low",
                        "rollback_criteria": [
                            "Correctness failure",
                            "Runtime regression > 3%",
                        ],
                    }
                ]
            )

        if "Communication & Resilience Expert" in prompt:
            pattern = _pick_pattern(
                allowed,
                prefer_keywords=[
                    "mpi",
                    "communication",
                    "message",
                    "non-blocking",
                    "async",
                    "asynchronous",
                    "collective",
                    "halo",
                    "overlap",
                    "checkpoint",
                    "resilience",
                    "latency",
                ],
            )
            return json.dumps(
                [
                    {
                        "pattern": pattern,
                        "target": "mpi_region:exchange_halos",
                        "rationale": (
                            "Telemetry suggests communication overhead or MPI wait "
                            "time is a likely bottleneck."
                        ),
                        "action_sketch": (
                            "Apply the selected communication-oriented catalog pattern "
                            "conservatively and validate message ordering/correctness."
                        ),
                        "preconditions": [
                            "Preserve ordering/semantics",
                            "Validate correctness after change",
                        ],
                        "parameters_to_sweep": {
                            "variant": [1, 2]
                        },
                        "correctness_checks": [
                            "Run regression tests",
                            "Compare outputs within tolerance",
                        ],
                        "performance_metrics": [
                            "runtime",
                            "mpi_wait_pct",
                            "throughput",
                            "scaling_efficiency",
                        ],
                        "risk_level": "medium",
                        "rollback_criteria": [
                            "Correctness failure",
                            "Hang/deadlock",
                            "Runtime regression > 3%",
                        ],
                    }
                ]
            )

        if "Kernel & System Efficiency Expert" in prompt:
            pattern = _pick_pattern(
                allowed,
                prefer_keywords=[
                    "cache",
                    "memory",
                    "locality",
                    "prefetch",
                    "vector",
                    "vectorization",
                    "simd",
                    "loop",
                    "tiling",
                    "unroll",
                    "bandwidth",
                    "precision",
                    "compiler",
                ],
            )
            return json.dumps(
                [
                    {
                        "pattern": pattern,
                        "target": "kernel.c:matmul():loop_3",
                        "rationale": (
                            "Telemetry suggests a kernel-level hotspot with likely "
                            "memory or locality inefficiency."
                        ),
                        "action_sketch": (
                            "Apply the selected catalog pattern conservatively with "
                            "a small parameter sweep around locality or vectorization."
                        ),
                        "preconditions": [
                            "No out-of-bounds accesses",
                            "Validate numeric tolerance if floating-point behavior changes",
                        ],
                        "parameters_to_sweep": {
                            "variant": [1, 2]
                        },
                        "correctness_checks": [
                            "Run regression tests",
                            "Numeric tolerance check",
                        ],
                        "performance_metrics": [
                            "runtime",
                            "cache",
                            "memory",
                        ],
                        "risk_level": "low",
                        "rollback_criteria": [
                            "Correctness failure",
                            "Runtime regression > 3%",
                        ],
                    }
                ]
            )

        return "[]"


class TransformersLLM:
    """
    Local Hugging Face Transformers-backed LLM client.

    Interface:
        complete(messages) -> raw text

    Downstream expert code is responsible for parsing and validating the JSON.
    """

    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = 800,
        temperature: float = 0.0,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
        )

        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _build_prompt(self, messages: List[LLMMessage]) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            chat_messages = [{"role": m.role, "content": m.content} for m in messages]
            return self.tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        parts: List[str] = []
        for m in messages:
            parts.append(f"{m.role.upper()}:\n{m.content}")
        parts.append("ASSISTANT:\n")
        return "\n\n".join(parts)

    def complete(self, messages: List[LLMMessage]) -> str:
        prompt = self._build_prompt(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]

        do_sample = self.temperature > 0.0
        generate_kwargs = {
            **inputs,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        if do_sample:
            generate_kwargs["temperature"] = self.temperature

        with torch.no_grad():
            outputs = self.model.generate(**generate_kwargs)

        generated_ids = outputs[0][prompt_len:]
        content = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()

        if not content:
            raise RuntimeError("Transformers model returned empty content.")

        return content