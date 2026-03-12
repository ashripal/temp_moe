from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
import sys
import os

# Ensure the repository root is on sys.path even when this script is run
# from inside the tests/ directory.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI

from implementation.advisor import MoEAdvisor
from implementation.generator import CodeGenerator, HuggingFaceGeneratorBackend
from implementation.generator.generator_schema import GeneratorInput, GeneratorResult
from implementation.kb import KnowledgeBase
from implementation.llm import LLMClient, LLMMessage


class HuggingFaceExpertLLM(LLMClient):
    """
    Real Hugging Face-backed LLM client for the advisor/expert stage.

    This implementation is stricter than the earlier version:
    - uses a stronger system instruction
    - retries once with a repair prompt if JSON is invalid
    - avoids passing unused sampling kwargs when do_sample=False
    """

    def __init__(
        self,
        model_name: str,
        *,
        device_map: str | None = "auto",
        torch_dtype: str = "auto",
        max_new_tokens: int = 768,
        temperature: float = 0.1,
        top_p: float = 0.9,
        do_sample: bool = False,
        trust_remote_code: bool = False,
    ) -> None:
        self.model_name = model_name
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        self.trust_remote_code = trust_remote_code

        self._tokenizer = None
        self._model = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._load()
        return self._tokenizer

    @property
    def model(self):
        if self._model is None:
            self._load()
        return self._model

    def complete(self, messages: List[LLMMessage]) -> str:
        """
        Produce a completion for the expert stage.

        The expert prompts expect a JSON array of candidate dictionaries.
        This method retries once with a JSON repair instruction if needed.
        """
        # First attempt: normal expert generation.
        raw = self._generate_from_messages(
            messages=messages,
            system_instruction=(
                "You are an HPC optimization expert assistant.\n"
                "Return exactly one valid JSON array and nothing else.\n"
                "The output must start with '[' and end with ']'.\n"
                "Do not include markdown fences, commentary, or explanation.\n"
                "Each element must be a JSON object representing one optimization candidate."
            ),
        )

        cleaned = raw.strip()
        if self._looks_like_json_array(cleaned):
            return cleaned

        # Second attempt: repair pass using the bad output as input.
        repair_messages = [
            LLMMessage(
                role="user",
                content=(
                    "Convert the following text into a valid JSON array only.\n"
                    "Return exactly one valid JSON array and nothing else.\n\n"
                    f"{raw}"
                ),
            )
        ]

        repaired = self._generate_from_messages(
            messages=repair_messages,
            system_instruction=(
                "You are a strict JSON repair assistant.\n"
                "Return exactly one valid JSON array and nothing else.\n"
                "Do not include markdown fences or explanation."
            ),
        )

        return repaired.strip()

    def _generate_from_messages(
        self,
        messages: List[LLMMessage],
        system_instruction: str,
    ) -> str:
        tokenizer = self.tokenizer
        model = self.model

        chat_messages = [{"role": "system", "content": system_instruction}]
        for msg in messages:
            chat_messages.append({"role": msg.role, "content": msg.content})

        if hasattr(tokenizer, "apply_chat_template"):
            model_inputs = tokenizer.apply_chat_template(
                chat_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
        else:
            prompt = system_instruction + "\n\n" + "\n\n".join(
                f"{m.role}: {m.content}" for m in messages
            )
            model_inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
            )

        model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}

        generate_kwargs: Dict[str, Any] = {
            **model_inputs,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }

        if self.do_sample:
            generate_kwargs["temperature"] = self.temperature
            generate_kwargs["top_p"] = self.top_p

        output_ids = model.generate(**generate_kwargs)
        input_len = model_inputs["input_ids"].shape[-1]
        generated_only = output_ids[0][input_len:]
        return tokenizer.decode(generated_only, skip_special_tokens=True).strip()

    def _looks_like_json_array(self, text: str) -> bool:
        text = text.strip()
        if not (text.startswith("[") and text.endswith("]")):
            return False
        try:
            parsed = json.loads(text)
            return isinstance(parsed, list)
        except Exception:
            return False

    def _load(self) -> None:
        import torch

        dtype = self._resolve_torch_dtype(torch)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
            device_map=self.device_map,
            torch_dtype=dtype,
        )

        if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token_id is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    def _resolve_torch_dtype(self, torch_module):
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
    
class OpenAIExpertLLM(LLMClient):
    """
    OpenAI API-backed LLM client for the advisor/expert stage.

    It returns raw text because the expert layer in implementation/experts.py
    already parses and validates the JSON candidate list.
    """

    def __init__(
        self,
        model_name: str = "gpt-5.2",
        api_key: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def complete(self, messages: List[LLMMessage]) -> str:
        system_instruction = (
            "You are an HPC optimization expert assistant.\n"
            "Return exactly one valid JSON array and nothing else.\n"
            "The output must start with '[' and end with ']'.\n"
            "Do not include markdown fences, prose, or explanation.\n"
            "Each array element must be a JSON object representing one optimization candidate."
        )

        input_text = "\n\n".join(f"{m.role}: {m.content}" for m in messages)

        response = self.client.responses.create(
            model=self.model_name,
            instructions=system_instruction,
            input=input_text,
        )
        return response.output_text.strip()
    
class OpenAIGeneratorBackend:
    """
    OpenAI API-backed generator backend.

    This implements the same generate(...) contract that CodeGenerator expects.
    """

    def __init__(
        self,
        prompts_dir: str | Path,
        model_name: str = "gpt-5.2",
        api_key: Optional[str] = None,
    ) -> None:
        from jinja2 import Environment, FileSystemLoader, TemplateNotFound, select_autoescape

        self.prompts_dir = Path(prompts_dir)
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

        self.env = Environment(
            loader=FileSystemLoader(str(self.prompts_dir)),
            autoescape=select_autoescape(disabled_extensions=("jinja",)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def generate(self, generator_input: GeneratorInput) -> GeneratorResult:
        from jinja2 import TemplateNotFound

        from implementation.generator.generator_utils import (
            build_failure_result,
            build_prompt_payload,
            extract_json_object,
            normalize_generator_response,
            sanitize_generated_text,
        )

        selected_candidate = generator_input.selected_candidate

        try:
            payload = build_prompt_payload(generator_input)
            template_name = (
                "generator_feedback_prompt.jinja"
                if payload["has_feedback"]
                else "generator_prompt.jinja"
            )
            template = self.env.get_template(template_name)
            prompt = template.render(**payload)
        except TemplateNotFound as exc:
            return build_failure_result(
                original_code=generator_input.original_code,
                selected_candidate=selected_candidate,
                failure_reason=f"Prompt rendering failed: missing template {exc}",
                used_feedback=bool(generator_input.resolved_feedback_text()),
            )
        except Exception as exc:
            return build_failure_result(
                original_code=generator_input.original_code,
                selected_candidate=selected_candidate,
                failure_reason=f"Prompt rendering failed: {exc}",
                used_feedback=bool(generator_input.resolved_feedback_text()),
            )

        try:
            response = self.client.responses.create(
                model=self.model_name,
                instructions=(
                    "You are an expert HPC code optimization generator. "
                    "Return exactly one valid JSON object and nothing else."
                ),
                input=prompt,
            )
            raw_text = response.output_text
        except Exception as exc:
            return build_failure_result(
                original_code=generator_input.original_code,
                selected_candidate=selected_candidate,
                failure_reason=f"Model inference failed: {exc}",
                used_feedback=bool(generator_input.resolved_feedback_text()),
            )

        cleaned = sanitize_generated_text(raw_text)
        parsed = extract_json_object(cleaned)

        if parsed is None:
            return build_failure_result(
                original_code=generator_input.original_code,
                selected_candidate=selected_candidate,
                failure_reason="Model output could not be parsed into the required JSON schema.",
                used_feedback=bool(generator_input.resolved_feedback_text()),
            )

        result = normalize_generator_response(
            response_dict=parsed,
            selected_candidate=selected_candidate,
            used_feedback=bool(generator_input.resolved_feedback_text()),
        )

        if not result.final_code.strip():
            return build_failure_result(
                original_code=generator_input.original_code,
                selected_candidate=selected_candidate,
                failure_reason="Model returned empty final_code.",
                used_feedback=bool(generator_input.resolved_feedback_text()),
            )

        return result


def benchmark_inputs(benchmark_name: str) -> Dict[str, Any]:
    """
    Provide stable benchmark-specific profiling/telemetry inputs.

    These are simple seed inputs so the advisor routes each benchmark toward the
    intended expert family.
    """
    if benchmark_name == "mpi_pingpong":
        telemetry_struct = {
            "mpi_wait_pct": 40.0,
            "omp_barrier_pct": 0.0,
            "omp_imbalance_ratio": 1.0,
            "memory_bound_score": 0.1,
        }
        profiling_summary = (
            "Hotspots: MPI_Waitall 40%, message exchange dominates runtime, "
            "communication latency is the primary bottleneck."
        )
    elif benchmark_name == "omp_imbalance":
        telemetry_struct = {
            "mpi_wait_pct": 0.0,
            "omp_barrier_pct": 22.0,
            "omp_imbalance_ratio": 1.8,
            "memory_bound_score": 0.2,
        }
        profiling_summary = (
            "Hotspots: OpenMP barrier 22%, parallel region shows load imbalance, "
            "threads spend significant time waiting at synchronization points."
        )
    elif benchmark_name == "mem_saxpy":
        telemetry_struct = {
            "mpi_wait_pct": 0.0,
            "omp_barrier_pct": 0.0,
            "omp_imbalance_ratio": 1.0,
            "memory_bound_score": 0.85,
        }
        profiling_summary = (
            "Hotspots: SAXPY kernel is memory bandwidth bound, low arithmetic intensity, "
            "performance appears limited by cache/memory behavior."
        )
    else:
        telemetry_struct = {
            "mpi_wait_pct": 0.0,
            "omp_barrier_pct": 0.0,
            "omp_imbalance_ratio": 1.0,
            "memory_bound_score": 0.2,
        }
        profiling_summary = "Generic benchmark profile."

    telemetry_summary = ", ".join(f"{k}={v}" for k, v in telemetry_struct.items())

    return {
        "profiling_summary": profiling_summary,
        "telemetry_summary": telemetry_summary,
        "telemetry_struct": telemetry_struct,
    }


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate optimized code for local benchmarks using the MoE advisor + generator pipeline."
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Hugging Face model name for both advisor and generator stages.",
    )
    # parser.add_argument(
    #     "--advisor-model",
    #     default=None,
    #     help="Optional separate model for advisor expert generation.",
    # )
    # parser.add_argument(
    #     "--generator-model",
    #     default=None,
    #     help="Optional separate model for generator code rewrite.",
    # )
    parser.add_argument(
        "--benchmarks",
        nargs="*",
        default=["mem_saxpy", "mpi_pingpong", "omp_imbalance"],
        help="Benchmark folder names under benchmarks/.",
    )
    parser.add_argument(
        "--output-dir",
        default="generated_optimizations",
        help="Directory where generated artifacts will be written.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Transformers device_map value, e.g. auto, cpu, cuda:0.",
    )
    parser.add_argument(
        "--torch-dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Torch dtype for model loading.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing optimized_main.c files if they already exist.",
    )

    parser.add_argument(
        "--provider",
        choices=["openai"],
        default="openai",
        help="LLM provider to use for advisor and generator.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Optional API key override. If omitted, OPENAI_API_KEY is used.",
    )
    parser.add_argument(
        "--advisor-model",
        default="gpt-5.2",
        help="Model for the advisor/expert stage.",
    )
    parser.add_argument(
        "--generator-model",
        default="gpt-5.2",
        help="Model for the generator stage.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    benchmarks_root = repo_root / "benchmarks"
    prompts_dir = repo_root / "implementation" / "prompts"
    generator_prompts_dir = repo_root / "implementation" / "generator" / "prompts"
    catalog_path = repo_root / "updated_optimization_catalog.csv"
    output_root = repo_root / args.output_dir

    advisor_model = args.advisor_model or args.model
    generator_model = args.generator_model or args.model

    kb = KnowledgeBase.from_csv(catalog_path)
    if args.provider == "openai":
        print(f"Using OpenAI models: advisor={advisor_model}, generator={generator_model}")
        
    # advisor_llm = HuggingFaceExpertLLM(
    #     model_name=advisor_model,
    #     device_map=args.device_map,
    #     torch_dtype=args.torch_dtype,
    #     max_new_tokens=768,
    #     do_sample=False,
    # )
    advisor_llm = OpenAIExpertLLM(
        model_name=args.advisor_model,
        api_key=args.api_key
    )

    advisor = MoEAdvisor(
        llm=advisor_llm,
        kb=kb,
        prompts_dir=prompts_dir,
    )

    # generator_backend = HuggingFaceGeneratorBackend(
    #     prompts_dir=generator_prompts_dir,
    #     model_name=generator_model,
    #     device_map=args.device_map,
    #     torch_dtype=args.torch_dtype,
    #     max_new_tokens=1024,
    #     do_sample=False,
    # )
    generator_backend = OpenAIGeneratorBackend(
        prompts_dir=generator_prompts_dir,
        model_name=args.generator_model,
        api_key=args.api_key
    )
    generator = CodeGenerator(backend=generator_backend)

    output_root.mkdir(parents=True, exist_ok=True)

    for bench_name in args.benchmarks:
        try:
            bench_dir = benchmarks_root / bench_name
            code_path = bench_dir / "main.c"

            if not code_path.exists():
                print(f"[skip] {bench_name}: missing {code_path}")
                continue

            code_text = code_path.read_text(encoding="utf-8")
            bench_meta = benchmark_inputs(bench_name)

            print(f"[run] {bench_name}")
            advisor_result = advisor.run(
                code_snippets=code_text,
                profiling_summary=bench_meta["profiling_summary"],
                telemetry_summary=bench_meta["telemetry_summary"],
                telemetry_struct=bench_meta["telemetry_struct"],
            )

            generator_input = generator.from_advisor_result(
                advisor_result=advisor_result,
                original_code=code_text,
                profiling_summary=bench_meta["profiling_summary"],
                telemetry_summary=bench_meta["telemetry_summary"],
                telemetry_struct=bench_meta["telemetry_struct"],
                ast=None,
                flame_report=None,
            )

            generator_result = generator.generate(generator_input)

            bench_output_dir = output_root / bench_name
            bench_output_dir.mkdir(parents=True, exist_ok=True)

            optimized_code_path = bench_output_dir / "optimized_main.c"
            if optimized_code_path.exists() and not args.overwrite:
                print(f"[skip] {bench_name}: {optimized_code_path} exists (use --overwrite)")
            else:
                optimized_code_path.write_text(generator_result.final_code, encoding="utf-8")

            advisor_json = {
                "routing": asdict(advisor_result.routing),
                "expert_outputs": [eo.to_dict() for eo in advisor_result.expert_outputs],
                "final_ranked_candidates": advisor_result.final_ranked_candidates,
            }

            write_json(bench_output_dir / "advisor_result.json", advisor_json)
            write_json(bench_output_dir / "generator_input.json", generator_input.to_dict())
            write_json(bench_output_dir / "generator_result.json", generator_result.to_dict())

            print(
                f"[done] {bench_name}: success={generator_result.generation_succeeded} "
                f"output={optimized_code_path}"
            )

        except Exception as exc:
            print(f"[error] {bench_name}: {exc}")
            continue

        generator_result = generator.generate(generator_input)

        bench_output_dir = output_root / bench_name
        bench_output_dir.mkdir(parents=True, exist_ok=True)

        optimized_code_path = bench_output_dir / "optimized_main.c"
        if optimized_code_path.exists() and not args.overwrite:
            print(f"[skip] {bench_name}: {optimized_code_path} exists (use --overwrite)")
        else:
            optimized_code_path.write_text(generator_result.final_code, encoding="utf-8")

        advisor_json = {
            "routing": asdict(advisor_result.routing),
            "expert_outputs": [eo.to_dict() for eo in advisor_result.expert_outputs],
            "final_ranked_candidates": advisor_result.final_ranked_candidates,
        }
        write_json(bench_output_dir / "advisor_result.json", advisor_json)
        write_json(bench_output_dir / "generator_input.json", generator_input.to_dict())
        write_json(bench_output_dir / "generator_result.json", generator_result.to_dict())

        print(
            f"[done] {bench_name}: success={generator_result.generation_succeeded} "
            f"output={optimized_code_path}"
        )


if __name__ == "__main__":
    main()