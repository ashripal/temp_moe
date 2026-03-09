"""
profiler_parser.py

Parses raw profiling or benchmark outputs into a normalized ProfilingSummary.

This module is intentionally lightweight for the current temp_moe repository.
At this stage, it is designed to handle simple benchmark/runtime outputs rather
than full profiler traces from tools like perf, VTune, or HPCToolkit.

Current responsibilities:
- Accept raw profiling text, benchmark logs, or simple metric dictionaries
- Extract key runtime-oriented signals when present
- Normalize them into a ProfilingSummary object
- Produce a compact human-readable summary for downstream prompting

Design philosophy:
- Be permissive with input formats
- Keep parsing logic deterministic and easy to debug
- Provide useful defaults when information is missing
- Avoid overfitting to one profiler format too early
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .analysis_bundle import ProfilingSummary


class ProfilerParser:
    """
    Lightweight parser for profiling and benchmark outputs.

    This parser currently supports three main input modes:
    1. Raw text logs
    2. File paths to logs
    3. Precomputed metric dictionaries

    The goal is to produce a stable ProfilingSummary regardless of how the
    performance data was collected.
    """

    # Common runtime patterns that may appear in logs.
    _RUNTIME_PATTERNS = [
        re.compile(r"\bruntime\s*[:=]\s*([0-9]*\.?[0-9]+)\s*(s|sec|secs|seconds)?\b", re.IGNORECASE),
        re.compile(r"\btime\s*[:=]\s*([0-9]*\.?[0-9]+)\s*(s|sec|secs|seconds)?\b", re.IGNORECASE),
        re.compile(r"\belapsed\s*time\s*[:=]\s*([0-9]*\.?[0-9]+)\s*(s|sec|secs|seconds)?\b", re.IGNORECASE),
        re.compile(r"\bmean\s*time\s*\(s\)\s*[:=]?\s*([0-9]*\.?[0-9]+)\b", re.IGNORECASE),
    ]

    # A few generic hotspot-style patterns for future extensibility.
    _HOTSPOT_PATTERNS = [
        re.compile(r"\bhotspot\s*[:=]\s*([A-Za-z_][A-Za-z0-9_]*)\b", re.IGNORECASE),
        re.compile(r"\bhot\s*function\s*[:=]\s*([A-Za-z_][A-Za-z0-9_]*)\b", re.IGNORECASE),
        re.compile(r"\btop\s*function\s*[:=]\s*([A-Za-z_][A-Za-z0-9_]*)\b", re.IGNORECASE),
    ]

    # Useful keywords that may hint at communication, imbalance, or memory behavior.
    _KEYWORD_GROUPS = {
        "communication": [
            "mpi",
            "barrier",
            "wait",
            "allreduce",
            "send",
            "recv",
            "communication",
        ],
        "parallelism": [
            "thread",
            "openmp",
            "imbalance",
            "scheduling",
            "parallel",
            "core",
        ],
        "memory": [
            "memory",
            "bandwidth",
            "cache",
            "miss",
            "load/store",
            "stream",
        ],
    }

    def parse_from_text(self, raw_text: str) -> ProfilingSummary:
        """
        Parse a raw profiling/benchmark text blob into a ProfilingSummary.

        Parameters
        ----------
        raw_text:
            Raw text captured from benchmark output, a profiler report, or
            another performance-related source.

        Returns
        -------
        ProfilingSummary
            Normalized profiling summary suitable for downstream stages.
        """
        cleaned_text = raw_text.strip()
        notes: List[str] = []

        if not cleaned_text:
            notes.append("Received empty profiling text; returning default profiling summary.")
            return ProfilingSummary(
                runtime_seconds=None,
                hotspot_function=None,
                hotspot_description="",
                raw_metrics={},
                summary_text="No profiling data was available.",
                notes=notes,
            )

        runtime_seconds = self._extract_runtime_seconds(cleaned_text)
        hotspot_function = self._extract_hotspot_function(cleaned_text)
        inferred_signals = self._infer_keyword_signals(cleaned_text)

        raw_metrics: Dict[str, Any] = dict(inferred_signals)
        if runtime_seconds is not None:
            raw_metrics["runtime_seconds"] = runtime_seconds
        if hotspot_function is not None:
            raw_metrics["hotspot_function"] = hotspot_function

        hotspot_description = self._build_hotspot_description(
            hotspot_function=hotspot_function,
            inferred_signals=inferred_signals,
        )

        summary_text = self._build_summary_text(
            runtime_seconds=runtime_seconds,
            hotspot_function=hotspot_function,
            inferred_signals=inferred_signals,
        )

        return ProfilingSummary(
            runtime_seconds=runtime_seconds,
            hotspot_function=hotspot_function,
            hotspot_description=hotspot_description,
            raw_metrics=raw_metrics,
            summary_text=summary_text,
            notes=notes,
        )

    def parse_from_file(self, log_path: str | Path) -> ProfilingSummary:
        """
        Read profiling text from disk and parse it.

        Parameters
        ----------
        log_path:
            Path to a text log or report file.

        Returns
        -------
        ProfilingSummary
            Parsed profiling summary.
        """
        log_path = Path(log_path)

        if not log_path.exists():
            raise FileNotFoundError(f"Profiling log not found: {log_path}")

        raw_text = log_path.read_text(encoding="utf-8")
        summary = self.parse_from_text(raw_text)

        # Preserve provenance for debugging / artifact tracking.
        summary.notes.append(f"Parsed profiling data from file: {log_path}")
        return summary

    def parse_from_metrics(self, metrics: Dict[str, Any]) -> ProfilingSummary:
        """
        Normalize a precomputed metrics dictionary into a ProfilingSummary.

        This is useful when the benchmark harness already produces structured
        results and we want to avoid reparsing text.

        Expected keys are flexible, but the parser will look for common fields
        like:
            - runtime_seconds
            - runtime
            - hotspot_function
            - hotspot
        """
        normalized = dict(metrics)
        notes: List[str] = ["Profiling summary created from precomputed metrics dictionary."]

        runtime_seconds = self._coerce_runtime_seconds(normalized)
        hotspot_function = self._coerce_hotspot_function(normalized)

        if runtime_seconds is not None:
            normalized["runtime_seconds"] = runtime_seconds
        if hotspot_function is not None:
            normalized["hotspot_function"] = hotspot_function

        inferred_signals = self._infer_signals_from_metrics(normalized)
        normalized.update(inferred_signals)

        hotspot_description = self._build_hotspot_description(
            hotspot_function=hotspot_function,
            inferred_signals=inferred_signals,
        )

        summary_text = self._build_summary_text(
            runtime_seconds=runtime_seconds,
            hotspot_function=hotspot_function,
            inferred_signals=inferred_signals,
        )

        return ProfilingSummary(
            runtime_seconds=runtime_seconds,
            hotspot_function=hotspot_function,
            hotspot_description=hotspot_description,
            raw_metrics=normalized,
            summary_text=summary_text,
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Low-level extraction helpers
    # ------------------------------------------------------------------

    def _extract_runtime_seconds(self, raw_text: str) -> Optional[float]:
        """
        Extract runtime in seconds from raw text using common patterns.
        """
        for pattern in self._RUNTIME_PATTERNS:
            match = pattern.search(raw_text)
            if match:
                try:
                    return float(match.group(1))
                except (TypeError, ValueError):
                    return None
        return None

    def _extract_hotspot_function(self, raw_text: str) -> Optional[str]:
        """
        Extract a hotspot function name from raw text when explicitly present.
        """
        for pattern in self._HOTSPOT_PATTERNS:
            match = pattern.search(raw_text)
            if match:
                return match.group(1)
        return None

    def _infer_keyword_signals(self, raw_text: str) -> Dict[str, Any]:
        """
        Infer high-level signal hints from keyword presence in raw text.

        These are intentionally simple binary/count-style indicators that can
        later help telemetry extraction.
        """
        lowered = raw_text.lower()
        signals: Dict[str, Any] = {}

        for group_name, keywords in self._KEYWORD_GROUPS.items():
            count = sum(1 for keyword in keywords if keyword.lower() in lowered)
            signals[f"{group_name}_signal_count"] = count
            signals[f"{group_name}_signal_present"] = count > 0

        return signals

    def _coerce_runtime_seconds(self, metrics: Dict[str, Any]) -> Optional[float]:
        """
        Attempt to normalize runtime from a precomputed metrics dictionary.
        """
        candidate_keys = [
            "runtime_seconds",
            "runtime",
            "elapsed_time_seconds",
            "elapsed_seconds",
            "time_seconds",
        ]

        for key in candidate_keys:
            if key in metrics and metrics[key] is not None:
                try:
                    return float(metrics[key])
                except (TypeError, ValueError):
                    return None

        return None

    def _coerce_hotspot_function(self, metrics: Dict[str, Any]) -> Optional[str]:
        """
        Attempt to normalize hotspot function name from a metrics dictionary.
        """
        candidate_keys = [
            "hotspot_function",
            "hotspot",
            "top_function",
            "hot_function",
        ]

        for key in candidate_keys:
            value = metrics.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        return None

    def _infer_signals_from_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Infer high-level signal hints from a metrics dictionary.

        This keeps the output shape somewhat similar between parse_from_text()
        and parse_from_metrics().
        """
        signals: Dict[str, Any] = {}

        # Generic communication-oriented hints.
        communication_keys = [
            "mpi_wait_ratio",
            "barrier_ratio",
            "communication_ratio",
            "comm_ratio",
        ]
        parallelism_keys = [
            "load_imbalance_score",
            "imbalance_score",
            "thread_idle_ratio",
        ]
        memory_keys = [
            "memory_bandwidth_utilization",
            "cache_miss_rate",
            "memory_bound_score",
        ]

        signals["communication_signal_present"] = any(
            self._is_positive_numeric(metrics.get(key)) for key in communication_keys
        )
        signals["parallelism_signal_present"] = any(
            self._is_positive_numeric(metrics.get(key)) for key in parallelism_keys
        )
        signals["memory_signal_present"] = any(
            self._is_positive_numeric(metrics.get(key)) for key in memory_keys
        )

        signals["communication_signal_count"] = sum(
            1 for key in communication_keys if self._is_positive_numeric(metrics.get(key))
        )
        signals["parallelism_signal_count"] = sum(
            1 for key in parallelism_keys if self._is_positive_numeric(metrics.get(key))
        )
        signals["memory_signal_count"] = sum(
            1 for key in memory_keys if self._is_positive_numeric(metrics.get(key))
        )

        return signals

    def _is_positive_numeric(self, value: Any) -> bool:
        """
        Return True if the value can be interpreted as a positive float.
        """
        try:
            return float(value) > 0.0
        except (TypeError, ValueError):
            return False

    # ------------------------------------------------------------------
    # Summary builders
    # ------------------------------------------------------------------

    def _build_hotspot_description(
        self,
        hotspot_function: Optional[str],
        inferred_signals: Dict[str, Any],
    ) -> str:
        """
        Build a compact hotspot-oriented description.
        """
        parts: List[str] = []

        if hotspot_function:
            parts.append(f"Primary hotspot function appears to be {hotspot_function}.")

        if inferred_signals.get("communication_signal_present"):
            parts.append("Profiling signals suggest communication-related overhead.")
        if inferred_signals.get("parallelism_signal_present"):
            parts.append("Profiling signals suggest parallelism or load-balance issues.")
        if inferred_signals.get("memory_signal_present"):
            parts.append("Profiling signals suggest memory-system pressure.")

        return " ".join(parts).strip()

    def _build_summary_text(
        self,
        runtime_seconds: Optional[float],
        hotspot_function: Optional[str],
        inferred_signals: Dict[str, Any],
    ) -> str:
        """
        Build a concise human-readable profiling summary for prompts and logs.
        """
        parts: List[str] = []

        if runtime_seconds is not None:
            parts.append(f"Observed runtime is approximately {runtime_seconds:.6f} seconds.")
        else:
            parts.append("Runtime information was not explicitly available.")

        if hotspot_function:
            parts.append(f"Hotspot function: {hotspot_function}.")

        signal_phrases: List[str] = []
        if inferred_signals.get("communication_signal_present"):
            signal_phrases.append("communication overhead indicators")
        if inferred_signals.get("parallelism_signal_present"):
            signal_phrases.append("parallel imbalance indicators")
        if inferred_signals.get("memory_signal_present"):
            signal_phrases.append("memory pressure indicators")

        if signal_phrases:
            parts.append("Detected " + ", ".join(signal_phrases) + ".")

        return " ".join(parts)