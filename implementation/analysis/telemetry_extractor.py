"""
telemetry_extractor.py

Builds normalized telemetry features from static code analysis and profiling
summaries.

This module is the bridge between:
- static source-code evidence from code_analyzer.py
- runtime/profiling evidence from profiler_parser.py
- the structured telemetry dictionary expected by the MoE router

Current goals:
- produce a stable router-friendly telemetry dictionary
- keep the logic lightweight and deterministic
- expose both numeric telemetry features and a short text summary
- work well with the current temp_moe benchmark suite

Design note:
This is intentionally heuristic. At this stage, the purpose is not to produce
ground-truth performance modeling, but to standardize evidence into signals that
can be used consistently by the advisor.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .analysis_bundle import CodeAnalysisSummary, ProfilingSummary, TelemetrySummary


class TelemetryExtractor:
    """
    Converts code-analysis and profiling outputs into normalized telemetry.

    The main output is a TelemetrySummary object whose `metrics` field is
    designed to be passed directly into the existing MoE router.

    Typical usage:
        extractor = TelemetryExtractor()
        telemetry = extractor.extract(code_analysis, profiling)
    """

    def extract(
        self,
        code_analysis: CodeAnalysisSummary,
        profiling: ProfilingSummary,
    ) -> TelemetrySummary:
        """
        Produce normalized telemetry from static and profiling evidence.

        Parameters
        ----------
        code_analysis:
            Structured static analysis summary for the source file.
        profiling:
            Structured profiling summary parsed from benchmark/profiler output.

        Returns
        -------
        TelemetrySummary
            Router-friendly telemetry metrics plus a concise summary.
        """
        notes: List[str] = []
        metrics: Dict[str, float] = {}

        # ------------------------------------------------------------------
        # Static code-derived signals
        # ------------------------------------------------------------------
        metrics.update(self._static_signals(code_analysis))

        # ------------------------------------------------------------------
        # Profiling-derived signals
        # ------------------------------------------------------------------
        metrics.update(self._profiling_signals(profiling))

        # ------------------------------------------------------------------
        # Combined / higher-level heuristic signals
        # ------------------------------------------------------------------
        metrics.update(self._combined_signals(code_analysis, profiling, metrics))

        # Ensure the output remains numeric and stable.
        metrics = self._sanitize_metrics(metrics)

        summary_text = self._build_summary_text(metrics)

        # Add light provenance notes for debugging.
        notes.append(
            "Telemetry metrics were derived heuristically from static code analysis "
            "and parsed profiling summaries."
        )

        return TelemetrySummary(
            metrics=metrics,
            summary_text=summary_text,
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Static signal extraction
    # ------------------------------------------------------------------

    def _static_signals(self, code_analysis: CodeAnalysisSummary) -> Dict[str, float]:
        """
        Derive telemetry-style numeric features from static source analysis.
        """
        metrics: Dict[str, float] = {}

        # Binary presence features.
        metrics["has_mpi"] = 1.0 if code_analysis.has_mpi else 0.0
        metrics["has_openmp"] = 1.0 if code_analysis.has_openmp else 0.0
        metrics["appears_memory_intensive"] = (
            1.0 if code_analysis.appears_memory_intensive else 0.0
        )

        # Raw structural counts as floats for consistency.
        metrics["function_count"] = float(code_analysis.function_count)
        metrics["loop_count"] = float(code_analysis.loop_count)
        metrics["mpi_call_count"] = float(code_analysis.mpi_call_count)
        metrics["omp_pragma_count"] = float(code_analysis.omp_pragma_count)
        metrics["line_count"] = float(code_analysis.line_count)

        # Ratios that can help routing without tying us to exact benchmark sizes.
        metrics["mpi_call_density"] = self._safe_divide(
            code_analysis.mpi_call_count,
            max(code_analysis.line_count, 1),
        )
        metrics["loop_density"] = self._safe_divide(
            code_analysis.loop_count,
            max(code_analysis.line_count, 1),
        )
        metrics["omp_pragma_density"] = self._safe_divide(
            code_analysis.omp_pragma_count,
            max(code_analysis.line_count, 1),
        )

        return metrics

    # ------------------------------------------------------------------
    # Profiling signal extraction
    # ------------------------------------------------------------------

    def _profiling_signals(self, profiling: ProfilingSummary) -> Dict[str, float]:
        """
        Derive telemetry-style numeric features from parsed profiling outputs.
        """
        metrics: Dict[str, float] = {}

        raw = profiling.raw_metrics or {}

        # Runtime signal (keep raw runtime if available).
        runtime_seconds = profiling.runtime_seconds
        metrics["runtime_seconds"] = float(runtime_seconds) if runtime_seconds is not None else 0.0
        metrics["runtime_available"] = 1.0 if runtime_seconds is not None else 0.0

        # Hotspot availability.
        metrics["hotspot_identified"] = 1.0 if profiling.hotspot_function else 0.0

        # Carry forward the generic signal counts generated by ProfilerParser.
        metrics["communication_signal_count"] = float(raw.get("communication_signal_count", 0.0))
        metrics["parallelism_signal_count"] = float(raw.get("parallelism_signal_count", 0.0))
        metrics["memory_signal_count"] = float(raw.get("memory_signal_count", 0.0))

        metrics["communication_signal_present"] = 1.0 if raw.get("communication_signal_present") else 0.0
        metrics["parallelism_signal_present"] = 1.0 if raw.get("parallelism_signal_present") else 0.0
        metrics["memory_signal_present"] = 1.0 if raw.get("memory_signal_present") else 0.0

        # Normalize common HPC-oriented fields if they already exist.
        metrics["mpi_wait_ratio"] = self._coerce_nonnegative_ratio(
            raw.get("mpi_wait_ratio", raw.get("wait_ratio", 0.0))
        )
        metrics["barrier_ratio"] = self._coerce_nonnegative_ratio(raw.get("barrier_ratio", 0.0))
        metrics["communication_ratio"] = self._coerce_nonnegative_ratio(
            raw.get("communication_ratio", raw.get("comm_ratio", 0.0))
        )
        metrics["load_imbalance_score"] = self._coerce_nonnegative_ratio(
            raw.get("load_imbalance_score", raw.get("imbalance_score", 0.0))
        )
        metrics["thread_idle_ratio"] = self._coerce_nonnegative_ratio(raw.get("thread_idle_ratio", 0.0))
        metrics["memory_bound_score"] = self._coerce_nonnegative_ratio(raw.get("memory_bound_score", 0.0))
        metrics["cache_miss_rate"] = self._coerce_nonnegative_ratio(raw.get("cache_miss_rate", 0.0))
        metrics["memory_bandwidth_utilization"] = self._coerce_nonnegative_ratio(
            raw.get("memory_bandwidth_utilization", 0.0)
        )

        return metrics

    # ------------------------------------------------------------------
    # Combined heuristic signal extraction
    # ------------------------------------------------------------------

    def _combined_signals(
        self,
        code_analysis: CodeAnalysisSummary,
        profiling: ProfilingSummary,
        current_metrics: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Derive higher-level telemetry features by combining static and profiling
        evidence.

        These are especially useful for routing because they reflect the kind of
        optimization domain suggested by the available evidence.
        """
        metrics: Dict[str, float] = {}

        # Communication pressure: strong if MPI is present and profiling also
        # hints at communication overhead.
        metrics["communication_pressure"] = self._clamp01(
            0.45 * current_metrics.get("has_mpi", 0.0)
            + 0.25 * current_metrics.get("communication_signal_present", 0.0)
            + 0.20 * current_metrics.get("mpi_wait_ratio", 0.0)
            + 0.10 * current_metrics.get("barrier_ratio", 0.0)
        )

        # Parallelism pressure: strong if OpenMP is present and imbalance/idling
        # is visible in profiling.
        metrics["parallelism_pressure"] = self._clamp01(
            0.35 * current_metrics.get("has_openmp", 0.0)
            + 0.20 * current_metrics.get("parallelism_signal_present", 0.0)
            + 0.30 * current_metrics.get("load_imbalance_score", 0.0)
            + 0.15 * current_metrics.get("thread_idle_ratio", 0.0)
        )

        # Memory pressure: combines static memory hints with runtime indicators.
        metrics["memory_pressure"] = self._clamp01(
            0.35 * current_metrics.get("appears_memory_intensive", 0.0)
            + 0.20 * current_metrics.get("memory_signal_present", 0.0)
            + 0.25 * current_metrics.get("memory_bound_score", 0.0)
            + 0.20 * current_metrics.get("cache_miss_rate", 0.0)
        )

        # A simple "compute intensity hint" based on loops without strong
        # communication/memory evidence.
        loop_density = current_metrics.get("loop_density", 0.0)
        communication_pressure = metrics["communication_pressure"]
        memory_pressure = metrics["memory_pressure"]

        metrics["compute_pressure"] = self._clamp01(
            min(1.0, loop_density * 50.0) * (1.0 - 0.5 * communication_pressure) * (1.0 - 0.4 * memory_pressure)
        )

        # Benchmark-style domain hints.
        metrics["communication_dominant"] = 1.0 if communication_pressure >= 0.50 else 0.0
        metrics["parallelism_issue_present"] = 1.0 if metrics["parallelism_pressure"] >= 0.40 else 0.0
        metrics["memory_issue_present"] = 1.0 if memory_pressure >= 0.40 else 0.0

        # Optional coarse routing hint priorities.
        metrics["router_comm_score"] = communication_pressure
        metrics["router_parallel_score"] = metrics["parallelism_pressure"]
        metrics["router_memory_score"] = memory_pressure

        # If profiling includes no explicit signals, static evidence can still
        # lightly influence the router.
        if profiling.runtime_seconds is None:
            metrics["profiling_sparse"] = 1.0
        else:
            metrics["profiling_sparse"] = 0.0

        # Add a lightweight signal for small/simple kernels.
        metrics["small_kernel_hint"] = 1.0 if code_analysis.line_count < 120 else 0.0

        return metrics

    # ------------------------------------------------------------------
    # Summary builder
    # ------------------------------------------------------------------

    def _build_summary_text(self, metrics: Dict[str, float]) -> str:
        """
        Build a concise human-readable telemetry summary for advisor prompts/logs.
        """
        parts: List[str] = []

        communication_pressure = metrics.get("communication_pressure", 0.0)
        parallelism_pressure = metrics.get("parallelism_pressure", 0.0)
        memory_pressure = metrics.get("memory_pressure", 0.0)
        compute_pressure = metrics.get("compute_pressure", 0.0)

        # Mention dominant behavior first.
        dominant_label = self._dominant_pressure_label(
            communication_pressure=communication_pressure,
            parallelism_pressure=parallelism_pressure,
            memory_pressure=memory_pressure,
            compute_pressure=compute_pressure,
        )
        if dominant_label:
            parts.append(f"Dominant telemetry signal suggests {dominant_label}.")

        # Include important supporting signals.
        if metrics.get("has_mpi", 0.0) > 0.0:
            parts.append("Source contains MPI usage.")
        if metrics.get("has_openmp", 0.0) > 0.0:
            parts.append("Source contains OpenMP pragmas.")
        if metrics.get("appears_memory_intensive", 0.0) > 0.0:
            parts.append("Static analysis suggests memory-intensive access patterns.")

        if metrics.get("mpi_wait_ratio", 0.0) > 0.0:
            parts.append(f"MPI wait ratio is {metrics['mpi_wait_ratio']:.3f}.")
        if metrics.get("load_imbalance_score", 0.0) > 0.0:
            parts.append(f"Load imbalance score is {metrics['load_imbalance_score']:.3f}.")
        if metrics.get("memory_bound_score", 0.0) > 0.0:
            parts.append(f"Memory-bound score is {metrics['memory_bound_score']:.3f}.")

        if metrics.get("runtime_available", 0.0) > 0.0:
            parts.append(f"Observed runtime is {metrics.get('runtime_seconds', 0.0):.6f} seconds.")

        return " ".join(parts).strip()

    def _dominant_pressure_label(
        self,
        communication_pressure: float,
        parallelism_pressure: float,
        memory_pressure: float,
        compute_pressure: float,
    ) -> Optional[str]:
        """
        Return a concise label for the strongest derived telemetry pressure.
        """
        candidates = {
            "communication overhead": communication_pressure,
            "parallelism or load-balance issues": parallelism_pressure,
            "memory-system pressure": memory_pressure,
            "compute-heavy loop structure": compute_pressure,
        }

        label, value = max(candidates.items(), key=lambda item: item[1])
        if value <= 0.0:
            return None
        return label

    # ------------------------------------------------------------------
    # Numeric utilities
    # ------------------------------------------------------------------

    def _sanitize_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Ensure all metrics are numeric floats and replace invalid values with 0.0.
        """
        sanitized: Dict[str, float] = {}

        for key, value in metrics.items():
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                numeric_value = 0.0

            # Avoid negative telemetry for this stage unless specifically needed.
            if numeric_value < 0.0:
                numeric_value = 0.0

            sanitized[key] = numeric_value

        return sanitized

    def _safe_divide(self, numerator: float, denominator: float) -> float:
        """
        Safely divide numerator by denominator.
        """
        if denominator == 0:
            return 0.0
        return float(numerator) / float(denominator)

    def _coerce_nonnegative_ratio(self, value: object) -> float:
        """
        Coerce a value into a nonnegative float.

        This is useful for metrics that are expected to be ratios or scores.
        """
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 0.0

        if numeric < 0.0:
            return 0.0
        return numeric

    def _clamp01(self, value: float) -> float:
        """
        Clamp a numeric value to the [0, 1] range.
        """
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value