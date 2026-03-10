"""
test_telemetry_extractor.py

Unit tests for implementation.analysis.telemetry_extractor.TelemetryExtractor.

These tests verify that the telemetry extractor can:
- convert static code-analysis features into numeric telemetry
- convert profiling-derived signals into numeric telemetry
- compute higher-level combined heuristic pressures
- sanitize invalid numeric values
- produce a concise telemetry summary
- expose router-friendly metrics required by downstream advisor logic
"""

from __future__ import annotations

from implementation.analysis import TelemetryExtractor
from implementation.analysis.analysis_bundle import (
    CodeAnalysisSummary,
    CodeRegion,
    ProfilingSummary,
    TelemetrySummary,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_code_analysis(
    *,
    has_mpi: bool = False,
    has_openmp: bool = False,
    appears_memory_intensive: bool = False,
    function_count: int = 1,
    loop_count: int = 1,
    mpi_call_count: int = 0,
    omp_pragma_count: int = 0,
    line_count: int = 50,
) -> CodeAnalysisSummary:
    """Construct a reusable CodeAnalysisSummary for telemetry tests."""
    return CodeAnalysisSummary(
        source_path="benchmarks/example/main.c",
        language="c",
        file_name="main.c",
        file_size_bytes=256,
        line_count=line_count,
        function_count=function_count,
        loop_count=loop_count,
        mpi_call_count=mpi_call_count,
        omp_pragma_count=omp_pragma_count,
        summary_text="Synthetic static analysis summary.",
        regions=[
            CodeRegion(
                label="main",
                region_type="function",
                start_line=1,
                end_line=max(1, line_count),
                snippet="int main() { return 0; }",
                tags=["function"],
            )
        ],
        has_mpi=has_mpi,
        has_openmp=has_openmp,
        appears_memory_intensive=appears_memory_intensive,
    )


def build_profiling(
    *,
    runtime_seconds: float | None = None,
    hotspot_function: str | None = None,
    raw_metrics: dict | None = None,
) -> ProfilingSummary:
    """Construct a reusable ProfilingSummary for telemetry tests."""
    return ProfilingSummary(
        runtime_seconds=runtime_seconds,
        hotspot_function=hotspot_function,
        hotspot_description="Synthetic hotspot description.",
        raw_metrics=raw_metrics or {},
        summary_text="Synthetic profiling summary.",
        notes=["Synthetic profiling object for tests."],
    )


# ---------------------------------------------------------------------------
# Basic return type / interface tests
# ---------------------------------------------------------------------------

def test_extract_returns_telemetry_summary() -> None:
    """Telemetry extraction should return the expected TelemetrySummary type."""
    extractor = TelemetryExtractor()
    code_analysis = build_code_analysis()
    profiling = build_profiling()

    telemetry = extractor.extract(code_analysis, profiling)

    assert isinstance(telemetry, TelemetrySummary)


def test_extract_returns_metrics_summary_and_notes() -> None:
    """TelemetrySummary should contain metrics, summary_text, and provenance notes."""
    extractor = TelemetryExtractor()
    code_analysis = build_code_analysis()
    profiling = build_profiling()

    telemetry = extractor.extract(code_analysis, profiling)

    assert isinstance(telemetry.metrics, dict)
    assert isinstance(telemetry.summary_text, str)
    assert telemetry.summary_text.strip() != ""
    assert isinstance(telemetry.notes, list)
    assert len(telemetry.notes) > 0


# ---------------------------------------------------------------------------
# Static signal extraction tests
# ---------------------------------------------------------------------------

def test_extract_static_flags_are_converted_to_numeric_metrics() -> None:
    """
    Binary static code-analysis features should be converted into 0.0/1.0
    telemetry metrics.
    """
    extractor = TelemetryExtractor()
    code_analysis = build_code_analysis(
        has_mpi=True,
        has_openmp=True,
        appears_memory_intensive=True,
    )
    profiling = build_profiling()

    telemetry = extractor.extract(code_analysis, profiling)

    assert telemetry.metrics["has_mpi"] == 1.0
    assert telemetry.metrics["has_openmp"] == 1.0
    assert telemetry.metrics["appears_memory_intensive"] == 1.0


def test_extract_static_counts_are_exposed_as_float_metrics() -> None:
    """Structural counts should be preserved as float-valued telemetry fields."""
    extractor = TelemetryExtractor()
    code_analysis = build_code_analysis(
        function_count=3,
        loop_count=5,
        mpi_call_count=2,
        omp_pragma_count=1,
        line_count=100,
    )
    profiling = build_profiling()

    telemetry = extractor.extract(code_analysis, profiling)

    assert telemetry.metrics["function_count"] == 3.0
    assert telemetry.metrics["loop_count"] == 5.0
    assert telemetry.metrics["mpi_call_count"] == 2.0
    assert telemetry.metrics["omp_pragma_count"] == 1.0
    assert telemetry.metrics["line_count"] == 100.0


def test_extract_static_density_metrics_are_computed() -> None:
    """Density metrics should be derived from counts and line count."""
    extractor = TelemetryExtractor()
    code_analysis = build_code_analysis(
        loop_count=4,
        mpi_call_count=2,
        omp_pragma_count=1,
        line_count=20,
    )
    profiling = build_profiling()

    telemetry = extractor.extract(code_analysis, profiling)

    assert telemetry.metrics["loop_density"] == 4.0 / 20.0
    assert telemetry.metrics["mpi_call_density"] == 2.0 / 20.0
    assert telemetry.metrics["omp_pragma_density"] == 1.0 / 20.0


# ---------------------------------------------------------------------------
# Profiling signal extraction tests
# ---------------------------------------------------------------------------

def test_extract_runtime_and_hotspot_fields_from_profiling() -> None:
    """Runtime and hotspot availability should be reflected in telemetry metrics."""
    extractor = TelemetryExtractor()
    code_analysis = build_code_analysis()
    profiling = build_profiling(
        runtime_seconds=0.75,
        hotspot_function="matmul_kernel",
        raw_metrics={},
    )

    telemetry = extractor.extract(code_analysis, profiling)

    assert telemetry.metrics["runtime_seconds"] == 0.75
    assert telemetry.metrics["runtime_available"] == 1.0
    assert telemetry.metrics["hotspot_identified"] == 1.0


def test_extract_generic_signal_counts_and_presence_from_profiling_raw_metrics() -> None:
    """
    Generic signal counts/presence flags produced by the profiler parser should
    carry over into telemetry metrics.
    """
    extractor = TelemetryExtractor()
    code_analysis = build_code_analysis()
    profiling = build_profiling(
        raw_metrics={
            "communication_signal_count": 2,
            "parallelism_signal_count": 1,
            "memory_signal_count": 3,
            "communication_signal_present": True,
            "parallelism_signal_present": True,
            "memory_signal_present": True,
        }
    )

    telemetry = extractor.extract(code_analysis, profiling)

    assert telemetry.metrics["communication_signal_count"] == 2.0
    assert telemetry.metrics["parallelism_signal_count"] == 1.0
    assert telemetry.metrics["memory_signal_count"] == 3.0

    assert telemetry.metrics["communication_signal_present"] == 1.0
    assert telemetry.metrics["parallelism_signal_present"] == 1.0
    assert telemetry.metrics["memory_signal_present"] == 1.0


def test_extract_common_hpc_ratios_from_profiling_raw_metrics() -> None:
    """
    HPC-oriented ratios/scores should be normalized from raw_metrics when present.
    """
    extractor = TelemetryExtractor()
    code_analysis = build_code_analysis()
    profiling = build_profiling(
        raw_metrics={
            "mpi_wait_ratio": 0.4,
            "barrier_ratio": 0.2,
            "communication_ratio": 0.35,
            "load_imbalance_score": 0.6,
            "thread_idle_ratio": 0.1,
            "memory_bound_score": 0.7,
            "cache_miss_rate": 0.25,
            "memory_bandwidth_utilization": 0.8,
        }
    )

    telemetry = extractor.extract(code_analysis, profiling)

    assert telemetry.metrics["mpi_wait_ratio"] == 0.4
    assert telemetry.metrics["barrier_ratio"] == 0.2
    assert telemetry.metrics["communication_ratio"] == 0.35
    assert telemetry.metrics["load_imbalance_score"] == 0.6
    assert telemetry.metrics["thread_idle_ratio"] == 0.1
    assert telemetry.metrics["memory_bound_score"] == 0.7
    assert telemetry.metrics["cache_miss_rate"] == 0.25
    assert telemetry.metrics["memory_bandwidth_utilization"] == 0.8


# ---------------------------------------------------------------------------
# Combined heuristic pressure tests
# ---------------------------------------------------------------------------

def test_extract_communication_pressure_is_high_for_mpi_communication_case() -> None:
    """
    Communication pressure should become substantial when MPI is present and
    profiling shows communication-related overhead.
    """
    extractor = TelemetryExtractor()
    code_analysis = build_code_analysis(has_mpi=True, mpi_call_count=5)
    profiling = build_profiling(
        raw_metrics={
            "communication_signal_present": True,
            "communication_signal_count": 2,
            "mpi_wait_ratio": 0.6,
            "barrier_ratio": 0.5,
        }
    )

    telemetry = extractor.extract(code_analysis, profiling)

    assert telemetry.metrics["communication_pressure"] > 0.5
    assert telemetry.metrics["communication_dominant"] == 1.0
    assert telemetry.metrics["router_comm_score"] == telemetry.metrics["communication_pressure"]


def test_extract_parallelism_pressure_is_high_for_openmp_imbalance_case() -> None:
    """
    Parallelism pressure should become substantial when OpenMP is present and
    profiling shows imbalance/idling.
    """
    extractor = TelemetryExtractor()
    code_analysis = build_code_analysis(has_openmp=True, omp_pragma_count=2)
    profiling = build_profiling(
        raw_metrics={
            "parallelism_signal_present": True,
            "parallelism_signal_count": 2,
            "load_imbalance_score": 0.7,
            "thread_idle_ratio": 0.4,
        }
    )

    telemetry = extractor.extract(code_analysis, profiling)

    assert telemetry.metrics["parallelism_pressure"] > 0.4
    assert telemetry.metrics["parallelism_issue_present"] == 1.0
    assert telemetry.metrics["router_parallel_score"] == telemetry.metrics["parallelism_pressure"]


def test_extract_memory_pressure_is_high_for_memory_bound_case() -> None:
    """
    Memory pressure should become substantial when static memory hints and
    profiling memory signals are both present.
    """
    extractor = TelemetryExtractor()
    code_analysis = build_code_analysis(appears_memory_intensive=True, loop_count=4)
    profiling = build_profiling(
        raw_metrics={
            "memory_signal_present": True,
            "memory_signal_count": 2,
            "memory_bound_score": 0.8,
            "cache_miss_rate": 0.5,
        }
    )

    telemetry = extractor.extract(code_analysis, profiling)

    assert telemetry.metrics["memory_pressure"] > 0.4
    assert telemetry.metrics["memory_issue_present"] == 1.0
    assert telemetry.metrics["router_memory_score"] == telemetry.metrics["memory_pressure"]


def test_extract_compute_pressure_is_present_for_loop_heavy_non_comm_non_memory_case() -> None:
    """
    Compute pressure should be nonzero for loop-heavy code when communication
    and memory pressures are not dominant.
    """
    extractor = TelemetryExtractor()
    code_analysis = build_code_analysis(
        loop_count=10,
        line_count=40,
        has_mpi=False,
        has_openmp=False,
        appears_memory_intensive=False,
    )
    profiling = build_profiling(raw_metrics={})

    telemetry = extractor.extract(code_analysis, profiling)

    assert telemetry.metrics["compute_pressure"] > 0.0


# ---------------------------------------------------------------------------
# Sparse / fallback behavior tests
# ---------------------------------------------------------------------------

def test_extract_marks_profiling_sparse_when_runtime_missing() -> None:
    """If runtime is unavailable, profiling_sparse should be set."""
    extractor = TelemetryExtractor()
    code_analysis = build_code_analysis()
    profiling = build_profiling(runtime_seconds=None)

    telemetry = extractor.extract(code_analysis, profiling)

    assert telemetry.metrics["profiling_sparse"] == 1.0
    assert telemetry.metrics["runtime_available"] == 0.0


def test_extract_marks_small_kernel_hint_for_short_sources() -> None:
    """Small source files should trigger the small_kernel_hint feature."""
    extractor = TelemetryExtractor()
    code_analysis = build_code_analysis(line_count=60)
    profiling = build_profiling()

    telemetry = extractor.extract(code_analysis, profiling)

    assert telemetry.metrics["small_kernel_hint"] == 1.0


def test_extract_does_not_mark_small_kernel_hint_for_larger_sources() -> None:
    """Larger source files should not trigger the small_kernel_hint feature."""
    extractor = TelemetryExtractor()
    code_analysis = build_code_analysis(line_count=200)
    profiling = build_profiling()

    telemetry = extractor.extract(code_analysis, profiling)

    assert telemetry.metrics["small_kernel_hint"] == 0.0


# ---------------------------------------------------------------------------
# Sanitization / numeric stability tests
# ---------------------------------------------------------------------------

def test_extract_sanitizes_negative_and_invalid_numeric_inputs() -> None:
    """
    Invalid or negative profiling values should be coerced into safe nonnegative
    numeric telemetry outputs.
    """
    extractor = TelemetryExtractor()
    code_analysis = build_code_analysis()
    profiling = build_profiling(
        raw_metrics={
            "mpi_wait_ratio": -0.3,
            "barrier_ratio": "not-a-number",
            "load_imbalance_score": -1.0,
            "memory_bound_score": None,
            "cache_miss_rate": "0.25",
        }
    )

    telemetry = extractor.extract(code_analysis, profiling)

    assert telemetry.metrics["mpi_wait_ratio"] == 0.0
    assert telemetry.metrics["barrier_ratio"] == 0.0
    assert telemetry.metrics["load_imbalance_score"] == 0.0
    assert telemetry.metrics["memory_bound_score"] == 0.0
    assert telemetry.metrics["cache_miss_rate"] == 0.25


def test_extract_all_metrics_are_numeric_and_nonnegative() -> None:
    """Final telemetry metrics should be numeric floats and nonnegative."""
    extractor = TelemetryExtractor()
    code_analysis = build_code_analysis(
        has_mpi=True,
        has_openmp=True,
        appears_memory_intensive=True,
        function_count=4,
        loop_count=6,
        mpi_call_count=3,
        omp_pragma_count=2,
        line_count=80,
    )
    profiling = build_profiling(
        runtime_seconds=1.2,
        hotspot_function="main",
        raw_metrics={
            "communication_signal_present": True,
            "parallelism_signal_present": True,
            "memory_signal_present": True,
            "mpi_wait_ratio": 0.4,
            "load_imbalance_score": 0.3,
            "memory_bound_score": 0.5,
        },
    )

    telemetry = extractor.extract(code_analysis, profiling)

    for key, value in telemetry.metrics.items():
        assert isinstance(value, float), f"Metric {key} is not a float"
        assert value >= 0.0, f"Metric {key} is negative"


# ---------------------------------------------------------------------------
# Summary text tests
# ---------------------------------------------------------------------------

def test_summary_text_mentions_dominant_communication_signal() -> None:
    """Summary text should mention communication when it is the dominant signal."""
    extractor = TelemetryExtractor()
    code_analysis = build_code_analysis(has_mpi=True)
    profiling = build_profiling(
        runtime_seconds=0.9,
        raw_metrics={
            "communication_signal_present": True,
            "mpi_wait_ratio": 0.8,
            "barrier_ratio": 0.6,
        },
    )

    telemetry = extractor.extract(code_analysis, profiling)

    assert "communication" in telemetry.summary_text.lower()


def test_summary_text_mentions_runtime_when_available() -> None:
    """Summary text should include runtime information when available."""
    extractor = TelemetryExtractor()
    code_analysis = build_code_analysis()
    profiling = build_profiling(runtime_seconds=0.321)

    telemetry = extractor.extract(code_analysis, profiling)

    assert "observed runtime is 0.321000 seconds" in telemetry.summary_text.lower()


def test_summary_text_mentions_openmp_and_memory_when_present() -> None:
    """
    Summary text should include supporting evidence such as OpenMP presence and
    memory intensity hints when relevant.
    """
    extractor = TelemetryExtractor()
    code_analysis = build_code_analysis(
        has_openmp=True,
        appears_memory_intensive=True,
    )
    profiling = build_profiling(
        raw_metrics={
            "memory_bound_score": 0.7,
            "memory_signal_present": True,
        }
    )

    telemetry = extractor.extract(code_analysis, profiling)

    lowered = telemetry.summary_text.lower()
    assert "openmp" in lowered
    assert "memory" in lowered


# ---------------------------------------------------------------------------
# Router-facing interface tests
# ---------------------------------------------------------------------------

def test_extract_exposes_router_friendly_score_fields() -> None:
    """
    The final telemetry dictionary should expose explicit router score fields for
    communication, parallelism, and memory domains.
    """
    extractor = TelemetryExtractor()
    code_analysis = build_code_analysis(
        has_mpi=True,
        has_openmp=True,
        appears_memory_intensive=True,
    )
    profiling = build_profiling(
        raw_metrics={
            "communication_signal_present": True,
            "parallelism_signal_present": True,
            "memory_signal_present": True,
            "mpi_wait_ratio": 0.2,
            "load_imbalance_score": 0.3,
            "memory_bound_score": 0.4,
        }
    )

    telemetry = extractor.extract(code_analysis, profiling)

    assert "router_comm_score" in telemetry.metrics
    assert "router_parallel_score" in telemetry.metrics
    assert "router_memory_score" in telemetry.metrics


# ---------------------------------------------------------------------------
# Low-level helper tests
# ---------------------------------------------------------------------------

def test_clamp01_bounds_values_correctly() -> None:
    """_clamp01() should bound values to the [0, 1] range."""
    extractor = TelemetryExtractor()

    assert extractor._clamp01(-0.5) == 0.0  # noqa: SLF001
    assert extractor._clamp01(0.4) == 0.4   # noqa: SLF001
    assert extractor._clamp01(1.5) == 1.0   # noqa: SLF001


def test_safe_divide_handles_zero_denominator() -> None:
    """_safe_divide() should return 0.0 when denominator is zero."""
    extractor = TelemetryExtractor()

    assert extractor._safe_divide(10, 0) == 0.0  # noqa: SLF001
    assert extractor._safe_divide(10, 2) == 5.0  # noqa: SLF001


def test_coerce_nonnegative_ratio_converts_invalid_values_to_zero() -> None:
    """_coerce_nonnegative_ratio() should sanitize invalid ratio inputs."""
    extractor = TelemetryExtractor()

    assert extractor._coerce_nonnegative_ratio("0.3") == 0.3   # noqa: SLF001
    assert extractor._coerce_nonnegative_ratio(-1.0) == 0.0    # noqa: SLF001
    assert extractor._coerce_nonnegative_ratio(None) == 0.0    # noqa: SLF001
    assert extractor._coerce_nonnegative_ratio("bad") == 0.0   # noqa: SLF001