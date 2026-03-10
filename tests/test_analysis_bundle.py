"""
test_analysis_bundle.py

Unit tests for implementation.analysis.analysis_bundle.

These tests focus on the structured data containers used by the analysis stage,
especially the top-level AnalysisBundle that will be passed into the MoE advisor.

Main goals:
- verify dataclass construction works as expected
- verify to_dict() outputs are stable and structured
- verify advisor_inputs() matches the current advisor interface
- verify fallback behavior when selected_snippets is empty
- verify from_source_path() reads source text from disk correctly
"""

from __future__ import annotations

from pathlib import Path

from implementation.analysis.analysis_bundle import (
    AnalysisBundle,
    CodeAnalysisSummary,
    CodeRegion,
    ProfilingSummary,
    TelemetrySummary,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS_DIR = REPO_ROOT / "benchmarks"


def build_sample_code_region() -> CodeRegion:
    """Create a representative CodeRegion for repeated test use."""
    return CodeRegion(
        label="main",
        region_type="function",
        start_line=1,
        end_line=10,
        snippet="int main() { return 0; }",
        tags=["function"],
    )


def build_sample_code_analysis(source_path: str = "benchmarks/example/main.c") -> CodeAnalysisSummary:
    """Create a representative CodeAnalysisSummary for repeated test use."""
    return CodeAnalysisSummary(
        source_path=source_path,
        language="c",
        file_name="main.c",
        file_size_bytes=128,
        line_count=12,
        function_count=1,
        loop_count=2,
        mpi_call_count=0,
        omp_pragma_count=1,
        summary_text="Static analysis summary.",
        regions=[build_sample_code_region()],
        has_mpi=False,
        has_openmp=True,
        appears_memory_intensive=True,
    )


def build_sample_profiling() -> ProfilingSummary:
    """Create a representative ProfilingSummary for repeated test use."""
    return ProfilingSummary(
        runtime_seconds=0.123456,
        hotspot_function="main",
        hotspot_description="Primary hotspot function appears to be main.",
        raw_metrics={
            "runtime_seconds": 0.123456,
            "hotspot_function": "main",
            "memory_signal_present": True,
        },
        summary_text="Observed runtime is approximately 0.123456 seconds.",
        notes=["Created from synthetic test metrics."],
    )


def build_sample_telemetry() -> TelemetrySummary:
    """Create a representative TelemetrySummary for repeated test use."""
    return TelemetrySummary(
        metrics={
            "has_openmp": 1.0,
            "loop_count": 2.0,
            "memory_pressure": 0.65,
            "router_memory_score": 0.65,
        },
        summary_text="Dominant telemetry signal suggests memory-system pressure.",
        notes=["Synthetic telemetry for testing."],
    )


def build_sample_bundle(selected_snippets: list[str] | None = None) -> AnalysisBundle:
    """Create a representative AnalysisBundle for repeated test use."""
    return AnalysisBundle(
        benchmark_name="sample_benchmark",
        source_path="benchmarks/example/main.c",
        source_text="int main() {\n    return 0;\n}\n",
        code_analysis=build_sample_code_analysis(),
        profiling=build_sample_profiling(),
        telemetry=build_sample_telemetry(),
        metadata={"iteration": 1, "benchmark_type": "synthetic"},
        selected_snippets=selected_snippets or [],
    )


# ---------------------------------------------------------------------------
# Tests for CodeRegion
# ---------------------------------------------------------------------------

def test_code_region_to_dict_contains_expected_fields() -> None:
    """CodeRegion.to_dict() should preserve the main region fields."""
    region = build_sample_code_region()

    region_dict = region.to_dict()

    assert region_dict["label"] == "main"
    assert region_dict["region_type"] == "function"
    assert region_dict["start_line"] == 1
    assert region_dict["end_line"] == 10
    assert region_dict["snippet"] == "int main() { return 0; }"
    assert region_dict["tags"] == ["function"]


# ---------------------------------------------------------------------------
# Tests for CodeAnalysisSummary
# ---------------------------------------------------------------------------

def test_code_analysis_summary_to_dict_serializes_regions() -> None:
    """
    CodeAnalysisSummary.to_dict() should preserve scalar fields and serialize
    nested CodeRegion objects properly.
    """
    summary = build_sample_code_analysis()

    summary_dict = summary.to_dict()

    assert summary_dict["source_path"] == "benchmarks/example/main.c"
    assert summary_dict["language"] == "c"
    assert summary_dict["function_count"] == 1
    assert summary_dict["loop_count"] == 2
    assert summary_dict["has_openmp"] is True
    assert isinstance(summary_dict["regions"], list)
    assert len(summary_dict["regions"]) == 1
    assert summary_dict["regions"][0]["label"] == "main"


# ---------------------------------------------------------------------------
# Tests for ProfilingSummary / TelemetrySummary
# ---------------------------------------------------------------------------

def test_profiling_summary_to_dict_contains_expected_content() -> None:
    """ProfilingSummary.to_dict() should preserve all profiling fields."""
    profiling = build_sample_profiling()

    profiling_dict = profiling.to_dict()

    assert profiling_dict["runtime_seconds"] == 0.123456
    assert profiling_dict["hotspot_function"] == "main"
    assert "summary_text" in profiling_dict
    assert "raw_metrics" in profiling_dict
    assert profiling_dict["raw_metrics"]["hotspot_function"] == "main"


def test_telemetry_summary_to_dict_contains_expected_content() -> None:
    """TelemetrySummary.to_dict() should preserve telemetry metrics and notes."""
    telemetry = build_sample_telemetry()

    telemetry_dict = telemetry.to_dict()

    assert "metrics" in telemetry_dict
    assert telemetry_dict["metrics"]["memory_pressure"] == 0.65
    assert telemetry_dict["summary_text"] == "Dominant telemetry signal suggests memory-system pressure."
    assert telemetry_dict["notes"] == ["Synthetic telemetry for testing."]


# ---------------------------------------------------------------------------
# Tests for AnalysisBundle basic behavior
# ---------------------------------------------------------------------------

def test_analysis_bundle_to_dict_contains_nested_sections() -> None:
    """
    AnalysisBundle.to_dict() should include all major sections needed for
    logging, serialization, and debugging.
    """
    bundle = build_sample_bundle(selected_snippets=["for (int i = 0; i < n; i++) { ... }"])

    bundle_dict = bundle.to_dict()

    assert bundle_dict["benchmark_name"] == "sample_benchmark"
    assert bundle_dict["source_path"] == "benchmarks/example/main.c"
    assert "source_text" in bundle_dict
    assert "code_analysis" in bundle_dict
    assert "profiling" in bundle_dict
    assert "telemetry" in bundle_dict
    assert "metadata" in bundle_dict
    assert "selected_snippets" in bundle_dict

    # Verify nested content exists and remains structured.
    assert bundle_dict["code_analysis"]["file_name"] == "main.c"
    assert bundle_dict["profiling"]["hotspot_function"] == "main"
    assert bundle_dict["telemetry"]["metrics"]["router_memory_score"] == 0.65
    assert bundle_dict["selected_snippets"] == ["for (int i = 0; i < n; i++) { ... }"]


def test_analysis_bundle_advisor_inputs_uses_selected_snippets_when_available() -> None:
    """
    advisor_inputs() should use selected_snippets preferentially when they are
    available, because these are the intended prompt-ready code excerpts.
    """
    selected_snippets = [
        "for (int i = 0; i < n; i++) {",
        "    y[i] = a * x[i] + y[i];",
        "}",
    ]
    bundle = build_sample_bundle(selected_snippets=selected_snippets)

    advisor_inputs = bundle.advisor_inputs()

    assert "code_snippets" in advisor_inputs
    assert "profiling_summary" in advisor_inputs
    assert "telemetry_summary" in advisor_inputs
    assert "telemetry_struct" in advisor_inputs

    # The selected snippets should be joined together into the code_snippets field.
    expected_code_snippets = "\n\n".join(selected_snippets)
    assert advisor_inputs["code_snippets"] == expected_code_snippets

    # The other fields should map directly from profiling/telemetry summaries.
    assert advisor_inputs["profiling_summary"] == bundle.profiling.summary_text
    assert advisor_inputs["telemetry_summary"] == bundle.telemetry.summary_text
    assert advisor_inputs["telemetry_struct"] == bundle.telemetry.metrics


def test_analysis_bundle_advisor_inputs_falls_back_to_full_source_text() -> None:
    """
    advisor_inputs() should fall back to the full source text when no
    selected snippets were provided.
    """
    bundle = build_sample_bundle(selected_snippets=[])

    advisor_inputs = bundle.advisor_inputs()

    assert advisor_inputs["code_snippets"] == bundle.source_text
    assert advisor_inputs["profiling_summary"] == bundle.profiling.summary_text
    assert advisor_inputs["telemetry_summary"] == bundle.telemetry.summary_text
    assert advisor_inputs["telemetry_struct"] == bundle.telemetry.metrics


def test_analysis_bundle_short_description_contains_key_fields() -> None:
    """
    short_description() should provide a compact summary that includes useful
    identifiers and feature counts for debugging/logging.
    """
    bundle = build_sample_bundle()

    description = bundle.short_description()

    assert "AnalysisBundle(" in description
    assert "sample_benchmark" in description
    assert "functions=1" in description
    assert "loops=2" in description
    assert "has_mpi=False" in description
    assert "has_openmp=True" in description


# ---------------------------------------------------------------------------
# Tests for AnalysisBundle.from_source_path
# ---------------------------------------------------------------------------

def test_analysis_bundle_from_source_path_reads_real_source_file() -> None:
    """
    from_source_path() should read source text from disk and use the provided
    structured summaries to build a full AnalysisBundle.
    """
    source_path = BENCHMARKS_DIR / "mem_saxpy" / "main.c"

    code_analysis = build_sample_code_analysis(source_path=str(source_path))
    profiling = build_sample_profiling()
    telemetry = build_sample_telemetry()

    bundle = AnalysisBundle.from_source_path(
        benchmark_name="mem_saxpy",
        source_path=source_path,
        code_analysis=code_analysis,
        profiling=profiling,
        telemetry=telemetry,
        metadata={"source": "unit_test"},
        selected_snippets=["y[i] = a * x[i] + y[i];"],
    )

    assert bundle.benchmark_name == "mem_saxpy"
    assert bundle.source_path == str(source_path)
    assert isinstance(bundle.source_text, str)
    assert len(bundle.source_text) > 0
    assert "main" in bundle.source_text
    assert bundle.metadata == {"source": "unit_test"}
    assert bundle.selected_snippets == ["y[i] = a * x[i] + y[i];"]


def test_analysis_bundle_from_source_path_preserves_structured_inputs() -> None:
    """
    from_source_path() should not alter the structured code_analysis,
    profiling, or telemetry objects passed into it.
    """
    source_path = BENCHMARKS_DIR / "omp_imbalance" / "main.c"

    code_analysis = build_sample_code_analysis(source_path=str(source_path))
    profiling = build_sample_profiling()
    telemetry = build_sample_telemetry()

    bundle = AnalysisBundle.from_source_path(
        benchmark_name="omp_imbalance",
        source_path=source_path,
        code_analysis=code_analysis,
        profiling=profiling,
        telemetry=telemetry,
    )

    assert bundle.code_analysis is code_analysis
    assert bundle.profiling is profiling
    assert bundle.telemetry is telemetry


# ---------------------------------------------------------------------------
# Stability / interface shape tests
# ---------------------------------------------------------------------------

def test_analysis_bundle_advisor_inputs_matches_current_advisor_interface() -> None:
    """
    advisor_inputs() should return exactly the four fields expected by the
    current MoE advisor.run(...) interface.
    """
    bundle = build_sample_bundle()

    advisor_inputs = bundle.advisor_inputs()

    assert set(advisor_inputs.keys()) == {
        "code_snippets",
        "profiling_summary",
        "telemetry_summary",
        "telemetry_struct",
    }


def test_analysis_bundle_to_dict_metadata_and_selected_snippets_are_preserved() -> None:
    """
    Metadata and selected snippets should survive serialization because they are
    important for experiment bookkeeping and prompt construction.
    """
    bundle = build_sample_bundle(
        selected_snippets=["snippet one", "snippet two"]
    )

    bundle_dict = bundle.to_dict()

    assert bundle_dict["metadata"]["iteration"] == 1
    assert bundle_dict["metadata"]["benchmark_type"] == "synthetic"
    assert bundle_dict["selected_snippets"] == ["snippet one", "snippet two"]