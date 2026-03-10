"""
test_analysis_integration.py

Integration tests for the analysis pipeline:

    CodeAnalyzer -> ProfilerParser -> TelemetryExtractor -> AnalysisBundle

These tests verify that the current analysis components work together and
produce advisor-ready outputs on the real benchmark sources in this repository.

The goal here is not to validate perfect profiling realism, but to ensure that:
- real source files can be analyzed successfully
- profiling summaries can be parsed and integrated
- telemetry is produced in the expected structure
- the final AnalysisBundle is ready for the current MoE advisor interface
"""

from __future__ import annotations

from pathlib import Path

from implementation.analysis import (
    AnalysisBundle,
    CodeAnalyzer,
    ProfilerParser,
    TelemetryExtractor,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS_DIR = REPO_ROOT / "benchmarks"


def benchmark_source(benchmark_name: str) -> Path:
    """Return the path to the main source file for a benchmark."""
    return BENCHMARKS_DIR / benchmark_name / "main.c"


# ---------------------------------------------------------------------------
# End-to-end analysis flow tests
# ---------------------------------------------------------------------------

def test_analysis_pipeline_mem_saxpy_end_to_end() -> None:
    """
    The full analysis flow should work on the mem_saxpy benchmark and produce
    a valid advisor-ready AnalysisBundle.
    """
    analyzer = CodeAnalyzer()
    parser = ProfilerParser()
    extractor = TelemetryExtractor()

    source_path = benchmark_source("mem_saxpy")

    # Step 1: analyze the real benchmark source file.
    code_analysis = analyzer.analyze(source_path)

    # Step 2: simulate profiling text that is plausible for a memory-oriented kernel.
    profiling = parser.parse_from_text(
        """
        runtime: 0.05207 seconds
        memory bandwidth pressure and cache miss behavior detected
        hotspot: saxpy_kernel
        """
    )

    # Step 3: convert static + profiling evidence into telemetry.
    telemetry = extractor.extract(code_analysis, profiling)

    # Step 4: package everything into the top-level bundle.
    bundle = AnalysisBundle.from_source_path(
        benchmark_name="mem_saxpy",
        source_path=source_path,
        code_analysis=code_analysis,
        profiling=profiling,
        telemetry=telemetry,
        metadata={"test_case": "integration_mem_saxpy"},
    )

    # Validate the integrated object.
    assert bundle.benchmark_name == "mem_saxpy"
    assert bundle.source_path == str(source_path)
    assert len(bundle.source_text) > 0

    # Static analysis should have discovered useful source structure.
    assert bundle.code_analysis.loop_count > 0
    assert bundle.code_analysis.file_name == "main.c"

    # Profiling summary should carry parsed runtime evidence.
    assert bundle.profiling.runtime_seconds == 0.05207
    assert bundle.profiling.hotspot_function == "saxpy_kernel"

    # Telemetry should expose router-facing scores.
    assert "router_comm_score" in bundle.telemetry.metrics
    assert "router_parallel_score" in bundle.telemetry.metrics
    assert "router_memory_score" in bundle.telemetry.metrics

    # The advisor input shape should match the current interface.
    advisor_inputs = bundle.advisor_inputs()
    assert set(advisor_inputs.keys()) == {
        "code_snippets",
        "profiling_summary",
        "telemetry_summary",
        "telemetry_struct",
    }
    assert isinstance(advisor_inputs["code_snippets"], str)
    assert isinstance(advisor_inputs["profiling_summary"], str)
    assert isinstance(advisor_inputs["telemetry_summary"], str)
    assert isinstance(advisor_inputs["telemetry_struct"], dict)


def test_analysis_pipeline_mpi_pingpong_end_to_end() -> None:
    """
    The full analysis flow should work on the mpi_pingpong benchmark and
    preserve MPI/communication-oriented signals through the pipeline.
    """
    analyzer = CodeAnalyzer()
    parser = ProfilerParser()
    extractor = TelemetryExtractor()

    source_path = benchmark_source("mpi_pingpong")

    code_analysis = analyzer.analyze(source_path)
    profiling = parser.parse_from_text(
        """
        runtime: 0.02254 seconds
        MPI barrier wait dominates communication overhead
        hotspot: MPI_Send
        """
    )
    telemetry = extractor.extract(code_analysis, profiling)

    bundle = AnalysisBundle.from_source_path(
        benchmark_name="mpi_pingpong",
        source_path=source_path,
        code_analysis=code_analysis,
        profiling=profiling,
        telemetry=telemetry,
        metadata={"test_case": "integration_mpi_pingpong"},
    )

    # Static analysis should identify this source as MPI-related.
    assert bundle.code_analysis.has_mpi is True
    assert bundle.code_analysis.mpi_call_count > 0

    # Profiling should reflect the provided communication-oriented runtime text.
    assert bundle.profiling.runtime_seconds == 0.02254
    assert bundle.profiling.hotspot_function == "MPI_Send"

    # Telemetry should indicate non-trivial communication pressure.
    assert bundle.telemetry.metrics["has_mpi"] == 1.0
    assert bundle.telemetry.metrics["communication_pressure"] > 0.0
    assert "communication" in bundle.telemetry.summary_text.lower()

    # Final integrated bundle should serialize cleanly.
    bundle_dict = bundle.to_dict()
    assert bundle_dict["benchmark_name"] == "mpi_pingpong"
    assert bundle_dict["code_analysis"]["has_mpi"] is True
    assert bundle_dict["profiling"]["hotspot_function"] == "MPI_Send"


def test_analysis_pipeline_omp_imbalance_end_to_end() -> None:
    """
    The full analysis flow should work on the omp_imbalance benchmark and
    preserve OpenMP / parallelism-oriented signals through the pipeline.
    """
    analyzer = CodeAnalyzer()
    parser = ProfilerParser()
    extractor = TelemetryExtractor()

    source_path = benchmark_source("omp_imbalance")

    code_analysis = analyzer.analyze(source_path)
    profiling = parser.parse_from_text(
        """
        elapsed time: 0.02269 sec
        OpenMP thread imbalance and parallel scheduling issues observed
        hotspot: compute_region
        """
    )
    telemetry = extractor.extract(code_analysis, profiling)

    bundle = AnalysisBundle.from_source_path(
        benchmark_name="omp_imbalance",
        source_path=source_path,
        code_analysis=code_analysis,
        profiling=profiling,
        telemetry=telemetry,
        metadata={"test_case": "integration_omp_imbalance"},
    )

    # Static analysis should identify this source as OpenMP-related.
    assert bundle.code_analysis.has_openmp is True
    assert bundle.code_analysis.omp_pragma_count > 0

    # Profiling should preserve the runtime and hotspot evidence.
    assert bundle.profiling.runtime_seconds == 0.02269
    assert bundle.profiling.hotspot_function == "compute_region"

    # Telemetry should indicate some degree of parallelism pressure.
    assert bundle.telemetry.metrics["has_openmp"] == 1.0
    assert bundle.telemetry.metrics["parallelism_pressure"] > 0.0

    # The integrated advisor inputs should remain well-formed.
    advisor_inputs = bundle.advisor_inputs()
    assert isinstance(advisor_inputs["code_snippets"], str)
    assert isinstance(advisor_inputs["telemetry_struct"], dict)
    assert "router_parallel_score" in advisor_inputs["telemetry_struct"]


# ---------------------------------------------------------------------------
# AnalysisBundle selection behavior in integrated flow
# ---------------------------------------------------------------------------

def test_analysis_pipeline_uses_selected_snippets_when_provided() -> None:
    """
    When selected snippets are supplied during bundle construction, the final
    advisor inputs should prefer those snippets over the full source text.
    """
    analyzer = CodeAnalyzer()
    parser = ProfilerParser()
    extractor = TelemetryExtractor()

    source_path = benchmark_source("mem_saxpy")

    code_analysis = analyzer.analyze(source_path)
    profiling = parser.parse_from_text("runtime: 0.100 seconds")
    telemetry = extractor.extract(code_analysis, profiling)

    selected_snippets = [
        "for (int i = 0; i < n; i++) {",
        "    y[i] = a * x[i] + y[i];",
        "}",
    ]

    bundle = AnalysisBundle.from_source_path(
        benchmark_name="mem_saxpy",
        source_path=source_path,
        code_analysis=code_analysis,
        profiling=profiling,
        telemetry=telemetry,
        selected_snippets=selected_snippets,
    )

    advisor_inputs = bundle.advisor_inputs()

    assert advisor_inputs["code_snippets"] == "\n\n".join(selected_snippets)
    assert advisor_inputs["code_snippets"] != bundle.source_text


def test_analysis_pipeline_falls_back_to_full_source_when_no_snippets_selected() -> None:
    """
    If no selected snippets are provided, the integrated analysis bundle should
    fall back to the full source text for advisor consumption.
    """
    analyzer = CodeAnalyzer()
    parser = ProfilerParser()
    extractor = TelemetryExtractor()

    source_path = benchmark_source("mem_saxpy")

    code_analysis = analyzer.analyze(source_path)
    profiling = parser.parse_from_text("runtime: 0.100 seconds")
    telemetry = extractor.extract(code_analysis, profiling)

    bundle = AnalysisBundle.from_source_path(
        benchmark_name="mem_saxpy",
        source_path=source_path,
        code_analysis=code_analysis,
        profiling=profiling,
        telemetry=telemetry,
        selected_snippets=[],
    )

    advisor_inputs = bundle.advisor_inputs()

    assert advisor_inputs["code_snippets"] == bundle.source_text


# ---------------------------------------------------------------------------
# Internal consistency tests across analysis stages
# ---------------------------------------------------------------------------

def test_analysis_pipeline_outputs_are_internally_consistent() -> None:
    """
    The integrated analysis pipeline should produce outputs whose sections are
    mutually consistent:
    - source file metadata matches the selected benchmark
    - profiling is reflected in telemetry
    - telemetry is reflected in advisor inputs
    """
    analyzer = CodeAnalyzer()
    parser = ProfilerParser()
    extractor = TelemetryExtractor()

    source_path = benchmark_source("mpi_pingpong")

    code_analysis = analyzer.analyze(source_path)
    profiling = parser.parse_from_text(
        """
        runtime: 0.030 seconds
        MPI wait and communication overhead detected
        hotspot: MPI_Wait
        """
    )
    telemetry = extractor.extract(code_analysis, profiling)

    bundle = AnalysisBundle.from_source_path(
        benchmark_name="mpi_pingpong",
        source_path=source_path,
        code_analysis=code_analysis,
        profiling=profiling,
        telemetry=telemetry,
    )

    # File-level consistency.
    assert bundle.code_analysis.source_path == str(source_path)
    assert bundle.source_path == str(source_path)

    # Profiling should be reflected in telemetry fields.
    assert bundle.telemetry.metrics["runtime_seconds"] == 0.030
    assert bundle.telemetry.metrics["runtime_available"] == 1.0
    assert bundle.telemetry.metrics["hotspot_identified"] == 1.0

    # MPI-related evidence should remain visible throughout the integrated output.
    assert bundle.code_analysis.has_mpi is True
    assert bundle.telemetry.metrics["has_mpi"] == 1.0
    assert bundle.telemetry.metrics["communication_pressure"] > 0.0

    advisor_inputs = bundle.advisor_inputs()
    assert advisor_inputs["telemetry_struct"]["has_mpi"] == 1.0
    assert advisor_inputs["telemetry_struct"]["communication_pressure"] > 0.0


def test_analysis_pipeline_bundle_short_description_is_useful() -> None:
    """
    The integrated AnalysisBundle short description should include useful
    high-level debugging information after a full pipeline run.
    """
    analyzer = CodeAnalyzer()
    parser = ProfilerParser()
    extractor = TelemetryExtractor()

    source_path = benchmark_source("omp_imbalance")

    code_analysis = analyzer.analyze(source_path)
    profiling = parser.parse_from_text("runtime: 0.200 seconds")
    telemetry = extractor.extract(code_analysis, profiling)

    bundle = AnalysisBundle.from_source_path(
        benchmark_name="omp_imbalance",
        source_path=source_path,
        code_analysis=code_analysis,
        profiling=profiling,
        telemetry=telemetry,
    )

    description = bundle.short_description()

    assert "AnalysisBundle(" in description
    assert "omp_imbalance" in description
    assert "functions=" in description
    assert "loops=" in description
    assert "has_openmp=True" in description


# ---------------------------------------------------------------------------
# Serialization sanity in integrated flow
# ---------------------------------------------------------------------------

def test_analysis_pipeline_bundle_to_dict_is_complete() -> None:
    """
    After a full integrated analysis flow, the final bundle should serialize
    into a dictionary containing all expected major sections.
    """
    analyzer = CodeAnalyzer()
    parser = ProfilerParser()
    extractor = TelemetryExtractor()

    source_path = benchmark_source("mem_saxpy")

    code_analysis = analyzer.analyze(source_path)
    profiling = parser.parse_from_text(
        """
        runtime: 0.052 seconds
        memory bandwidth pressure observed
        hotspot: main
        """
    )
    telemetry = extractor.extract(code_analysis, profiling)

    bundle = AnalysisBundle.from_source_path(
        benchmark_name="mem_saxpy",
        source_path=source_path,
        code_analysis=code_analysis,
        profiling=profiling,
        telemetry=telemetry,
        metadata={"stage": "integration_test"},
    )

    bundle_dict = bundle.to_dict()

    assert set(bundle_dict.keys()) == {
        "benchmark_name",
        "source_path",
        "source_text",
        "code_analysis",
        "profiling",
        "telemetry",
        "metadata",
        "selected_snippets",
    }

    assert bundle_dict["metadata"]["stage"] == "integration_test"
    assert isinstance(bundle_dict["source_text"], str)
    assert isinstance(bundle_dict["code_analysis"], dict)
    assert isinstance(bundle_dict["profiling"], dict)
    assert isinstance(bundle_dict["telemetry"], dict)