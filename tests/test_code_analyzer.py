"""
test_code_analyzer.py

Unit tests for implementation.analysis.code_analyzer.CodeAnalyzer.

These tests focus on the current lightweight static-analysis behavior:
- file metadata extraction
- detection of loops, MPI calls, and OpenMP pragmas
- region extraction
- summary generation

The tests use the existing benchmark sources in this repository so they remain
grounded in the current temp_moe codebase.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from implementation.analysis import CodeAnalyzer
from implementation.analysis.analysis_bundle import CodeAnalysisSummary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS_DIR = REPO_ROOT / "benchmarks"


def benchmark_source(benchmark_name: str) -> Path:
    """Return the path to the main source file for a benchmark."""
    return BENCHMARKS_DIR / benchmark_name / "main.c"


# ---------------------------------------------------------------------------
# Basic construction / file handling
# ---------------------------------------------------------------------------

def test_analyze_returns_code_analysis_summary() -> None:
    """Analyzer should return the expected summary object type."""
    analyzer = CodeAnalyzer()
    source_path = benchmark_source("mem_saxpy")

    summary = analyzer.analyze(source_path)

    assert isinstance(summary, CodeAnalysisSummary)


def test_analyze_raises_for_missing_file() -> None:
    """Analyzer should raise FileNotFoundError for a missing source path."""
    analyzer = CodeAnalyzer()
    missing_path = BENCHMARKS_DIR / "does_not_exist" / "main.c"

    with pytest.raises(FileNotFoundError):
        analyzer.analyze(missing_path)


# ---------------------------------------------------------------------------
# Metadata / general structure
# ---------------------------------------------------------------------------

def test_mem_saxpy_basic_metadata() -> None:
    """mem_saxpy analysis should populate core metadata fields."""
    analyzer = CodeAnalyzer()
    source_path = benchmark_source("mem_saxpy")

    summary = analyzer.analyze(source_path)

    assert summary.source_path == str(source_path)
    assert summary.file_name == "main.c"
    assert summary.language == "c"
    assert summary.file_size_bytes > 0
    assert summary.line_count > 0
    assert isinstance(summary.summary_text, str)
    assert summary.summary_text.strip() != ""


def test_analyze_extracts_some_regions() -> None:
    """Analyzer should extract at least one code region from a benchmark file."""
    analyzer = CodeAnalyzer()
    source_path = benchmark_source("mem_saxpy")

    summary = analyzer.analyze(source_path)

    assert len(summary.regions) > 0


def test_region_line_ranges_are_valid() -> None:
    """All extracted regions should have valid line spans and non-empty snippets."""
    analyzer = CodeAnalyzer()
    source_path = benchmark_source("mem_saxpy")

    summary = analyzer.analyze(source_path)

    for region in summary.regions:
        assert region.start_line >= 1
        assert region.end_line >= region.start_line
        assert isinstance(region.snippet, str)
        assert region.snippet.strip() != ""


def test_regions_are_sorted_by_source_position() -> None:
    """Merged regions should be ordered in source order."""
    analyzer = CodeAnalyzer()
    source_path = benchmark_source("mem_saxpy")

    summary = analyzer.analyze(source_path)

    ordered_pairs = [(region.start_line, region.end_line, region.region_type) for region in summary.regions]
    assert ordered_pairs == sorted(ordered_pairs)


# ---------------------------------------------------------------------------
# Benchmark-specific feature detection
# ---------------------------------------------------------------------------

def test_mem_saxpy_detects_loops_and_memory_hint() -> None:
    """
    mem_saxpy should look loop-based, and it should likely appear memory-intensive
    under the current heuristic analyzer.
    """
    analyzer = CodeAnalyzer()
    source_path = benchmark_source("mem_saxpy")

    summary = analyzer.analyze(source_path)

    assert summary.loop_count > 0
    assert summary.appears_memory_intensive is True


def test_mpi_pingpong_detects_mpi_usage() -> None:
    """mpi_pingpong should be recognized as MPI-related code."""
    analyzer = CodeAnalyzer()
    source_path = benchmark_source("mpi_pingpong")

    summary = analyzer.analyze(source_path)

    assert summary.has_mpi is True
    assert summary.mpi_call_count > 0

    mpi_regions = [region for region in summary.regions if region.region_type == "mpi"]
    assert len(mpi_regions) > 0


def test_omp_imbalance_detects_openmp_usage() -> None:
    """omp_imbalance should be recognized as OpenMP-related code."""
    analyzer = CodeAnalyzer()
    source_path = benchmark_source("omp_imbalance")

    summary = analyzer.analyze(source_path)

    assert summary.has_openmp is True
    assert summary.omp_pragma_count > 0

    omp_regions = [region for region in summary.regions if region.region_type == "openmp"]
    assert len(omp_regions) > 0


# ---------------------------------------------------------------------------
# Region typing
# ---------------------------------------------------------------------------

def test_function_regions_have_function_tag() -> None:
    """Function regions should be labeled consistently."""
    analyzer = CodeAnalyzer()
    source_path = benchmark_source("mem_saxpy")

    summary = analyzer.analyze(source_path)

    function_regions = [region for region in summary.regions if region.region_type == "function"]

    # Current benchmarks should have at least one function definition.
    assert len(function_regions) > 0
    for region in function_regions:
        assert "function" in region.tags


def test_loop_regions_have_loop_tag() -> None:
    """Loop regions should be labeled with loop-related tags."""
    analyzer = CodeAnalyzer()
    source_path = benchmark_source("mem_saxpy")

    summary = analyzer.analyze(source_path)

    loop_regions = [region for region in summary.regions if region.region_type == "loop"]

    assert len(loop_regions) > 0
    for region in loop_regions:
        assert "loop" in region.tags


def test_mpi_regions_have_communication_tag() -> None:
    """MPI regions should carry MPI/communication tags."""
    analyzer = CodeAnalyzer()
    source_path = benchmark_source("mpi_pingpong")

    summary = analyzer.analyze(source_path)

    mpi_regions = [region for region in summary.regions if region.region_type == "mpi"]

    assert len(mpi_regions) > 0
    for region in mpi_regions:
        assert "mpi" in region.tags
        assert "communication" in region.tags


def test_openmp_regions_have_parallel_tag() -> None:
    """OpenMP pragma regions should carry OpenMP/parallel tags."""
    analyzer = CodeAnalyzer()
    source_path = benchmark_source("omp_imbalance")

    summary = analyzer.analyze(source_path)

    omp_regions = [region for region in summary.regions if region.region_type == "openmp"]

    assert len(omp_regions) > 0
    for region in omp_regions:
        assert "openmp" in region.tags
        assert "parallel" in region.tags


# ---------------------------------------------------------------------------
# Summary text sanity
# ---------------------------------------------------------------------------

def test_summary_text_mentions_mpi_for_mpi_benchmark() -> None:
    """Human-readable summary should mention MPI when MPI is present."""
    analyzer = CodeAnalyzer()
    source_path = benchmark_source("mpi_pingpong")

    summary = analyzer.analyze(source_path)

    assert "mpi" in summary.summary_text.lower()


def test_summary_text_mentions_openmp_for_openmp_benchmark() -> None:
    """Human-readable summary should mention OpenMP when OpenMP is present."""
    analyzer = CodeAnalyzer()
    source_path = benchmark_source("omp_imbalance")

    summary = analyzer.analyze(source_path)

    assert "openmp" in summary.summary_text.lower()


# ---------------------------------------------------------------------------
# Internal helper behavior
# ---------------------------------------------------------------------------

def test_find_matching_brace_simple_case() -> None:
    """The brace-matching helper should find the correct closing brace."""
    analyzer = CodeAnalyzer()
    text = "int main() { int x = 0; if (x) { x++; } return 0; }"

    open_idx = text.index("{")
    close_idx = analyzer._find_matching_brace(text, open_idx)  # noqa: SLF001

    assert close_idx is not None
    assert text[close_idx] == "}"


def test_char_span_to_line_span_maps_correctly() -> None:
    """Character spans should convert to 1-indexed line ranges correctly."""
    analyzer = CodeAnalyzer()
    text = "line1\nline2\nline3\nline4"

    start_idx = text.index("line2")
    end_idx = text.index("line3") + len("line3")

    start_line, end_line = analyzer._char_span_to_line_span(text, start_idx, end_idx)  # noqa: SLF001

    assert start_line == 2
    assert end_line == 3