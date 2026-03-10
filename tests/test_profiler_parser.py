"""
test_profiler_parser.py

Unit tests for implementation.analysis.profiler_parser.ProfilerParser.

These tests verify that the profiler parser can:
- handle empty profiling text safely
- extract runtime information from simple text logs
- extract hotspot function names when explicitly present
- infer keyword-based profiling signals
- parse profiling text from files
- normalize precomputed metrics dictionaries

The parser is currently lightweight and heuristic, so these tests are written
to validate the intended behavior of the current implementation rather than a
full profiler pipeline.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from implementation.analysis import ProfilerParser
from implementation.analysis.analysis_bundle import ProfilingSummary


# ---------------------------------------------------------------------------
# Basic construction / return type
# ---------------------------------------------------------------------------

def test_parse_from_text_returns_profiling_summary() -> None:
    """Parser should return a ProfilingSummary object for raw text input."""
    parser = ProfilerParser()
    raw_text = "runtime: 0.123 seconds"

    summary = parser.parse_from_text(raw_text)

    assert isinstance(summary, ProfilingSummary)


# ---------------------------------------------------------------------------
# Empty / minimal input behavior
# ---------------------------------------------------------------------------

def test_parse_from_text_empty_input_returns_default_summary() -> None:
    """
    Empty profiling text should not crash parsing and should instead return a
    default summary with helpful notes.
    """
    parser = ProfilerParser()

    summary = parser.parse_from_text("")

    assert summary.runtime_seconds is None
    assert summary.hotspot_function is None
    assert summary.raw_metrics == {}
    assert summary.summary_text == "No profiling data was available."
    assert len(summary.notes) > 0
    assert "empty profiling text" in summary.notes[0].lower()


def test_parse_from_text_whitespace_only_returns_default_summary() -> None:
    """Whitespace-only input should be treated the same as empty input."""
    parser = ProfilerParser()

    summary = parser.parse_from_text("   \n\t   ")

    assert summary.runtime_seconds is None
    assert summary.hotspot_function is None
    assert summary.raw_metrics == {}
    assert summary.summary_text == "No profiling data was available."


# ---------------------------------------------------------------------------
# Runtime extraction from text
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("raw_text", "expected_runtime"),
    [
        ("runtime: 0.500 seconds", 0.500),
        ("time = 1.25 sec", 1.25),
        ("elapsed time: 2.75 s", 2.75),
        ("Mean Time (s) 0.05207", 0.05207),
    ],
)
def test_parse_from_text_extracts_runtime_from_supported_patterns(
    raw_text: str,
    expected_runtime: float,
) -> None:
    """
    Parser should extract runtime values from the common supported text
    patterns defined in the implementation.
    """
    parser = ProfilerParser()

    summary = parser.parse_from_text(raw_text)

    assert summary.runtime_seconds == pytest.approx(expected_runtime)
    assert summary.raw_metrics["runtime_seconds"] == pytest.approx(expected_runtime)
    assert "runtime" in summary.summary_text.lower()


def test_parse_from_text_without_runtime_reports_runtime_missing() -> None:
    """
    If no runtime pattern is present, the parser should leave runtime_seconds as
    None and state that runtime information was not explicitly available.
    """
    parser = ProfilerParser()
    raw_text = "profiling completed successfully; no explicit timing field found"

    summary = parser.parse_from_text(raw_text)

    assert summary.runtime_seconds is None
    assert "runtime information was not explicitly available" in summary.summary_text.lower()


# ---------------------------------------------------------------------------
# Hotspot extraction from text
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("raw_text", "expected_hotspot"),
    [
        ("hotspot: matmul_kernel", "matmul_kernel"),
        ("hot function = compute_step", "compute_step"),
        ("top function: main_loop", "main_loop"),
    ],
)
def test_parse_from_text_extracts_hotspot_function_when_present(
    raw_text: str,
    expected_hotspot: str,
) -> None:
    """Parser should extract hotspot function names from supported text fields."""
    parser = ProfilerParser()

    summary = parser.parse_from_text(raw_text)

    assert summary.hotspot_function == expected_hotspot
    assert summary.raw_metrics["hotspot_function"] == expected_hotspot


def test_parse_from_text_without_hotspot_leaves_hotspot_none() -> None:
    """If no hotspot field is present, hotspot_function should remain None."""
    parser = ProfilerParser()
    raw_text = "runtime: 0.321 seconds"

    summary = parser.parse_from_text(raw_text)

    assert summary.hotspot_function is None
    assert "hotspot_function" not in summary.raw_metrics


# ---------------------------------------------------------------------------
# Keyword-based signal inference from text
# ---------------------------------------------------------------------------

def test_parse_from_text_infers_communication_parallelism_and_memory_signals() -> None:
    """
    The parser should infer coarse signal counts/presence flags based on
    performance-related keywords in the raw text.
    """
    parser = ProfilerParser()
    raw_text = """
    runtime: 0.420 seconds
    MPI barrier wait dominates communication overhead.
    OpenMP thread imbalance and parallel scheduling issues observed.
    High memory bandwidth pressure and cache miss behavior detected.
    """

    summary = parser.parse_from_text(raw_text)

    assert summary.raw_metrics["communication_signal_present"] is True
    assert summary.raw_metrics["parallelism_signal_present"] is True
    assert summary.raw_metrics["memory_signal_present"] is True

    assert summary.raw_metrics["communication_signal_count"] > 0
    assert summary.raw_metrics["parallelism_signal_count"] > 0
    assert summary.raw_metrics["memory_signal_count"] > 0


def test_parse_from_text_builds_hotspot_description_from_inferred_signals() -> None:
    """
    Even when no explicit hotspot function is present, the parser should build a
    hotspot_description from inferred high-level signals when possible.
    """
    parser = ProfilerParser()
    raw_text = """
    elapsed time: 0.90 sec
    communication overhead from MPI wait and barrier operations
    memory bandwidth pressure and cache miss activity
    """

    summary = parser.parse_from_text(raw_text)

    assert isinstance(summary.hotspot_description, str)
    assert summary.hotspot_description.strip() != ""
    assert "communication-related overhead" in summary.hotspot_description.lower()
    assert "memory-system pressure" in summary.hotspot_description.lower()


# ---------------------------------------------------------------------------
# File-based parsing
# ---------------------------------------------------------------------------

def test_parse_from_file_reads_and_parses_log_file(tmp_path: Path) -> None:
    """
    parse_from_file() should read profiling text from disk, parse it correctly,
    and attach a note indicating the source file path.
    """
    parser = ProfilerParser()
    log_path = tmp_path / "profile.log"
    log_path.write_text(
        "runtime: 1.234 seconds\nhotspot: matmul_kernel\n",
        encoding="utf-8",
    )

    summary = parser.parse_from_file(log_path)

    assert summary.runtime_seconds == pytest.approx(1.234)
    assert summary.hotspot_function == "matmul_kernel"
    assert len(summary.notes) > 0
    assert str(log_path) in summary.notes[-1]


def test_parse_from_file_raises_for_missing_file(tmp_path: Path) -> None:
    """parse_from_file() should raise FileNotFoundError for a missing path."""
    parser = ProfilerParser()
    missing_path = tmp_path / "missing_profile.log"

    with pytest.raises(FileNotFoundError):
        parser.parse_from_file(missing_path)


# ---------------------------------------------------------------------------
# Metrics-dictionary normalization
# ---------------------------------------------------------------------------

def test_parse_from_metrics_normalizes_runtime_and_hotspot_fields() -> None:
    """
    parse_from_metrics() should normalize common runtime/hotspot keys into the
    canonical fields used by ProfilingSummary.
    """
    parser = ProfilerParser()
    metrics = {
        "runtime": 0.777,
        "hotspot": "compute_kernel",
        "custom_metric": 42,
    }

    summary = parser.parse_from_metrics(metrics)

    assert summary.runtime_seconds == pytest.approx(0.777)
    assert summary.hotspot_function == "compute_kernel"
    assert summary.raw_metrics["runtime_seconds"] == pytest.approx(0.777)
    assert summary.raw_metrics["hotspot_function"] == "compute_kernel"
    assert summary.raw_metrics["custom_metric"] == 42


def test_parse_from_metrics_infers_communication_parallelism_and_memory_signals() -> None:
    """
    parse_from_metrics() should infer coarse signal presence/counts from known
    HPC-oriented numeric metrics.
    """
    parser = ProfilerParser()
    metrics = {
        "runtime_seconds": 1.5,
        "mpi_wait_ratio": 0.3,
        "load_imbalance_score": 0.4,
        "memory_bound_score": 0.7,
    }

    summary = parser.parse_from_metrics(metrics)

    assert summary.raw_metrics["communication_signal_present"] is True
    assert summary.raw_metrics["parallelism_signal_present"] is True
    assert summary.raw_metrics["memory_signal_present"] is True

    assert summary.raw_metrics["communication_signal_count"] >= 1
    assert summary.raw_metrics["parallelism_signal_count"] >= 1
    assert summary.raw_metrics["memory_signal_count"] >= 1


def test_parse_from_metrics_handles_missing_runtime_and_hotspot_gracefully() -> None:
    """
    If the metrics dictionary lacks runtime and hotspot fields, the parser
    should still return a valid summary with those fields unset.
    """
    parser = ProfilerParser()
    metrics = {"cache_miss_rate": 0.12}

    summary = parser.parse_from_metrics(metrics)

    assert summary.runtime_seconds is None
    assert summary.hotspot_function is None
    assert summary.raw_metrics["cache_miss_rate"] == 0.12
    assert "runtime information was not explicitly available" in summary.summary_text.lower()


def test_parse_from_metrics_records_creation_note() -> None:
    """parse_from_metrics() should include a note describing the input mode."""
    parser = ProfilerParser()
    metrics = {"runtime_seconds": 0.222}

    summary = parser.parse_from_metrics(metrics)

    assert len(summary.notes) > 0
    assert "precomputed metrics dictionary" in summary.notes[0].lower()


# ---------------------------------------------------------------------------
# Low-level helper behavior
# ---------------------------------------------------------------------------

def test_is_positive_numeric_accepts_positive_numbers() -> None:
    """Positive numeric values should be recognized as valid positive metrics."""
    parser = ProfilerParser()

    assert parser._is_positive_numeric(1.0) is True  # noqa: SLF001
    assert parser._is_positive_numeric("0.5") is True  # noqa: SLF001


def test_is_positive_numeric_rejects_zero_negative_and_invalid_values() -> None:
    """Zero, negative, and non-numeric values should not count as positive metrics."""
    parser = ProfilerParser()

    assert parser._is_positive_numeric(0.0) is False  # noqa: SLF001
    assert parser._is_positive_numeric(-1.0) is False  # noqa: SLF001
    assert parser._is_positive_numeric("not_a_number") is False  # noqa: SLF001
    assert parser._is_positive_numeric(None) is False  # noqa: SLF001