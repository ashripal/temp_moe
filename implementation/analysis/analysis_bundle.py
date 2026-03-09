"""
analysis_bundle.py

Defines the structured data containers used by the analysis stage.

This file is intentionally lightweight and dependency-free. Its main role is to
standardize how static code analysis, profiling summaries, and telemetry signals
are packaged together before being passed into the MoE advisor and, later, the
generator stage.

Design goals:
- Keep the interface explicit and easy to inspect
- Make it simple to serialize to dictionaries / JSON
- Provide small helper methods for downstream integration
- Avoid embedding analysis logic here; this file is for data structure only
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Source-level analysis structures
# ---------------------------------------------------------------------------

@dataclass
class CodeRegion:
    """
    Represents a meaningful region of code identified during static analysis.

    Examples:
    - a function
    - a loop nest
    - a block containing MPI calls
    - a block containing OpenMP pragmas

    The goal is to preserve enough context for the advisor to reason about where
    a potential optimization might apply.
    """
    label: str
    region_type: str
    start_line: int
    end_line: int
    snippet: str
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the region into a plain dictionary."""
        return asdict(self)


@dataclass
class CodeAnalysisSummary:
    """
    Stores the output of static source-code inspection.

    This is the structured result produced by code_analyzer.py.
    """
    source_path: str
    language: str
    file_name: str
    file_size_bytes: int
    line_count: int

    # High-level static feature counts
    function_count: int = 0
    loop_count: int = 0
    mpi_call_count: int = 0
    omp_pragma_count: int = 0

    # Human-readable summary of the code
    summary_text: str = ""

    # Selected source regions that may be useful to downstream modules
    regions: List[CodeRegion] = field(default_factory=list)

    # Optional flags for quick downstream checks
    has_mpi: bool = False
    has_openmp: bool = False
    appears_memory_intensive: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert the summary into a plain dictionary."""
        data = asdict(self)
        data["regions"] = [region.to_dict() for region in self.regions]
        return data


# ---------------------------------------------------------------------------
# Profiling / runtime analysis structures
# ---------------------------------------------------------------------------

@dataclass
class ProfilingSummary:
    """
    Stores parsed performance or profiler information.

    This is the normalized result produced by profiler_parser.py. It is intended
    to capture runtime-oriented evidence in a form that is consistent across
    benchmarks, even if the raw profiler formats differ.
    """
    runtime_seconds: Optional[float] = None
    hotspot_function: Optional[str] = None
    hotspot_description: str = ""

    # General-purpose parsed metrics
    raw_metrics: Dict[str, Any] = field(default_factory=dict)

    # Human-readable summary used directly in prompts
    summary_text: str = ""

    # Optional notes about parser assumptions or missing data
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the profiling summary into a plain dictionary."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Telemetry structures
# ---------------------------------------------------------------------------

@dataclass
class TelemetrySummary:
    """
    Stores normalized telemetry features derived from static analysis and/or
    profiling results.

    This object is especially important because the MoE router already expects a
    structured telemetry dictionary for expert selection.
    """
    metrics: Dict[str, float] = field(default_factory=dict)
    summary_text: str = ""
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the telemetry summary into a plain dictionary."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Top-level analysis bundle
# ---------------------------------------------------------------------------

@dataclass
class AnalysisBundle:
    """
    Canonical container for all analysis-stage outputs.

    This object should be the primary output of the analysis pipeline and the
    primary input to the advisor stage.

    Expected flow:
        code_analyzer.py       -> CodeAnalysisSummary
        profiler_parser.py     -> ProfilingSummary
        telemetry_extractor.py -> TelemetrySummary
        analysis_bundle.py     -> AnalysisBundle
    """
    benchmark_name: str
    source_path: str

    # Full raw source text, useful for prompt construction and downstream logic
    source_text: str

    # Structured analysis outputs
    code_analysis: CodeAnalysisSummary
    profiling: ProfilingSummary
    telemetry: TelemetrySummary

    # Optional metadata for experiment bookkeeping
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Optional selected snippets for advisor prompts
    selected_snippets: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the entire bundle into a plain dictionary.

        Useful for:
        - debugging
        - serialization
        - JSON logging
        """
        return {
            "benchmark_name": self.benchmark_name,
            "source_path": self.source_path,
            "source_text": self.source_text,
            "code_analysis": self.code_analysis.to_dict(),
            "profiling": self.profiling.to_dict(),
            "telemetry": self.telemetry.to_dict(),
            "metadata": self.metadata,
            "selected_snippets": self.selected_snippets,
        }

    def advisor_inputs(self) -> Dict[str, Any]:
        """
        Return the exact fields needed by the current MoE advisor.

        Your current advisor.run(...) expects:
            - code_snippets: str
            - profiling_summary: str
            - telemetry_summary: str
            - telemetry_struct: Dict[str, float]

        This helper keeps the rest of the pipeline clean and avoids repeated
        formatting logic in multiple places.
        """
        code_snippets = "\n\n".join(self.selected_snippets).strip()

        # Fall back to the full source text if no snippets were pre-selected.
        if not code_snippets:
            code_snippets = self.source_text

        return {
            "code_snippets": code_snippets,
            "profiling_summary": self.profiling.summary_text,
            "telemetry_summary": self.telemetry.summary_text,
            "telemetry_struct": self.telemetry.metrics,
        }

    def short_description(self) -> str:
        """
        Return a compact human-readable description of the bundle.

        Useful for logs, debugging, and quick test assertions.
        """
        return (
            f"AnalysisBundle("
            f"benchmark_name={self.benchmark_name!r}, "
            f"source_path={self.source_path!r}, "
            f"functions={self.code_analysis.function_count}, "
            f"loops={self.code_analysis.loop_count}, "
            f"has_mpi={self.code_analysis.has_mpi}, "
            f"has_openmp={self.code_analysis.has_openmp}, "
            f"telemetry_keys={list(self.telemetry.metrics.keys())}"
            f")"
        )

    @classmethod
    def from_source_path(
        cls,
        benchmark_name: str,
        source_path: str | Path,
        code_analysis: CodeAnalysisSummary,
        profiling: ProfilingSummary,
        telemetry: TelemetrySummary,
        metadata: Optional[Dict[str, Any]] = None,
        selected_snippets: Optional[List[str]] = None,
    ) -> "AnalysisBundle":
        """
        Convenience constructor that reads the source text from disk.

        This is useful when the code analyzer has already identified the source
        path and produced structured analysis, and we now want to package the
        full bundle without duplicating file-reading logic elsewhere.
        """
        source_path = str(source_path)
        source_text = Path(source_path).read_text(encoding="utf-8")

        return cls(
            benchmark_name=benchmark_name,
            source_path=source_path,
            source_text=source_text,
            code_analysis=code_analysis,
            profiling=profiling,
            telemetry=telemetry,
            metadata=metadata or {},
            selected_snippets=selected_snippets or [],
        )