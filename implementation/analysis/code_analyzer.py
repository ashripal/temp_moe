"""
code_analyzer.py

Performs lightweight static analysis over benchmark source files.

This module is intentionally simple and deterministic. Its role is to inspect
source code and extract useful structural/contextual signals for downstream
stages, especially the MoE advisor.

Current responsibilities:
- Read source text from disk
- Infer basic language/file metadata
- Count common structural features (functions, loops, MPI calls, OpenMP pragmas)
- Identify meaningful code regions
- Produce a compact CodeAnalysisSummary object

Important note:
This is not a full parser. It uses practical heuristics suitable for the
current C/C++-style benchmark programs in this repository. The goal is to
provide stable and useful context, not perfect parsing.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple

from .analysis_bundle import CodeAnalysisSummary, CodeRegion


class CodeAnalyzer:
    """
    Lightweight static analyzer for benchmark source files.

    This class is designed for the current temp_moe benchmarks, which are
    primarily single-file C programs. It can be extended later if you add
    multi-file programs or richer parsing.

    Typical usage:
        analyzer = CodeAnalyzer()
        summary = analyzer.analyze("benchmarks/mem_saxpy/main.c")
    """

    # Common MPI call prefixes/patterns to detect communication-oriented code.
    MPI_CALL_PATTERN = re.compile(r"\bMPI_[A-Za-z0-9_]+\s*\(")

    # Common OpenMP pragma patterns.
    OMP_PRAGMA_PATTERN = re.compile(r"^\s*#\s*pragma\s+omp\b", re.MULTILINE)

    # Rough loop detectors for C-like languages.
    FOR_LOOP_PATTERN = re.compile(r"\bfor\s*\(")
    WHILE_LOOP_PATTERN = re.compile(r"\bwhile\s*\(")

    # Very lightweight heuristic for function definitions in C/C++.
    # This is intentionally conservative and will work best on your benchmark code.
    FUNCTION_DEF_PATTERN = re.compile(
        r"""
        ^\s*                                   # optional leading whitespace
        (?:[A-Za-z_][\w\s\*\(\)]*?)            # return type-ish prefix
        \s+                                    # space before function name
        ([A-Za-z_]\w*)                         # function name
        \s*                                    # optional whitespace
        \(([^;]*)\)                            # parameter list (not ending with ;)
        \s*                                    # optional whitespace
        \{                                     # opening brace suggests definition
        """,
        re.MULTILINE | re.VERBOSE,
    )

    def analyze(self, source_path: str | Path) -> CodeAnalysisSummary:
        """
        Analyze a source file and return a structured static summary.

        Parameters
        ----------
        source_path:
            Path to the source file to analyze.

        Returns
        -------
        CodeAnalysisSummary
            Structured summary containing file metadata, feature counts,
            detected code regions, and a short human-readable summary.
        """
        source_path = Path(source_path)

        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        source_text = source_path.read_text(encoding="utf-8")
        file_size_bytes = source_path.stat().st_size
        line_count = len(source_text.splitlines())
        language = self._infer_language(source_path)

        # Extract regions and feature counts.
        function_regions = self._extract_function_regions(source_text)
        loop_regions = self._extract_loop_regions(source_text)
        mpi_regions = self._extract_mpi_regions(source_text)
        omp_regions = self._extract_openmp_regions(source_text)

        function_count = len(function_regions)
        loop_count = len(loop_regions)
        mpi_call_count = len(self.MPI_CALL_PATTERN.findall(source_text))
        omp_pragma_count = len(self.OMP_PRAGMA_PATTERN.findall(source_text))

        has_mpi = mpi_call_count > 0
        has_openmp = omp_pragma_count > 0
        appears_memory_intensive = self._infer_memory_intensity(source_text)

        # Merge regions into a single list for downstream consumption.
        all_regions = self._merge_regions(
            function_regions=function_regions,
            loop_regions=loop_regions,
            mpi_regions=mpi_regions,
            omp_regions=omp_regions,
        )

        summary_text = self._build_summary_text(
            file_name=source_path.name,
            language=language,
            function_count=function_count,
            loop_count=loop_count,
            has_mpi=has_mpi,
            has_openmp=has_openmp,
            appears_memory_intensive=appears_memory_intensive,
        )

        return CodeAnalysisSummary(
            source_path=str(source_path),
            language=language,
            file_name=source_path.name,
            file_size_bytes=file_size_bytes,
            line_count=line_count,
            function_count=function_count,
            loop_count=loop_count,
            mpi_call_count=mpi_call_count,
            omp_pragma_count=omp_pragma_count,
            summary_text=summary_text,
            regions=all_regions,
            has_mpi=has_mpi,
            has_openmp=has_openmp,
            appears_memory_intensive=appears_memory_intensive,
        )

    # ---------------------------------------------------------------------
    # Core extraction helpers
    # ---------------------------------------------------------------------

    def _extract_function_regions(self, source_text: str) -> List[CodeRegion]:
        """
        Identify likely function definitions and return them as CodeRegion objects.

        This is a heuristic matcher intended for C-style benchmark programs.
        It attempts to find a function signature and then locate the matching
        brace-delimited body.
        """
        regions: List[CodeRegion] = []

        for match in self.FUNCTION_DEF_PATTERN.finditer(source_text):
            function_name = match.group(1)
            body_start_idx = source_text.find("{", match.start())

            if body_start_idx == -1:
                continue

            body_end_idx = self._find_matching_brace(source_text, body_start_idx)
            if body_end_idx is None:
                continue

            snippet = source_text[match.start():body_end_idx + 1]
            start_line, end_line = self._char_span_to_line_span(
                source_text,
                match.start(),
                body_end_idx + 1,
            )

            regions.append(
                CodeRegion(
                    label=function_name,
                    region_type="function",
                    start_line=start_line,
                    end_line=end_line,
                    snippet=snippet.strip(),
                    tags=["function"],
                )
            )

        return regions

    def _extract_loop_regions(self, source_text: str) -> List[CodeRegion]:
        """
        Identify likely loop regions.

        For the current version, each loop region captures the line containing
        the loop header plus a small local context window. This is enough to
        provide useful evidence to the advisor without requiring a full parser.
        """
        regions: List[CodeRegion] = []
        lines = source_text.splitlines()

        loop_patterns = (
            ("for_loop", self.FOR_LOOP_PATTERN),
            ("while_loop", self.WHILE_LOOP_PATTERN),
        )

        for idx, line in enumerate(lines):
            for label, pattern in loop_patterns:
                if pattern.search(line):
                    start_line = max(1, idx + 1 - 2)
                    end_line = min(len(lines), idx + 1 + 4)
                    snippet = "\n".join(lines[start_line - 1:end_line])

                    tags = ["loop"]
                    if label == "for_loop":
                        tags.append("for")
                    elif label == "while_loop":
                        tags.append("while")

                    if "omp" in snippet.lower():
                        tags.append("openmp_context")

                    regions.append(
                        CodeRegion(
                            label=f"{label}_{len(regions) + 1}",
                            region_type="loop",
                            start_line=start_line,
                            end_line=end_line,
                            snippet=snippet.strip(),
                            tags=tags,
                        )
                    )
                    break

        return regions

    def _extract_mpi_regions(self, source_text: str) -> List[CodeRegion]:
        """
        Identify source regions containing MPI calls.

        Each detected region contains the matching line and a small local context
        window around it.
        """
        regions: List[CodeRegion] = []
        lines = source_text.splitlines()

        for idx, line in enumerate(lines):
            if self.MPI_CALL_PATTERN.search(line):
                start_line = max(1, idx + 1 - 2)
                end_line = min(len(lines), idx + 1 + 2)
                snippet = "\n".join(lines[start_line - 1:end_line])

                # Attempt to extract a cleaner label like MPI_Send / MPI_Barrier.
                call_match = re.search(r"\b(MPI_[A-Za-z0-9_]+)\s*\(", line)
                call_name = call_match.group(1) if call_match else f"mpi_call_{len(regions) + 1}"

                regions.append(
                    CodeRegion(
                        label=call_name,
                        region_type="mpi",
                        start_line=start_line,
                        end_line=end_line,
                        snippet=snippet.strip(),
                        tags=["mpi", "communication"],
                    )
                )

        return regions

    def _extract_openmp_regions(self, source_text: str) -> List[CodeRegion]:
        """
        Identify source regions containing OpenMP pragmas.
        """
        regions: List[CodeRegion] = []
        lines = source_text.splitlines()

        for idx, line in enumerate(lines):
            if self.OMP_PRAGMA_PATTERN.search(line):
                start_line = max(1, idx + 1)
                end_line = min(len(lines), idx + 1 + 3)
                snippet = "\n".join(lines[start_line - 1:end_line])

                regions.append(
                    CodeRegion(
                        label=f"omp_pragma_{len(regions) + 1}",
                        region_type="openmp",
                        start_line=start_line,
                        end_line=end_line,
                        snippet=snippet.strip(),
                        tags=["openmp", "parallel"],
                    )
                )

        return regions

    # ---------------------------------------------------------------------
    # Heuristics / summary helpers
    # ---------------------------------------------------------------------

    def _infer_language(self, source_path: Path) -> str:
        """
        Infer the programming language from the file extension.

        This is intentionally simple because the current benchmarks are C-based.
        """
        suffix = source_path.suffix.lower()

        if suffix == ".c":
            return "c"
        if suffix in {".cc", ".cpp", ".cxx"}:
            return "cpp"
        if suffix == ".h":
            return "c_header"
        if suffix == ".py":
            return "python"
        return "unknown"

    def _infer_memory_intensity(self, source_text: str) -> bool:
        """
        Heuristically infer whether the code appears memory-intensive.

        This uses practical keyword checks only. It does not attempt true
        performance modeling.
        """
        memory_keywords = [
            "malloc",
            "calloc",
            "free",
            "memcpy",
            "memset",
            "double *",
            "float *",
            "int *",
            "[i]",
            "[j]",
            "[k]",
        ]

        score = 0
        lowered = source_text.lower()

        for keyword in memory_keywords:
            if keyword.lower() in lowered:
                score += 1

        # A low threshold is fine here because this is just a hint for the
        # advisor and not a hard classification.
        return score >= 3

    def _build_summary_text(
        self,
        file_name: str,
        language: str,
        function_count: int,
        loop_count: int,
        has_mpi: bool,
        has_openmp: bool,
        appears_memory_intensive: bool,
    ) -> str:
        """
        Construct a compact human-readable summary for use in prompts/logs.
        """
        characteristics = []

        if has_mpi:
            characteristics.append("contains MPI communication")
        if has_openmp:
            characteristics.append("contains OpenMP pragmas")
        if appears_memory_intensive:
            characteristics.append("appears memory intensive")
        if loop_count > 0:
            characteristics.append("contains loop-based computation")

        if not characteristics:
            characteristics.append("contains general sequential code")

        characteristics_text = ", ".join(characteristics)

        return (
            f"Static analysis of {file_name} ({language}): "
            f"{function_count} function(s), {loop_count} loop(s); "
            f"{characteristics_text}."
        )

    def _merge_regions(
        self,
        function_regions: List[CodeRegion],
        loop_regions: List[CodeRegion],
        mpi_regions: List[CodeRegion],
        omp_regions: List[CodeRegion],
    ) -> List[CodeRegion]:
        """
        Merge all region types into a single ordered list.

        Regions are sorted by source position so downstream consumers can inspect
        them in a natural reading order.
        """
        all_regions = function_regions + loop_regions + mpi_regions + omp_regions
        all_regions.sort(key=lambda region: (region.start_line, region.end_line, region.region_type))
        return all_regions

    # ---------------------------------------------------------------------
    # Text / brace utilities
    # ---------------------------------------------------------------------

    def _find_matching_brace(self, text: str, open_brace_idx: int) -> Optional[int]:
        """
        Find the matching closing brace for the brace at open_brace_idx.

        Returns None if no matching brace is found.
        """
        if open_brace_idx < 0 or open_brace_idx >= len(text) or text[open_brace_idx] != "{":
            return None

        depth = 0
        for idx in range(open_brace_idx, len(text)):
            if text[idx] == "{":
                depth += 1
            elif text[idx] == "}":
                depth -= 1
                if depth == 0:
                    return idx

        return None

    def _char_span_to_line_span(self, text: str, start_idx: int, end_idx: int) -> Tuple[int, int]:
        """
        Convert a character span into 1-indexed line numbers.
        """
        start_line = text.count("\n", 0, start_idx) + 1
        end_line = text.count("\n", 0, end_idx) + 1
        return start_line, end_line