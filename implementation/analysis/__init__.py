"""
analysis package

This package contains the analysis stage of the optimization pipeline.

The analysis stage is responsible for:
1. Inspecting source code (static analysis)
2. Parsing profiling or benchmark outputs
3. Extracting normalized telemetry signals
4. Packaging everything into a structured AnalysisBundle

This bundle is then passed to the MoE advisor for optimization reasoning.
"""

# Core data structures used across the pipeline
from .analysis_bundle import (
    AnalysisBundle,
    CodeAnalysisSummary,
    ProfilingSummary,
    TelemetrySummary,
    CodeRegion,
)

# Static source code analyzer
from .code_analyzer import CodeAnalyzer

# Profiling / runtime output parser
from .profiler_parser import ProfilerParser

# Telemetry feature extractor (feeds the router)
from .telemetry_extractor import TelemetryExtractor


# Define what gets exported when users do:
#     from implementation.analysis import *
__all__ = [
    "AnalysisBundle",
    "CodeAnalysisSummary",
    "ProfilingSummary",
    "TelemetrySummary",
    "CodeRegion",
    "CodeAnalyzer",
    "ProfilerParser",
    "TelemetryExtractor",
]