from __future__ import annotations

from typing import Any, Dict, Optional


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_environment_failure(comparison_report: Any) -> bool:
    original_build = _get(comparison_report, "original_build")
    optimized_build = _get(comparison_report, "optimized_build")

    original_compile = bool(_get(original_build, "compile_success", False))
    optimized_compile = bool(_get(optimized_build, "compile_success", False))

    # If the baseline/original does not compile, the benchmark cannot be used
    # to judge the generated optimization.
    return (not original_compile) and (not optimized_compile or True)


def _build_hard_gates(comparison_report: Any) -> Dict[str, bool]:
    original_build = _get(comparison_report, "original_build")
    optimized_build = _get(comparison_report, "optimized_build")
    original_run = _get(comparison_report, "original_run")
    optimized_run = _get(comparison_report, "optimized_run")
    output_cmp = _get(comparison_report, "output_comparison")

    return {
        "original_compile_pass": bool(_get(original_build, "compile_success", False)),
        "compile_pass": bool(_get(optimized_build, "compile_success", False)),
        "original_run_pass": bool(_get(original_run, "run_success", False)),
        "run_pass": bool(_get(optimized_run, "run_success", False)),
        "timeout_free": int(_get(optimized_run, "timeout_count", 0) or 0) == 0,
        "crash_free": int(_get(optimized_run, "crash_count", 0) or 0) == 0,
        "exit_code_match": bool(_get(output_cmp, "exit_code_match", False)),
        "stderr_match": bool(_get(output_cmp, "stderr_match", False)),
    }


def _compute_numeric_correctness(generator_result: Any, comparison_report: Any) -> Dict[str, Any]:
    output_cmp = _get(comparison_report, "output_comparison")
    if output_cmp is None:
        return {
            "correctness_pass": False,
            "stdout_match": False,
            "within_tolerance": False,
            "relative_error": None,
            "numeric_tolerance": None,
            "comparison_mode": None,
            "original_numeric_signal": None,
            "optimized_numeric_signal": None,
            "acceptable_drift": False,
            "drift_reason": "No output comparison available.",
        }

    correctness_pass = bool(_get(output_cmp, "correctness_pass", False))
    stdout_match = bool(_get(output_cmp, "stdout_match", False))
    within_tolerance = bool(_get(output_cmp, "within_tolerance", False))
    relative_error = _to_float(_get(output_cmp, "relative_error"))
    numeric_tolerance = _to_float(_get(output_cmp, "numeric_tolerance"))
    comparison_mode = _get(output_cmp, "comparison_mode")
    original_numeric_signal = _to_float(_get(output_cmp, "original_numeric_signal"))
    optimized_numeric_signal = _to_float(_get(output_cmp, "optimized_numeric_signal"))

    benchmark_hint = (
        str(_get(comparison_report, "optimized_source", "")).lower()
        + " "
        + str(_get(generator_result, "selected_candidate_pattern", "")).lower()
    )

    # HPC-aware "accept with warning" window for parallel / floating-point drift.
    acceptable_drift = False
    drift_reason = "No acceptable drift detected."

    if not correctness_pass and relative_error is not None:
        relaxed_threshold = None

        if any(token in benchmark_hint for token in ["omp", "openmp", "reduction", "schedule", "parallel"]):
            relaxed_threshold = 1e-4
        elif any(token in benchmark_hint for token in ["saxpy", "float", "precision", "smaller data"]):
            relaxed_threshold = 1e-5
        elif any(token in benchmark_hint for token in ["mpi", "pingpong", "message", "communication"]):
            relaxed_threshold = 0.0
        else:
            relaxed_threshold = 1e-5

        acceptable_drift = relative_error <= relaxed_threshold and relaxed_threshold > 0.0
        if acceptable_drift:
            drift_reason = (
                f"Relative error {relative_error:.6g} is within relaxed HPC drift threshold "
                f"{relaxed_threshold:.6g}."
            )
        else:
            drift_reason = (
                f"Relative error {relative_error:.6g} exceeds allowed threshold "
                f"{relaxed_threshold:.6g}."
            )

    return {
        "correctness_pass": correctness_pass,
        "stdout_match": stdout_match,
        "within_tolerance": within_tolerance,
        "relative_error": relative_error,
        "numeric_tolerance": numeric_tolerance,
        "comparison_mode": comparison_mode,
        "original_numeric_signal": original_numeric_signal,
        "optimized_numeric_signal": optimized_numeric_signal,
        "acceptable_drift": acceptable_drift,
        "drift_reason": drift_reason,
    }


def evaluate_with_rubric(generator_result: Any, comparison_report: Any) -> Dict[str, Any]:
    hard_gates = _build_hard_gates(comparison_report)
    numeric = _compute_numeric_correctness(generator_result, comparison_report)

    diff_metrics = _get(comparison_report, "diff_metrics")
    optimized_build = _get(comparison_report, "optimized_build")
    optimized_run = _get(comparison_report, "optimized_run")

    percent_file_changed = _to_float(_get(diff_metrics, "percent_file_changed")) or 100.0
    percent_improvement = _to_float(_get(comparison_report, "percent_improvement"))
    speedup = _to_float(_get(comparison_report, "speedup"))
    likely_significant = _get(comparison_report, "likely_significant")
    significance_reason = _get(comparison_report, "significance_reason")

    selected_pattern = str(_get(generator_result, "selected_candidate_pattern", "") or "")
    selected_target = str(_get(generator_result, "selected_candidate_target", "") or "")

    # ---------- Environment failure classification ----------
    if _is_environment_failure(comparison_report):
        return {
            "evaluation_status": "environment_failure",
            "hard_gates": hard_gates,
            "numeric_correctness": numeric,
            "rubric": {
                "correctness_safety": 0,
                "optimization_faithfulness": 0,
                "hpc_applicability": 0,
                "repairability": 0,
                "total": 0,
            },
            "performance": {
                "speedup": speedup,
                "percent_improvement": percent_improvement,
                "likely_significant": likely_significant,
                "significance_reason": significance_reason,
            },
            "decision": {
                "action": "environment_failure",
                "reason": "Original baseline failed to compile or run, so the optimization cannot be judged reliably.",
            },
            "diagnostics": {
                "original_build_stderr": _get(_get(comparison_report, "original_build"), "stderr"),
                "optimized_build_stderr": _get(optimized_build, "stderr"),
            },
        }

    # ---------- Hard fail that is genuinely attributable to optimized candidate ----------
    if not hard_gates["compile_pass"]:
        return {
            "evaluation_status": "evaluated",
            "hard_gates": hard_gates,
            "numeric_correctness": numeric,
            "rubric": {
                "correctness_safety": 0,
                "optimization_faithfulness": 0,
                "hpc_applicability": 0,
                "repairability": 10 if hard_gates["original_compile_pass"] else 0,
                "total": 10 if hard_gates["original_compile_pass"] else 0,
            },
            "performance": {
                "speedup": speedup,
                "percent_improvement": percent_improvement,
                "likely_significant": likely_significant,
                "significance_reason": significance_reason,
            },
            "decision": {
                "action": "repair_once" if hard_gates["original_compile_pass"] else "reject",
                "reason": "Optimized candidate failed to compile while baseline compiled successfully."
                if hard_gates["original_compile_pass"]
                else "Compilation failed and baseline status is insufficient for safe repair.",
            },
            "diagnostics": {
                "optimized_build_stderr": _get(optimized_build, "stderr"),
            },
        }

    if not hard_gates["run_pass"] or not hard_gates["timeout_free"] or not hard_gates["crash_free"]:
        return {
            "evaluation_status": "evaluated",
            "hard_gates": hard_gates,
            "numeric_correctness": numeric,
            "rubric": {
                "correctness_safety": 15,
                "optimization_faithfulness": 0,
                "hpc_applicability": 0,
                "repairability": 10,
                "total": 25,
            },
            "performance": {
                "speedup": speedup,
                "percent_improvement": percent_improvement,
                "likely_significant": likely_significant,
                "significance_reason": significance_reason,
            },
            "decision": {
                "action": "repair_once",
                "reason": "Optimized candidate compiled but failed during execution, timed out, or crashed.",
            },
            "diagnostics": {
                "optimized_run_stderr": _get(optimized_run, "representative_stderr"),
            },
        }

    # ---------- Correctness and safety ----------
    correctness_safety = 0
    correctness_safety += 15 if hard_gates["compile_pass"] else 0

    if numeric["correctness_pass"]:
        correctness_safety += 15 if numeric["stdout_match"] else 12
    elif numeric["acceptable_drift"]:
        correctness_safety += 10
    else:
        correctness_safety += 0

    if percent_file_changed <= 35.0:
        correctness_safety += 10
    elif percent_file_changed <= 70.0:
        correctness_safety += 5

    # ---------- Optimization faithfulness ----------
    optimization_faithfulness = 0
    if selected_pattern:
        optimization_faithfulness += 10
    if selected_target:
        optimization_faithfulness += 10
    if percent_file_changed <= 35.0:
        optimization_faithfulness += 5
    elif percent_file_changed <= 70.0:
        optimization_faithfulness += 2

    # ---------- HPC applicability ----------
    hpc_applicability = 0
    pattern_l = selected_pattern.lower()
    optimized_source = str(_get(comparison_report, "optimized_source", "")).lower()

    if any(k in optimized_source for k in ["mem_saxpy", "saxpy"]):
        if any(k in pattern_l for k in ["smaller data", "precision", "type", "memory", "locality", "vector"]):
            hpc_applicability += 10
        else:
            hpc_applicability += 5
    elif any(k in optimized_source for k in ["omp", "imbalance", "openmp"]):
        if any(k in pattern_l for k in ["schedule", "load", "imbalance", "parallel", "openmp", "chunk"]):
            hpc_applicability += 10
        else:
            hpc_applicability += 5
    elif any(k in optimized_source for k in ["mpi", "pingpong"]):
        if any(k in pattern_l for k in ["mpi", "communication", "async", "message", "overlap"]):
            hpc_applicability += 10
        else:
            hpc_applicability += 5
    else:
        hpc_applicability += 5

    if hard_gates["run_pass"] and hard_gates["timeout_free"] and hard_gates["crash_free"]:
        hpc_applicability += 10

    if percent_improvement is not None:
        if percent_improvement >= 5.0:
            hpc_applicability += 10
        elif percent_improvement > 0.0:
            hpc_applicability += 5

    if likely_significant is True:
        hpc_applicability += 5

    # ---------- Repairability ----------
    repairability = 0
    if not numeric["correctness_pass"]:
        if numeric["acceptable_drift"]:
            repairability += 2
        else:
            repairability += 5
    if percent_file_changed <= 70.0:
        repairability += 5

    total = correctness_safety + optimization_faithfulness + hpc_applicability + repairability

    # ---------- Decision ----------
    if numeric["correctness_pass"] and total >= 75:
        action = "accept"
        reason = "Passed deterministic checks with strong rubric score."
    elif numeric["acceptable_drift"] and percent_improvement is not None and percent_improvement >= 5.0:
        action = "accept_with_warning"
        reason = (
            "Accepted with warning: strong performance gain with acceptable HPC-style numerical drift."
        )
    elif not numeric["correctness_pass"] and not numeric["acceptable_drift"]:
        if total >= 45:
            action = "repair_once"
            reason = "Correctness deviation is too large for acceptance, but one repair attempt is justified."
        else:
            action = "reject"
            reason = "Correctness deviation is too large and the overall result is too weak for repair."
    elif total >= 75:
        action = "accept"
        reason = "Accepted based on rubric score."
    elif total >= 45:
        action = "repair_once"
        reason = "Moderate result; one repair attempt justified."
    else:
        action = "reject"
        reason = "Low rubric score or non-repairable result."

    return {
        "evaluation_status": "evaluated",
        "hard_gates": hard_gates,
        "numeric_correctness": numeric,
        "rubric": {
            "correctness_safety": correctness_safety,
            "optimization_faithfulness": optimization_faithfulness,
            "hpc_applicability": hpc_applicability,
            "repairability": repairability,
            "total": total,
        },
        "performance": {
            "speedup": speedup,
            "percent_improvement": percent_improvement,
            "likely_significant": likely_significant,
            "significance_reason": significance_reason,
        },
        "decision": {
            "action": action,
            "reason": reason,
        },
        "diagnostics": {
            "selected_candidate_pattern": selected_pattern,
            "selected_candidate_target": selected_target,
            "percent_file_changed": percent_file_changed,
            "optimized_build_stderr": _get(optimized_build, "stderr"),
            "optimized_run_stderr": _get(optimized_run, "representative_stderr"),
        },
    }