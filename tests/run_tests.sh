#!/usr/bin/env bash

# run_tests.sh
#
# Test runner for temp_moe with optional benchmark post-evaluation.
#
# Supports:
#   - running all tests
#   - running only analysis tests
#   - running only benchmark tests
#   - running only advisor/MoE tests
#   - running one specific test file
#   - running one specific test node/id
#   - generating compare_c_versions + rubric outputs for benchmark artifacts
#
# Examples:
#   bash tests/run_tests.sh all
#   bash tests/run_tests.sh analysis
#   bash tests/run_tests.sh benchmarks
#   bash tests/run_tests.sh advisor
#   bash tests/run_tests.sh file test_code_analyzer.py
#   bash tests/run_tests.sh node tests/test_code_analyzer.py::test_mem_saxpy_basic_metadata
#   bash tests/run_tests.sh benchmark-eval
#   bash tests/run_tests.sh benchmarks --with-eval
#
# Optional:
#   VERBOSE=1 bash tests/run_tests.sh analysis
#   FAILFAST=1 bash tests/run_tests.sh all
#   RUBRIC_EVALUATOR_SCRIPT=tests/rubric_evaluator.py bash tests/run_tests.sh benchmark-eval

set -euo pipefail

# Resolve repository root from the location of this script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"

# Use python -m pytest for consistent environment behavior.
PYTEST_CMD=("${PYTHON_BIN}" -m pytest)

# Optional flags for easier debugging.
if [[ "${VERBOSE:-0}" == "1" ]]; then
    PYTEST_CMD+=(-vv)
else
    PYTEST_CMD+=(-v)
fi

if [[ "${FAILFAST:-0}" == "1" ]]; then
    PYTEST_CMD+=(-x)
fi

# Optional external evaluator.
RUBRIC_EVALUATOR_SCRIPT="${RUBRIC_EVALUATOR_SCRIPT:-}"

# Small helper for clean output.
print_header() {
    echo
    echo "============================================================"
    echo "$1"
    echo "============================================================"
}

# Keep file groups centralized so they are easy to maintain.
ALL_TESTS=(
    tests/test_router.py
    tests/test_experts_mock_llm.py
    tests/test_advisor_end_to_end.py
    tests/test_bench_mem_saxpy.py
    tests/test_bench_mpi_pingpong.py
    tests/test_bench_omp_imbalance.py
    tests/test_analysis_bundle.py
    tests/test_code_analyzer.py
    tests/test_profiler_parser.py
    tests/test_telemetry_extractor.py
    tests/test_llm_backends.py
    tests/test_advisor_hf_integration.py
)

ANALYSIS_TESTS=(
    tests/test_analysis_bundle.py
    tests/test_code_analyzer.py
    tests/test_profiler_parser.py
    tests/test_telemetry_extractor.py
)

BENCHMARK_TESTS=(
    tests/test_bench_mem_saxpy.py
    tests/test_bench_mpi_pingpong.py
    tests/test_bench_omp_imbalance.py
)

ADVISOR_TESTS=(
    tests/test_router.py
    tests/test_experts_mock_llm.py
    tests/test_advisor_end_to_end.py
)

BENCHMARK_NAMES=(
    mem_saxpy
    mpi_pingpong
    omp_imbalance
)

usage() {
    cat <<EOF
Usage:
  bash tests/run_tests.sh <mode> [extra] [--with-eval]

Modes:
  all              Run all tests
  analysis         Run only analysis-related tests
  benchmarks       Run only benchmark tests
  advisor          Run only advisor / MoE tests
  benchmark-eval   Run compare_c_versions + rubric generation for benchmark artifacts
  file <name>      Run one specific test file from tests/
  node <id>        Run one exact pytest node id
  list             Show the grouped test files
  help             Show this help message

Examples:
  bash tests/run_tests.sh all
  bash tests/run_tests.sh analysis
  bash tests/run_tests.sh benchmarks
  bash tests/run_tests.sh benchmarks --with-eval
  bash tests/run_tests.sh advisor
  bash tests/run_tests.sh file test_code_analyzer.py
  bash tests/run_tests.sh node tests/test_code_analyzer.py::test_mem_saxpy_basic_metadata
  bash tests/run_tests.sh benchmark-eval

Environment variables:
  VERBOSE=1                 Use pytest -vv
  FAILFAST=1                Stop on first failure
  PYTHON_BIN=python3        Python executable override
  RUBRIC_EVALUATOR_SCRIPT   Optional path to an external evaluator script
                            that accepts:
                              --benchmark <name>
                              --comparison-json <path>
                              --output-json <path>
EOF
}

list_groups() {
    print_header "All tests"
    printf '%s\n' "${ALL_TESTS[@]}"

    print_header "Analysis tests"
    printf '%s\n' "${ANALYSIS_TESTS[@]}"

    print_header "Benchmark tests"
    printf '%s\n' "${BENCHMARK_TESTS[@]}"

    print_header "Advisor / MoE tests"
    printf '%s\n' "${ADVISOR_TESTS[@]}"
}

run_group() {
    local group_name="$1"
    shift

    print_header "Running ${group_name}"
    "${PYTEST_CMD[@]}" "$@"
}

benchmark_compare_args() {
    local bench_name="$1"

    case "${bench_name}" in
        mem_saxpy)
            echo "cc|-O3||20|20|2|"
            ;;
        mpi_pingpong)
            echo "mpicc|-O3||20|10|2|mpirun -np 2"
            ;;
        omp_imbalance)
            echo "cc|-O3 -Xpreprocessor -fopenmp -I$(brew --prefix libomp)/include -L$(brew --prefix libomp)/lib -lomp||20|10|2|"
            ;;
        *)
            echo "cc|-O3||20|20|2|"
            ;;
    esac
}

run_inline_rubric_evaluator() {
    local bench_name="$1"
    local comparison_json="$2"
    local rubric_json="$3"

    "${PYTHON_BIN}" - "${bench_name}" "${comparison_json}" "${rubric_json}" <<'PY'
import json
import sys
from pathlib import Path

bench_name = sys.argv[1]
comparison_path = Path(sys.argv[2])
rubric_path = Path(sys.argv[3])

data = json.loads(comparison_path.read_text(encoding="utf-8"))

optimized_build = data.get("optimized_build") or {}
optimized_run = data.get("optimized_run") or {}
output_cmp = data.get("output_comparison") or {}
diff_metrics = data.get("diff_metrics") or {}

evaluation_status = data.get("evaluation_status")
environment_failure = bool(data.get("environment_failure", False))
environment_reason = data.get("environment_reason")

compile_pass = bool(optimized_build.get("compile_success", False))
run_pass = bool(optimized_run.get("run_success", False))
timeout_free = int(optimized_run.get("timeout_count", 0) or 0) == 0
crash_free = int(optimized_run.get("crash_count", 0) or 0) == 0
exit_code_match = bool(output_cmp.get("exit_code_match", False))
stderr_match = bool(output_cmp.get("stderr_match", False))

correctness_pass = bool(output_cmp.get("correctness_pass", False))
stdout_match = bool(output_cmp.get("stdout_match", False))
within_tolerance = bool(output_cmp.get("within_tolerance", False))
acceptable_drift = bool(output_cmp.get("acceptable_drift", False))

percent_file_changed = float(diff_metrics.get("percent_file_changed", 100.0))
percent_improvement = data.get("percent_improvement")
speedup = data.get("speedup")
likely_significant = data.get("likely_significant")
significance_reason = data.get("significance_reason")

hard_gates = {
    "compile_pass": compile_pass,
    "run_pass": run_pass,
    "timeout_free": timeout_free,
    "crash_free": crash_free,
    "exit_code_match": exit_code_match,
    "stderr_match": stderr_match,
}

def telemetry_pattern_alignment(bench_name: str, pattern_hint: str = "") -> bool:
    # In shell fallback mode we usually do not have generator_result pattern text,
    # so use benchmark-family defaults as a conservative approximation.
    pattern_hint = (pattern_hint or "").lower()

    if bench_name == "mem_saxpy":
        if not pattern_hint:
            return True
        return any(k in pattern_hint for k in ["smaller data", "precision", "type", "memory", "locality", "vector"])

    if bench_name == "omp_imbalance":
        if not pattern_hint:
            return True
        return any(k in pattern_hint for k in ["schedule", "load", "imbalance", "parallel", "openmp", "chunk"])

    if bench_name == "mpi_pingpong":
        if not pattern_hint:
            return True
        return any(k in pattern_hint for k in ["mpi", "communication", "async", "message", "overlap"])

    return True

candidate_faithful = compile_pass and run_pass and percent_file_changed <= 70.0
telemetry_aligned = telemetry_pattern_alignment(bench_name)

if environment_failure or evaluation_status == "environment_failure":
    rubric = {
        "benchmark": bench_name,
        "evaluation_status": "environment_failure",
        "hard_gates": hard_gates,
        "numeric_correctness": {
            "correctness_pass": correctness_pass,
            "stdout_match": stdout_match,
            "within_tolerance": within_tolerance,
            "acceptable_drift": acceptable_drift,
            "relative_error": output_cmp.get("relative_error"),
            "numeric_tolerance": output_cmp.get("numeric_tolerance"),
            "drift_threshold": output_cmp.get("drift_threshold"),
            "comparison_mode": output_cmp.get("comparison_mode"),
            "original_numeric_signal": output_cmp.get("original_numeric_signal"),
            "optimized_numeric_signal": output_cmp.get("optimized_numeric_signal"),
        },
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
            "reason": environment_reason or "Baseline failed to compile or run, so the optimization cannot be judged reliably.",
        },
        "diagnostics": {
            "candidate_faithful": candidate_faithful,
            "telemetry_pattern_aligned": telemetry_aligned,
        },
    }
    rubric_path.write_text(json.dumps(rubric, indent=2), encoding="utf-8")
    raise SystemExit(0)

correctness_safety = 0
correctness_safety += 15 if compile_pass else 0

if correctness_pass:
    correctness_safety += 15 if stdout_match else 12
elif acceptable_drift:
    correctness_safety += 10

if percent_file_changed <= 35.0:
    correctness_safety += 10
elif percent_file_changed <= 70.0:
    correctness_safety += 5

optimization_faithfulness = 0
if percent_file_changed <= 35.0:
    optimization_faithfulness += 25
elif percent_file_changed <= 70.0:
    optimization_faithfulness += 18
elif percent_file_changed <= 90.0:
    optimization_faithfulness += 10
else:
    optimization_faithfulness += 4

hpc_applicability = 0
if run_pass and timeout_free and crash_free:
    hpc_applicability += 10

if percent_improvement is not None:
    if percent_improvement >= 5.0:
        hpc_applicability += 10
    elif percent_improvement > 0.0:
        hpc_applicability += 5

if likely_significant is True:
    hpc_applicability += 5

if bench_name == "mem_saxpy":
    hpc_applicability += 5
elif bench_name == "mpi_pingpong":
    hpc_applicability += 5
elif bench_name == "omp_imbalance":
    hpc_applicability += 5

repairability = 0
if not correctness_pass:
    if acceptable_drift:
        repairability += 2
    elif compile_pass:
        repairability += 5
if percent_file_changed <= 70.0:
    repairability += 5

total = correctness_safety + optimization_faithfulness + hpc_applicability + repairability

if not compile_pass:
    action = "repair_once"
    reason = "Optimized candidate failed to compile while baseline evaluation was valid."

elif not run_pass or not timeout_free or not crash_free:
    action = "repair_once"
    reason = "Optimized candidate compiled but failed during execution, timed out, or crashed."

elif correctness_pass and total >= 75:
    action = "accept"
    reason = "Passed deterministic checks with strong rubric score."

elif acceptable_drift and percent_improvement is not None and percent_improvement >= 5.0:
    action = "accept_with_warning"
    reason = "Accepted with warning: strong performance gain with acceptable HPC-style numerical drift."

elif (
    candidate_faithful
    and correctness_pass
    and percent_improvement is not None
    and percent_improvement <= 5.0
    and not telemetry_aligned
):
    action = "reconsider_advisor"
    reason = (
        "Generated code appears faithful and correct, but the selected optimization "
        "strategy seems weakly aligned with the apparent bottleneck and produced limited gain."
    )

elif (
    candidate_faithful
    and (correctness_pass or acceptable_drift)
    and percent_improvement is not None
    and percent_improvement <= 0.0
):
    action = "reconsider_advisor"
    reason = (
        "Generated code appears faithful and executable, but performance did not improve. "
        "This suggests the selected optimization strategy may be mismatched."
    )

elif not correctness_pass and not acceptable_drift:
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

rubric = {
    "benchmark": bench_name,
    "evaluation_status": "evaluated",
    "hard_gates": hard_gates,
    "numeric_correctness": {
        "correctness_pass": correctness_pass,
        "stdout_match": stdout_match,
        "within_tolerance": within_tolerance,
        "acceptable_drift": acceptable_drift,
        "relative_error": output_cmp.get("relative_error"),
        "numeric_tolerance": output_cmp.get("numeric_tolerance"),
        "drift_threshold": output_cmp.get("drift_threshold"),
        "comparison_mode": output_cmp.get("comparison_mode"),
        "original_numeric_signal": output_cmp.get("original_numeric_signal"),
        "optimized_numeric_signal": output_cmp.get("optimized_numeric_signal"),
    },
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
        "candidate_faithful": candidate_faithful,
        "telemetry_pattern_aligned": telemetry_aligned,
    },
}
rubric_path.write_text(json.dumps(rubric, indent=2), encoding="utf-8")
PY
}

evaluate_benchmark_artifact() {
    local bench_name="$1"
    local original_source="${REPO_ROOT}/benchmarks/${bench_name}/main.c"
    local optimized_source="${REPO_ROOT}/generated_optimizations/${bench_name}/optimized_main.c"
    local output_dir="${REPO_ROOT}/generated_optimizations/${bench_name}"
    local comparison_json="${output_dir}/tester_result.json"
    local rubric_json="${output_dir}/rubric_result.json"

    if [[ ! -f "${original_source}" ]]; then
        echo "[skip] ${bench_name}: missing original source ${original_source}"
        return 0
    fi

    if [[ ! -f "${optimized_source}" ]]; then
        echo "[skip] ${bench_name}: missing optimized source ${optimized_source}"
        return 0
    fi

    IFS='|' read -r compiler cflags program_args timeout_seconds trials warmup_trials run_prefix <<<"$(benchmark_compare_args "${bench_name}")"

    print_header "Comparing benchmark versions for ${bench_name}"
    "${PYTHON_BIN}" tests/compare_c_versions.py \
        --original "${original_source}" \
        --optimized "${optimized_source}" \
        --compiler "${compiler}" \
        --cflags="${cflags}" \
        --program-args="${program_args}" \
        --timeout-seconds "${timeout_seconds}" \
        --trials "${trials}" \
        --warmup-trials "${warmup_trials}" \
        --run-prefix="${run_prefix}" \
        --output-json "${comparison_json}"

    print_header "Generating rubric for ${bench_name}"
    if [[ -n "${RUBRIC_EVALUATOR_SCRIPT}" ]]; then
        if [[ ! -f "${RUBRIC_EVALUATOR_SCRIPT}" ]]; then
            echo "Error: RUBRIC_EVALUATOR_SCRIPT not found: ${RUBRIC_EVALUATOR_SCRIPT}"
            exit 1
        fi

        "${PYTHON_BIN}" "${RUBRIC_EVALUATOR_SCRIPT}" \
            --benchmark "${bench_name}" \
            --comparison-json "${comparison_json}" \
            --output-json "${rubric_json}"
    else
        run_inline_rubric_evaluator "${bench_name}" "${comparison_json}" "${rubric_json}"
    fi

    echo "[done] ${bench_name}: comparison=${comparison_json} rubric=${rubric_json}"
}

run_benchmark_evaluation() {
    print_header "Running compare_c_versions + rubric evaluation for generated benchmark artifacts"
    local bench_name
    for bench_name in "${BENCHMARK_NAMES[@]}"; do
        evaluate_benchmark_artifact "${bench_name}"
    done
}

MODE="${1:-help}"
if [[ "$#" -gt 0 ]]; then
    shift
fi

WITH_EVAL=0
REMAINING_ARGS=()

for arg in "$@"; do
    if [[ "${arg}" == "--with-eval" ]]; then
        WITH_EVAL=1
    else
        REMAINING_ARGS+=("${arg}")
    fi
done

case "${MODE}" in
    all)
        run_group "all tests" "${ALL_TESTS[@]}"
        if [[ "${WITH_EVAL}" == "1" ]]; then
            run_benchmark_evaluation
        fi
        ;;
    analysis)
        run_group "analysis tests" "${ANALYSIS_TESTS[@]}"
        ;;
    benchmarks)
        run_group "benchmark tests" "${BENCHMARK_TESTS[@]}"
        if [[ "${WITH_EVAL}" == "1" ]]; then
            run_benchmark_evaluation
        fi
        ;;
    advisor)
        run_group "advisor / MoE tests" "${ADVISOR_TESTS[@]}"
        ;;
    benchmark-eval)
        run_benchmark_evaluation
        ;;
    file)
        if [[ ${#REMAINING_ARGS[@]} -lt 1 ]]; then
            echo "Error: file mode requires a filename, e.g. test_code_analyzer.py"
            exit 1
        fi

        TEST_FILE="tests/${REMAINING_ARGS[0]}"
        if [[ ! -f "${TEST_FILE}" ]]; then
            echo "Error: test file not found: ${TEST_FILE}"
            exit 1
        fi

        run_group "single test file: ${TEST_FILE}" "${TEST_FILE}"
        ;;
    node)
        if [[ ${#REMAINING_ARGS[@]} -lt 1 ]]; then
            echo "Error: node mode requires a pytest node id."
            exit 1
        fi

        run_group "single pytest node: ${REMAINING_ARGS[0]}" "${REMAINING_ARGS[0]}"
        ;;
    list)
        list_groups
        ;;
    help|-h|--help)
        usage
        ;;
    *)
        echo "Error: unknown mode '${MODE}'"
        echo
        usage
        exit 1
        ;;
esac