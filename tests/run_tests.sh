#!/usr/bin/env bash

# run_tests.sh
#
# Test runner for temp_moe.
#
# Supports:
#   - running all tests
#   - running only analysis tests
#   - running only benchmark tests
#   - running only advisor/MoE tests
#   - running one specific test file
#   - running one specific test node/id
#
# Examples:
#   bash tests/run_tests.sh all
#   bash tests/run_tests.sh analysis
#   bash tests/run_tests.sh benchmarks
#   bash tests/run_tests.sh advisor
#   bash tests/run_tests.sh file test_code_analyzer.py
#   bash tests/run_tests.sh node tests/test_code_analyzer.py::test_mem_saxpy_basic_metadata
#   bash tests/run_tests.sh list
#
# Optional:
#   VERBOSE=1 bash tests/run_tests.sh analysis
#   FAILFAST=1 bash tests/run_tests.sh all

set -euo pipefail

# Resolve repository root from the location of this script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

# Use python -m pytest for consistent environment behavior.
PYTEST_CMD=(python -m pytest)

# Optional flags for easier debugging.
if [[ "${VERBOSE:-0}" == "1" ]]; then
    PYTEST_CMD+=(-vv)
else
    PYTEST_CMD+=(-v)
fi

if [[ "${FAILFAST:-0}" == "1" ]]; then
    PYTEST_CMD+=(-x)
fi

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

usage() {
    cat <<EOF
Usage:
  bash tests/run_tests.sh <mode> [extra]

Modes:
  all           Run all tests
  analysis      Run only analysis-related tests
  benchmarks    Run only benchmark tests
  advisor       Run only advisor / MoE tests
  file <name>   Run one specific test file from tests/
  node <id>     Run one exact pytest node id
  list          Show the grouped test files
  help          Show this help message

Examples:
  bash tests/run_tests.sh all
  bash tests/run_tests.sh analysis
  bash tests/run_tests.sh benchmarks
  bash tests/run_tests.sh advisor
  bash tests/run_tests.sh file test_code_analyzer.py
  bash tests/run_tests.sh node tests/test_code_analyzer.py::test_mem_saxpy_basic_metadata

Environment variables:
  VERBOSE=1   Use pytest -vv
  FAILFAST=1  Stop on first failure
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

MODE="${1:-help}"

case "${MODE}" in
    all)
        run_group "all tests" "${ALL_TESTS[@]}"
        ;;
    analysis)
        run_group "analysis tests" "${ANALYSIS_TESTS[@]}"
        ;;
    benchmarks)
        run_group "benchmark tests" "${BENCHMARK_TESTS[@]}"
        ;;
    advisor)
        run_group "advisor / MoE tests" "${ADVISOR_TESTS[@]}"
        ;;
    file)
        if [[ $# -lt 2 ]]; then
            echo "Error: file mode requires a filename, e.g. test_code_analyzer.py"
            exit 1
        fi

        TEST_FILE="tests/$2"
        if [[ ! -f "${TEST_FILE}" ]]; then
            echo "Error: test file not found: ${TEST_FILE}"
            exit 1
        fi

        run_group "single test file: ${TEST_FILE}" "${TEST_FILE}"
        ;;
    node)
        if [[ $# -lt 2 ]]; then
            echo "Error: node mode requires a pytest node id."
            exit 1
        fi

        run_group "single pytest node: $2" "$2"
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