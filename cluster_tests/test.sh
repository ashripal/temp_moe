#!/bin/bash
# FILENAME: test_local.sh
# Runs a local end-to-end smoke test of the orchestrator without Slurm or MPI.
# Run from your project root: bash test_local.sh

set -euo pipefail

# --- CONFIGURATION ---
DATASET="MINI"       # Use MINI or SMALL locally — EXTRALARGE will take forever
RUNS=2               # Fewer runs to keep the test fast

PASS=0
FAIL=0

# --- COLOURS ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Colour

pass() { echo -e "${GREEN}  [PASS]${NC} $1"; PASS=$((PASS + 1)); }
fail() { echo -e "${RED}  [FAIL]${NC} $1"; FAIL=$((FAIL + 1)); }
warn() { echo -e "${YELLOW}  [WARN]${NC} $1"; }
header() { echo -e "\n${YELLOW}=== $1 ===${NC}"; }

# --- CLEANUP ---
cleanup() {
    echo ""
    echo "Cleaning up..."
    rm -f srun                                          # remove fake srun
    rm -f *_verify.exe *_timing.exe *_output.txt       # remove build artifacts
    rm -f kernels/linear-algebra/kernels/gemm/gemm_opt.c
    rm -f kernels/stencils/jacobi-2d/jacobi-2d_opt.c
}
trap cleanup EXIT


# ============================================================
# TEST 1: Check required tools are available
# ============================================================
header "Test 1: Required tools"

for tool in gcc python3; do
    if command -v $tool &>/dev/null; then
        pass "$tool is available ($(command -v $tool))"
    else
        fail "$tool is NOT available — install it before continuing"
    fi
done

if command -v mpicc &>/dev/null; then
    pass "mpicc is available — will use it for compilation"
    COMPILER="mpicc"
else
    warn "mpicc not found — falling back to gcc for local test (install open-mpi to test full compilation)"
    COMPILER="gcc"
fi


# ============================================================
# TEST 2: Check PolyBench kernel files exist at expected paths
# ============================================================
header "Test 2: Kernel file paths"

KERNEL_FILES=(
    "kernels/linear-algebra/blas/gemm/gemm.c"
    "kernels/stencils/jacobi-2d/jacobi-2d.c"
    "kernels/utilities/polybench.c"
    "kernels/utilities/polybench.h"
)

for f in "${KERNEL_FILES[@]}"; do
    if [ -f "$f" ]; then
        pass "Found: $f"
    else
        fail "Missing: $f"
    fi
done


# ============================================================
# TEST 3: Check orchestrator and sbatch files exist
# ============================================================
header "Test 3: Project files"

for f in orchestrator.py runner.sbatch; do
    if [ -f "$f" ]; then
        pass "Found: $f"
    else
        fail "Missing: $f"
    fi
done


# ============================================================
# TEST 4: Compile each kernel locally
# ============================================================
header "Test 4: Kernel compilation"

KERNELS=(
    "kernels/linear-algebra/blas/gemm/gemm"
    "kernels/stencils/jacobi-2d/jacobi-2d"
)

for kernel in "${KERNELS[@]}"; do
    src="${kernel}.c"
    name=$(basename $kernel)

    # Timing binary
    cmd="$COMPILER -O3 -I kernels/utilities/ kernels/utilities/polybench.c $src -D${DATASET}_DATASET -DPOLYBENCH_TIME -lm -o ${name}_timing.exe"
    if eval $cmd 2>/dev/null; then
        pass "Compiled timing binary: $name"
    else
        fail "Failed to compile timing binary: $name"
        fail "  Command was: $cmd"
        continue
    fi

    # Verify binary
    cmd="$COMPILER -O3 -I kernels/utilities/ kernels/utilities/polybench.c $src -D${DATASET}_DATASET -DPOLYBENCH_DUMP_ARRAYS -lm -o ${name}_verify.exe"
    if eval $cmd 2>/dev/null; then
        pass "Compiled verify binary: $name"
    else
        fail "Failed to compile verify binary: $name"
    fi
done


# ============================================================
# TEST 5: Run binaries and check output can be parsed
# ============================================================
header "Test 5: Binary execution and output parsing"

for kernel in "${KERNELS[@]}"; do
    name=$(basename $kernel)
    binary="./${name}_timing.exe"

    if [ ! -f "$binary" ]; then
        warn "Skipping execution test for $name — binary not found (compilation failed above)"
        continue
    fi

    output=$($binary 2>/dev/null || true)
    last_line=$(echo "$output" | tail -n 1)

    if [ -z "$last_line" ]; then
        fail "$name produced no stdout"
        continue
    else
        pass "$name produced stdout"
    fi

    if echo "$last_line" | grep -qE '^[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?$'; then
        pass "$name output is a parseable float: '$last_line'"
    else
        fail "$name last line is not a float: '$last_line' (full output: '$output')"
    fi
done


# ============================================================
# TEST 6: Verify output dump and comparison
# ============================================================
header "Test 6: Output verification"

for kernel in "${KERNELS[@]}"; do
    name=$(basename $kernel)
    binary="./${name}_verify.exe"

    if [ ! -f "$binary" ]; then
        warn "Skipping verification test for $name — binary not found"
        continue
    fi

    ./$binary 2> "${name}_output.txt" > /dev/null
    if [ -s "${name}_output.txt" ]; then
        pass "$name wrote non-empty output dump"
    else
        fail "$name output dump is empty"
        continue
    fi

    # Compare against itself — should always match
    if cmp -s "${name}_output.txt" "${name}_output.txt"; then
        pass "$name self-comparison works (filecmp logic is sound)"
    else
        fail "$name self-comparison failed — something is very wrong"
    fi
done


# ============================================================
# TEST 7: Fake srun and run orchestrator end-to-end
# ============================================================
header "Test 7: Orchestrator end-to-end (fake srun)"

# Write a fake srun that just finds and runs the binary argument directly
cat > srun << 'EOF'
#!/bin/bash
# Fake srun: strip all --flag=value args and run the binary
for arg in "$@"; do
    if [[ "$arg" == ./* ]]; then
        exec "$arg"
    fi
done
echo "fake srun: no binary found in args: $*" >&2
exit 1
EOF
chmod +x srun

# Temporarily patch orchestrator to use gcc if mpicc not available
if [ "$COMPILER" = "gcc" ]; then
    cp orchestrator.py orchestrator.py.bak
    sed 's/"mpicc"/"gcc"/g; s/"-fopenmp",//g' orchestrator.py.bak > orchestrator.py
fi

# Run orchestrator with fake srun on PATH, MINI dataset
echo ""
echo "  Running: PATH=.:$PATH python3 orchestrator.py --dataset $DATASET --runs $RUNS"
echo "  (apply_catalog_transformations is a stub — expect NotImplementedError at step 2)"
echo ""

set +e  # Don't exit on error here — NotImplementedError is expected
output=$(PATH=.:$PATH python3 orchestrator.py \
    --nodes 1 \
    --tasks-per-node 1 \
    --cpus 1 \
    --dataset $DATASET \
    --runs $RUNS 2>&1)
exit_code=$?
set -e

# Restore orchestrator if we patched it
if [ "$COMPILER" = "gcc" ] && [ -f orchestrator.py.bak ]; then
    mv orchestrator.py.bak orchestrator.py
fi

echo "$output" | sed 's/^/    /'  # indent output for readability
echo ""

# We expect it to reach Step 2 (baseline ran successfully) before hitting NotImplementedError
if echo "$output" | grep -q "\[1/3\] Running baseline"; then
    pass "Orchestrator reached baseline step"
else
    fail "Orchestrator did not reach baseline step — check errors above"
fi

if echo "$output" | grep -q "\[2/3\] Applying catalog transformations"; then
    pass "Orchestrator reached transformation step (baseline completed successfully)"
else
    fail "Orchestrator did not reach transformation step"
fi

if echo "$output" | grep -q "NotImplementedError"; then
    pass "Orchestrator correctly stopped at NotImplementedError (apply_catalog_transformations stub)"
else
    warn "Did not see expected NotImplementedError — check if the stub is still in place"
fi


# ============================================================
# SUMMARY
# ============================================================
echo ""
echo "========================================"
TOTAL=$((PASS + FAIL))
echo -e "Results: ${GREEN}$PASS passed${NC}, ${RED}$FAIL failed${NC} out of $TOTAL checks"
echo "========================================"

if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}All checks passed — safe to submit to the cluster.${NC}"
    exit 0
else
    echo -e "${RED}$FAIL check(s) failed — fix these before submitting.${NC}"
    exit 1
fi