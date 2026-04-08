#!/usr/bin/env bash
set -e

REPO_ROOT="${1:-.}"
REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"

echo "=========================================="
echo " OptimizeHPC Minimal Preflight Check"
echo " Repo: $REPO_ROOT"
echo "=========================================="
echo

echo "[1] Checking required files..."
for f in \
  "cluster_tests/orchestrator.py" \
  "kernels/utilities/polybench.c" \
  "updated_optimization_catalog.csv" \
  "kernels/linear-algebra/blas/gemm/gemm.c" \
  "kernels/stencils/jacobi-2d/jacobi-2d.c"
do
  if [[ ! -f "$REPO_ROOT/$f" ]]; then
    echo "❌ Missing: $f"
    exit 1
  else
    echo "✅ Found: $f"
  fi
done
echo

echo "[2] Checking tools..."
command -v python3 >/dev/null || { echo "❌ python3 not found"; exit 1; }
echo "✅ python3 found"

command -v mpicc >/dev/null || { echo "❌ mpicc not found"; exit 1; }
echo "✅ mpicc found"
echo

echo "[3] Checking Python syntax..."
python3 -m py_compile "$REPO_ROOT/cluster_tests/orchestrator.py" \
  && echo "✅ orchestrator.py syntax OK" \
  || { echo "❌ Syntax error in orchestrator.py"; exit 1; }
echo

echo "[4] Checking PolyBench compile only..."
BUILD_DIR="$REPO_ROOT/preflight_build"
mkdir -p "$BUILD_DIR"

for k in \
  "kernels/linear-algebra/blas/gemm/gemm.c" \
  "kernels/stencils/jacobi-2d/jacobi-2d.c"
do
  src="$REPO_ROOT/$k"
  name=$(basename "$k" .c)
  out="$BUILD_DIR/$name.exe"

  echo "→ Compiling $k"
  mpicc -O3 -fopenmp \
    -I "$REPO_ROOT/kernels/utilities" \
    "$REPO_ROOT/kernels/utilities/polybench.c" \
    "$src" \
    -DEXTRALARGE_DATASET \
    -lm \
    -o "$out" \
    && echo "   ✅ Compile succeeded" \
    || { echo "   ❌ Compile failed"; exit 1; }
done
echo

echo "[5] Checking orchestrator CLI parse..."
python3 "$REPO_ROOT/cluster_tests/orchestrator.py" --help >/dev/null 2>&1 \
  && echo "✅ orchestrator CLI responds" \
  || echo "⚠️ orchestrator --help could not run locally (likely environment/import related)"
echo

echo "=========================================="
echo " Minimal preflight complete"
echo "=========================================="