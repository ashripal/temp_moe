#!/bin/bash -l
# FILENAME: runner.sbatch

#SBATCH --account=davisjam
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=batch
#SBATCH --mem=160G
#SBATCH --time=10:00:00
#SBATCH --job-name=OptimizeHPC_parallel
#SBATCH --output=/home/min/a/rcalvome/joboutput/OptimizeHPC_parallel.%j.out
#SBATCH --error=/home/min/a/rcalvome/joboutput/OptimizeHPC_parallel.%j.err

set -euo pipefail

export PROJECT_ROOT=/home/min/a/rcalvome/temp_moe
export VENV_DIR="$PROJECT_ROOT/.venv"
export JOB_SCRATCH="/local/scratch/a/rcalvome/${SLURM_JOB_ID:-manual_job}"
export TMPDIR="$JOB_SCRATCH/tmp"
export PIP_CACHE_DIR="$JOB_SCRATCH/pip-cache"
export HF_HOME="$JOB_SCRATCH/hf-home"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export AUTOUP_REPO_ROOT="$PROJECT_ROOT"

BENCHMARK_SELECTOR="${BENCHMARK_SELECTOR:-parallel-shortlist}"
ORCH_RUNS="${ORCH_RUNS:-5}"
REBUILD_EXTERNAL="${REBUILD_EXTERNAL:-0}"

case "$BENCHMARK_SELECTOR" in
    parallel-shortlist|all-external|external/*)
        REQUIRE_OPENAI_KEY=0
        ;;
    *)
        REQUIRE_OPENAI_KEY=1
        ;;
esac

if [ "$REQUIRE_OPENAI_KEY" -eq 1 ]; then
    : "${OPENAI_API_KEY:?Set OPENAI_API_KEY before submitting the job.}"
fi

mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$HF_HOME"
mkdir -p "$PROJECT_ROOT/build/orchestrator_runs" /home/min/a/rcalvome/joboutput

echo "Running on node: $(hostname)"
echo "Job start: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-unknown}"
echo "Allocated nodes:"
scontrol show hostnames "${SLURM_NODELIST:-}" || true
echo

if [ ! -d "$PROJECT_ROOT" ]; then
    echo "ERROR: PROJECT_ROOT does not exist: $PROJECT_ROOT" >&2
    exit 1
fi

cd "$PROJECT_ROOT"

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

python3 -m pip install --upgrade pip setuptools wheel

if [ -f requirements.txt ]; then
    pip install -r requirements.txt
fi

if ! command -v mpicc >/dev/null 2>&1; then
    echo "ERROR: mpicc not found in PATH. Load the required compiler/MPI modules before running sbatch." >&2
    exit 1
fi

if ! command -v srun >/dev/null 2>&1; then
    echo "ERROR: srun not found in PATH." >&2
    exit 1
fi

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "Python: $(which python3)"
python3 --version
echo "Project root: $(pwd)"
echo "Venv: $VENV_DIR"
echo "TMPDIR: $TMPDIR"
echo "Benchmark selector: $BENCHMARK_SELECTOR"
echo "Timed runs per benchmark: $ORCH_RUNS"
echo "Rebuild external bundle: $REBUILD_EXTERNAL"
if [ "$REQUIRE_OPENAI_KEY" -eq 0 ]; then
    echo "OPENAI_API_KEY: not required for external-only benchmark selection"
else
    echo "OPENAI_API_KEY: required"
fi
echo

ORCH_ARGS=(
  --nodes "${SLURM_JOB_NUM_NODES:-3}"
  --tasks-per-node 1
  --cpus "${SLURM_CPUS_PER_TASK}"
  --dataset LARGE
  --runs "$ORCH_RUNS"
  --kernels "$BENCHMARK_SELECTOR"
  --use-srun
  --build-dir "build/orchestrator_runs/${SLURM_JOB_ID}"
)

if [ "$REBUILD_EXTERNAL" = "1" ]; then
    ORCH_ARGS+=(--rebuild-external)
fi

python3 cluster_tests/orchestrator.py "${ORCH_ARGS[@]}"

echo
echo "Job end: $(date)"
