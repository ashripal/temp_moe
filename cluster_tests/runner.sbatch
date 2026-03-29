#!/bin/bash -l
# FILENAME: runner.sbatch

#SBATCH --account=davisjam
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=30
#SBATCH --partition=batch
#SBATCH --mem=160G
#SBATCH --time=00:30:00
#SBATCH --job-name=OptimizeHPC_test
#SBATCH --output=/home/min/a/rcalvome/joboutput/OptimizeHPC_test.%j.out
#SBATCH --error=/home/min/a/rcalvome/joboutput/OptimizeHPC_test.%j.err

set -euo pipefail

export BASE_SCRATCH=/local/scratch/a/rcalvome
export WORKDIR="$BASE_SCRATCH/temp_moe"
export VENV_DIR="$WORKDIR/.venv"

echo "Running on node: $(hostname)"
echo "Job start: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-unknown}"
echo

#module load gcc
#module load openmpi

rm -rf "$WORKDIR"
mkdir -p "$WORKDIR"

git clone https://github.com/ashripal/temp_moe.git "$WORKDIR"
cd "$WORKDIR"

python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python3 -m pip install --upgrade pip setuptools wheel

if [ -f requirements.txt ]; then
    pip install -r requirements.txt
fi

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "Python: $(which python3)"
python3 --version
echo "Workdir: $(pwd)"
echo

python3 cluster_tests/orchestrator.py --nodes 3 --tasks-per-node 1 --cpus 30 --dataset EXTRALARGE

echo
echo "Job end: $(date)"