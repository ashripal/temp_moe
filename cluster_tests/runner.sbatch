#!/bin/bash -l
# FILENAME: runner.sbatch

#SBATCH --account=davisjam
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --partition=cpu
#SBATCH --mem=160G
#SBATCH --qos=standby
#SBATCH --time=00:30:00
#SBATCH --job-name=OptimizeHPC_test
#SBATCH --output=/home/rcalvome/joboutput/OptimizeHPC_test.%j.out
#SBATCH --error=/home/rcalvome/joboutput/OptimizeHPC_test.%j.err

set -euo pipefail


: "${SCRATCH:?SCRATCH environment variable is not set. Aborting.}"

echo "Running on node: $(hostname)"
echo "Job start: $(date)"
echo ""


module load gcc
module load openmpi


cd "$SCRATCH/AutoUP"
source .venv/bin/activate

if [ ! -f polybench-c-4.2.1-beta.tar.gz ]; then
    wget https://sourceforge.net/projects/polybench/files/polybench-c-4.2.1-beta.tar.gz
fi

if [ ! -d kernels ]; then
    tar -xzf polybench-c-4.2.1-beta.tar.gz
    mv polybench-c-4.2.1-beta kernels
fi


export OMP_NUM_THREADS=40
export OMP_PROC_BIND=close
export OMP_PLACES=cores


python orchestrator.py --nodes 3 --tasks-per-node 1 --cpus 40 --dataset EXTRALARGE

echo ""
echo "Job end: $(date)"
