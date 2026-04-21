#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EPCC_DIR="$ROOT_DIR/benchmarks/external/epcc-openmp-microbenchmarks/openmpbench_C_v31"
OSU_LAT="$ROOT_DIR/preflight_build/osu/mpi/pt2pt/osu_latency"
BT_MZ="$ROOT_DIR/benchmarks/external/npb/NPB3.4-MZ/NPB3.4-MZ-MPI/bin/bt-mz.S.x"
MPI_LAUNCHER="${MPI_LAUNCHER:-mpirun}"
MPI_NP="${MPI_NP:-2}"
OMP_THREADS="${OMP_THREADS:-2}"

if [[ ! -x "$EPCC_DIR/syncbench" || ! -x "$OSU_LAT" || ! -x "$BT_MZ" ]]; then
    echo "ERROR: required binaries are missing. Run ./benchmarks/external/build_selected.sh first." >&2
    exit 1
fi

echo "[1/3] EPCC OpenMP syncbench"
(
    cd "$EPCC_DIR"
    OMP_NUM_THREADS="$OMP_THREADS" ./syncbench --outer-repetitions 1 --test-time 100
)

echo
echo "[2/3] OSU MPI latency"
"$MPI_LAUNCHER" -np "$MPI_NP" "$OSU_LAT" -x 10 -i 100

echo
echo "[3/3] NPB-MZ BT-MZ hybrid"
OMP_NUM_THREADS="$OMP_THREADS" "$MPI_LAUNCHER" -np "$MPI_NP" "$BT_MZ"
