#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EPCC_DIR="$ROOT_DIR/benchmarks/external/epcc-openmp-microbenchmarks/openmpbench_C_v31"
OSU_SRC_DIR="$ROOT_DIR/benchmarks/external/osu-micro-benchmarks"
OSU_BUILD_DIR="$ROOT_DIR/preflight_build/osu"
NPB_MPI_DIR="$ROOT_DIR/benchmarks/external/npb/NPB3.4/NPB3.4-MPI"
NPB_MZ_DIR="$ROOT_DIR/benchmarks/external/npb/NPB3.4-MZ/NPB3.4-MZ-MPI"
JOBS="${JOBS:-4}"

require_dir() {
    local dir="$1"
    if [[ ! -d "$dir" ]]; then
        echo "ERROR: missing directory: $dir" >&2
        echo "Make sure the submodules are initialized." >&2
        exit 1
    fi
}

require_dir "$EPCC_DIR"
require_dir "$OSU_SRC_DIR"
require_dir "$NPB_MPI_DIR"
require_dir "$NPB_MZ_DIR"

echo "[1/4] Building EPCC OpenMP microbenchmarks (v3.1)..."
(
    cd "$EPCC_DIR"
    make clean >/dev/null 2>&1 || true
    make \
        CC=gcc \
        CPP=/usr/bin/cpp \
        CFLAGS="-O2 -fopenmp" \
        LDFLAGS="-O2 -fopenmp" \
        syncbench schedbench taskbench IDA=81 prog
)

echo "[2/4] Building OSU MPI benchmarks..."
find "$OSU_SRC_DIR" \
    \( -name "Makefile.am" -o -name "configure.ac" \) \
    -exec touch -d "2000-01-01 00:00:00" {} +
touch -d "2000-01-01 00:00:00" "$OSU_SRC_DIR/aclocal.m4"

mkdir -p "$OSU_BUILD_DIR"
(
    cd "$OSU_BUILD_DIR"
    "$OSU_SRC_DIR/configure" CC=mpicc CXX=mpicxx >/dev/null
    make -j"$JOBS"
)

NPB_MPI_FLAGS="-O3 -fallow-argument-mismatch -fallow-invalid-boz"
NPB_MZ_FLAGS="-O3 -fopenmp -fallow-argument-mismatch -fallow-invalid-boz"

echo "[3/4] Building NPB MPI shortlist..."
(
    cd "$NPB_MPI_DIR"
    [[ -f config/make.def ]] || cp config/make.def.template config/make.def
    mkdir -p bin
    make cg CLASS=S FFLAGS="$NPB_MPI_FLAGS" FLINKFLAGS="$NPB_MPI_FLAGS"
    make mg CLASS=S FFLAGS="$NPB_MPI_FLAGS" FLINKFLAGS="$NPB_MPI_FLAGS"
    make ft CLASS=S FFLAGS="$NPB_MPI_FLAGS" FLINKFLAGS="$NPB_MPI_FLAGS"
)

echo "[4/4] Building NPB-MZ MPI+OpenMP shortlist..."
(
    cd "$NPB_MZ_DIR"
    [[ -f config/make.def ]] || cp config/make.def.template config/make.def
    mkdir -p bin
    make bt-mz CLASS=S FFLAGS="$NPB_MZ_FLAGS" FLINKFLAGS="$NPB_MZ_FLAGS"
    make sp-mz CLASS=S FFLAGS="$NPB_MZ_FLAGS" FLINKFLAGS="$NPB_MZ_FLAGS"
)

cat <<EOF

Build complete.

Key binaries:
- $EPCC_DIR/syncbench
- $EPCC_DIR/schedbench
- $EPCC_DIR/taskbench
- $EPCC_DIR/arraybench_81
- $OSU_BUILD_DIR/mpi/pt2pt/osu_latency
- $OSU_BUILD_DIR/mpi/pt2pt/osu_bw
- $OSU_BUILD_DIR/mpi/pt2pt/osu_bibw
- $OSU_BUILD_DIR/mpi/collective/osu_allreduce
- $OSU_BUILD_DIR/mpi/collective/osu_alltoall
- $OSU_BUILD_DIR/mpi/collective/osu_barrier
- $NPB_MPI_DIR/bin/cg.S.x
- $NPB_MPI_DIR/bin/mg.S.x
- $NPB_MPI_DIR/bin/ft.S.x
- $NPB_MZ_DIR/bin/bt-mz.S.x
- $NPB_MZ_DIR/bin/sp-mz.S.x
EOF
