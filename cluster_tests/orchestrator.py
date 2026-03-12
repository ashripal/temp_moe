import os
import subprocess
import filecmp
import statistics
import argparse

# --- ARGUMENT PARSING ---
parser = argparse.ArgumentParser(description="OptimizeHPC PolyBench Orchestrator")
parser.add_argument("--nodes",          type=int,   default=3,                help="Number of Slurm nodes")
parser.add_argument("--tasks-per-node", type=int,   default=1,                help="MPI tasks per node")
parser.add_argument("--cpus",           type=int,   default=40,               help="OpenMP threads per MPI task (cpus-per-task)")
parser.add_argument("--dataset",        type=str,   default="EXTRALARGE",     help="PolyBench dataset size (e.g. EXTRALARGE, LARGE)")
parser.add_argument("--runs",           type=int,   default=5,                help="Number of timed runs per kernel")
args = parser.parse_args()

# --- CONFIGURATION ---
KERNELS         = ["linear-algebra/kernels/gemm", "stencils/jacobi-2d"]
DATASET_SIZE    = f"{args.dataset}_DATASET"   # e.g. "EXTRALARGE_DATASET"
NODES           = args.nodes
TASKS_PER_NODE  = args.tasks_per_node
CPUS_PER_TASK   = args.cpus
RUNS            = args.runs


def apply_catalog_transformations(source_code, catalog):
    """
    This is where your LLM system lives.
    It takes the raw C code and applies the 'After' patterns
    from your catalog.

    Must return a non-None, non-empty string of optimized C code.
    """
    raise NotImplementedError(
        "apply_catalog_transformations() must be implemented before running."
    )


def compile_kernel(file_path, output_binary, extra_flags=None):
    """
    Compiles a PolyBench kernel using mpicc with OpenMP support.
    Uses the configured DATASET_SIZE flag.
    Raises RuntimeError if compilation fails.
    """
    flags = extra_flags or []
    cmd = [
        "mpicc",            # MPI-aware compiler wrapper (replaces gcc)
        "-O3",
        "-fopenmp",         # Enable OpenMP for hybrid parallelism
        "-I", "utilities/",
        "utilities/polybench.c",
        file_path,
        f"-D{DATASET_SIZE}",
        "-lm",
        "-o", output_binary
    ] + flags

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Compilation failed for {file_path}:\n"
            f"STDOUT: {result.stdout}\n"
            f"STDERR: {result.stderr}"
        )


def verify_output(baseline_output, candidate_output):
    """
    Compares two output dump files byte-for-byte.
    Returns True if they match, False otherwise.
    """
    return filecmp.cmp(baseline_output, candidate_output, shallow=False)


def run_benchmark(kernel_name, file_path, baseline_output_path=None, runs=RUNS):
    """
    Compiles and benchmarks a PolyBench kernel.

    - kernel_name:          short unique name to avoid filename collisions
    - file_path:            path to the .c source file
    - baseline_output_path: if provided, verifies this kernel's output against it
    - runs:                 number of timed srun executions

    Returns a dict with median, mean, stdev, all_times, verified, and output_file.
    """
    verify_binary = f"{kernel_name}_verify.exe"
    timing_binary = f"{kernel_name}_timing.exe"
    output_file   = f"{kernel_name}_output.txt"

    # --- Step 1: Compile for correctness verification ---
    compile_kernel(file_path, verify_binary, extra_flags=["-DPOLYBENCH_DUMP_ARRAYS"])

    result = subprocess.run(
        [f"./{verify_binary}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )
    if result.returncode != 0:
        raise RuntimeError(f"Verification binary failed to run for {kernel_name}")

    with open(output_file, "wb") as f:
        f.write(result.stdout)

    # --- Step 2: Verify correctness against baseline (if provided) ---
    if baseline_output_path is not None:
        verified = verify_output(baseline_output_path, output_file)
        if not verified:
            print(f"  WARNING: Output mismatch detected for {kernel_name}! "
                  f"Optimized result differs from baseline.")
    else:
        verified = True  # This IS the baseline; correctness assumed

    # --- Step 3: Compile for timing ---
    compile_kernel(file_path, timing_binary, extra_flags=["-DPOLYBENCH_TIME"])

    # --- Step 4: Timed runs via Slurm srun ---
    times = []
    for i in range(runs):
        result = subprocess.run(
            [
                "srun",
                f"--nodes={NODES}",
                f"--ntasks-per-node={TASKS_PER_NODE}",
                f"--cpus-per-task={CPUS_PER_TASK}",
                f"./{timing_binary}"
            ],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"srun failed on run {i+1} for {kernel_name}:\n"
                f"STDERR: {result.stderr}"
            )

        if not result.stdout.strip():
            raise RuntimeError(
                f"srun produced no stdout on run {i+1} for {kernel_name}.\n"
                f"STDERR: {result.stderr}"
            )

        try:
            raw_output = result.stdout.strip().split('\n')[-1]
            times.append(float(raw_output))
        except ValueError:
            raise RuntimeError(
                f"Could not parse timing output on run {i+1} for {kernel_name}.\n"
                f"Last line: '{raw_output}' | Full output: '{result.stdout.strip()}'"
            )

    sorted_times = sorted(times)
    n = len(sorted_times)
    if n % 2 == 0:
        median = (sorted_times[n // 2 - 1] + sorted_times[n // 2]) / 2.0
    else:
        median = sorted_times[n // 2]

    return {
        "median":      median,
        "mean":        statistics.mean(times),
        "stdev":       statistics.stdev(times) if len(times) > 1 else 0.0,
        "all_times":   times,
        "verified":    verified,
        "output_file": output_file,
    }


# --- MAIN EXECUTION LOOP ---
print(f"Configuration: {NODES} nodes | {TASKS_PER_NODE} MPI task(s)/node | "
      f"{CPUS_PER_TASK} OpenMP thread(s)/task | Dataset: {DATASET_SIZE} | Runs: {RUNS}")

for kernel_path in KERNELS:
    kernel_name = kernel_path.replace("/", "_")  # e.g. "linear-algebra_kernels_gemm"
    print(f"\n{'='*60}")
    print(f"Kernel: {kernel_path}")
    print(f"{'='*60}")

    # 1. Run Baseline
    print(f"  [1/3] Running baseline...")
    base_results = run_benchmark(
        kernel_name=f"{kernel_name}_base",
        file_path=f"{kernel_path}.c",
        baseline_output_path=None,
    )
    print(f"        Baseline median: {base_results['median']:.4f}s "
          f"(mean: {base_results['mean']:.4f}s, stdev: {base_results['stdev']:.4f}s)")

    # 2. Apply optimizations
    print(f"  [2/3] Applying catalog transformations...")
    with open(f"{kernel_path}.c", "r") as f:
        source_code = f.read()

    optimized_code = apply_catalog_transformations(source_code, "my_catalog.json")

    if not isinstance(optimized_code, str) or not optimized_code.strip():
        raise RuntimeError(
            f"apply_catalog_transformations() returned empty or None for {kernel_path}. "
            f"Aborting to avoid benchmarking a broken file."
        )

    opt_source_path = f"{kernel_path}_opt.c"
    with open(opt_source_path, "w") as f:
        f.write(optimized_code)

    # 3. Run Optimized (with correctness check against baseline output)
    print(f"  [3/3] Running optimized...")
    opt_results = run_benchmark(
        kernel_name=f"{kernel_name}_opt",
        file_path=opt_source_path,
        baseline_output_path=base_results["output_file"],
    )
    print(f"        Optimized median: {opt_results['median']:.4f}s "
          f"(mean: {opt_results['mean']:.4f}s, stdev: {opt_results['stdev']:.4f}s)")
    print(f"        Correctness verified: {opt_results['verified']}")

    # 4. Report
    if opt_results["verified"]:
        speedup = base_results["median"] / opt_results["median"]
        print(f"\n  >>> Speedup: {speedup:.2f}x")
    else:
        print(f"\n  >>> Speedup not reported — output verification FAILED.")