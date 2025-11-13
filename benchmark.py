"""
Performance benchmarking script for GEMM implementations.
Tests various matrix sizes and thread counts, generates performance plots.
"""

import json
import os
import subprocess
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["figure.dpi"] = 300
matplotlib.rcParams["savefig.dpi"] = 300

BASELINE_SCRIPT = "src/baseline/sw_baseline.py"
OPTIMIZED_BINARY = "bin/sw_opt"
OPTIMIZED_BINARY_RUN = "./bin/sw_opt"


def run_benchmark(mode, N, T):
    """Run a single benchmark and return timing data."""
    try:
        if mode == "baseline":
            cmd = ["python3", BASELINE_SCRIPT, str(N), str(T)]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300, check=False
            )
            if result.returncode != 0:
                return None, None, f"Error: {result.stderr}"

            for line in result.stdout.strip().split("\n"):
                if "Execution time:" in line:
                    time_str = line.split("Execution time:")[1].split()[0]
                    return float(time_str), None, None

            return None, None, "No time output found"

        else:
            cmd = [OPTIMIZED_BINARY_RUN, str(N), str(T)]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300, check=False
            )
            if result.returncode != 0:
                return None, None, f"Error: {result.stderr}"

            exec_time = None
            checksum = None

            for line in result.stdout.strip().split("\n"):
                if "Execution time:" in line:
                    exec_time = float(line.split("Execution time:")[1].split()[0])
                elif "checksum=" in line:
                    checksum = line.split("checksum=")[1].strip()

            return exec_time, checksum, None

    except subprocess.TimeoutExpired:
        return None, None, "Timeout"
    except (OSError, ValueError, subprocess.SubprocessError) as e:
        return None, None, f"Exception: {str(e)}"


def build_binaries():
    """Build the C++ binaries."""
    print("Building C++ binaries...")
    subprocess.run(["make", "clean"], capture_output=True, text=True, check=False)
    result = subprocess.run(["make"], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print(f"Build failed: {result.stderr}")
        return False
    print("Build successful!")
    return True


def benchmark_matrix_sizes():
    """Benchmark across different matrix sizes with fixed thread count."""
    print("\n=== Benchmarking Matrix Sizes ===")

    sizes = [256, 512, 1024, 2048, 4096, 8192]
    threads = 8

    baseline_times = []
    optimized_times = []
    speedups = []

    for N in sizes:
        print(f"Testing N={N} with T={threads}...")

        base_time, _, base_err = run_benchmark("baseline", N, threads)
        if base_err:
            print(f"  Baseline error: {base_err}")
            baseline_times.append(None)
        else:
            baseline_times.append(base_time)
            print(f"  Baseline: {base_time:.4f}s")

        opt_time, _, opt_err = run_benchmark("optimized", N, threads)
        if opt_err:
            print(f"  Optimized error: {opt_err}")
            optimized_times.append(None)
        else:
            optimized_times.append(opt_time)
            print(f"  Optimized: {opt_time:.4f}s")

        if base_time and opt_time:
            speedup = base_time / opt_time
            speedups.append(speedup)
            print(f"  Speedup: {speedup:.2f}x")
        else:
            speedups.append(None)

    return sizes, baseline_times, optimized_times, speedups


def benchmark_thread_scaling():
    """Benchmark thread scaling with fixed matrix size."""
    print("\n=== Benchmarking Thread Scaling ===")

    N = 2048
    thread_counts = [
        1,
        2,
        4,
        8,
        16,
        32,
        48,
        64,
        72,
        88,
        96,
        128,
        144,
        160,
        176,
        192,
        208,
        224,
        240,
        256,
    ]

    baseline_times = []
    optimized_times = []
    speedups = []

    for T in thread_counts:
        print(f"Testing N={N} with T={T}...")

        base_time, _, base_err = run_benchmark("baseline", N, T)
        if base_err:
            print(f"  Baseline error: {base_err}")
            baseline_times.append(None)
        else:
            baseline_times.append(base_time)
            print(f"  Baseline: {base_time:.4f}s")

        opt_time, _, opt_err = run_benchmark("optimized", N, T)
        if opt_err:
            print(f"  Optimized error: {opt_err}")
            optimized_times.append(None)
        else:
            optimized_times.append(opt_time)
            print(f"  Optimized: {opt_time:.4f}s")

        if base_time and opt_time:
            speedup = base_time / opt_time
            speedups.append(speedup)
            print(f"  Speedup: {speedup:.2f}x")
        else:
            speedups.append(None)

    return thread_counts, baseline_times, optimized_times, speedups


def pack_results(x_key, x_values, base, opt, speedups):
    return {
        x_key: x_values,
        "baseline_times": base,
        "optimized_times": opt,
        "speedups": speedups,
    }


def create_plots():
    """Create and save performance plots."""
    os.makedirs("plots", exist_ok=True)

    sizes, base_times, opt_times, size_speedups = benchmark_matrix_sizes()

    plt.figure(figsize=(10, 6))
    valid_sizes = []
    valid_base = []
    valid_opt = []

    for s, b, o in zip(sizes, base_times, opt_times):
        if b is not None and o is not None:
            valid_sizes.append(s)
            valid_base.append(b)
            valid_opt.append(o)

    plt.loglog(valid_sizes, valid_base, "o-", label="Python Baseline", linewidth=2)
    plt.loglog(valid_sizes, valid_opt, "s-", label="C++ Optimized", linewidth=2)
    plt.xlabel("Matrix Size (N)")
    plt.ylabel("Execution Time (seconds)")
    plt.title("GEMM Performance: Execution Time vs Matrix Size")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/time_vs_size.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    valid_sizes_speedup = []
    valid_speedups = []

    for s, sp in zip(sizes, size_speedups):
        if sp is not None:
            valid_sizes_speedup.append(s)
            valid_speedups.append(sp)

    plt.semilogx(valid_sizes_speedup, valid_speedups, "o-", linewidth=2)
    plt.xlabel("Matrix Size (N)")
    plt.ylabel("Speedup (x)")
    plt.title("GEMM Speedup vs Matrix Size")
    plt.grid(True, alpha=0.3)
    plt.axhline(y=1, color="red", linestyle="--")
    plt.tight_layout()
    plt.savefig("plots/speedup_vs_size.png")
    plt.close()

    threads, base_times_t, opt_times_t, thread_speedups = benchmark_thread_scaling()

    plt.figure(figsize=(10, 6))
    valid_threads = []
    valid_base_t = []
    valid_opt_t = []

    for t, b, o in zip(threads, base_times_t, opt_times_t):
        if b is not None and o is not None:
            valid_threads.append(t)
            valid_base_t.append(b)
            valid_opt_t.append(o)

    plt.semilogx(
        valid_threads, valid_base_t, "o-", label="Python Baseline", linewidth=2
    )
    plt.semilogx(valid_threads, valid_opt_t, "s-", label="C++ Optimized", linewidth=2)
    plt.xlabel("Number of Threads")
    plt.ylabel("Execution Time (seconds)")
    plt.title("GEMM Performance: Execution Time vs Thread Count (N=2048)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/time_vs_threads.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    valid_threads_speedup = []
    valid_speedups_t = []

    for t, sp in zip(threads, thread_speedups):
        if sp is not None:
            valid_threads_speedup.append(t)
            valid_speedups_t.append(sp)

    plt.semilogx(valid_threads_speedup, valid_speedups_t, "s-", linewidth=2)
    plt.xlabel("Number of Threads")
    plt.ylabel("Speedup (x)")
    plt.title("GEMM Speedup vs Thread Count (N=2048)")
    plt.grid(True, alpha=0.3)
    plt.axhline(y=1, color="red", linestyle="--")
    plt.tight_layout()
    plt.savefig("plots/speedup_vs_threads.png")
    plt.close()

    results = {
        "matrix_size_scaling": pack_results(
            "sizes", sizes, base_times, opt_times, size_speedups
        ),
        "thread_scaling": pack_results(
            "threads", threads, base_times_t, opt_times_t, thread_speedups
        ),
    }

    with open("plots/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n=== Results Summary ===")
    print("Plots saved in 'plots/' directory")

    if valid_speedups:
        i = np.argmax(valid_speedups)
        print(f"Best speedup: {valid_speedups[i]:.2f}x at N={valid_sizes_speedup[i]}")

    if valid_speedups_t:
        i = np.argmax(valid_speedups_t)
        print(
            f"Best thread speedup: {valid_speedups_t[i]:.2f}x at T={valid_threads_speedup[i]}"
        )


def main():
    """Main benchmarking function."""
    print("GEMM Performance Benchmarking Script")
    print("====================================")

    if not Path(BASELINE_SCRIPT).exists():
        print(f"Error: Missing baseline script: {BASELINE_SCRIPT}")
        return 1

    if not build_binaries():
        return 1

    create_plots()
    return 0


if __name__ == "__main__":
    exit(main())
