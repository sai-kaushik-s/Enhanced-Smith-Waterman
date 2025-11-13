# Acceleration of the Smith-Waterman Algorithm

[GitHub Repository](https://github.com/sai-kaushik-s/Enhanced-Smith-Waterman/)

## Team: Sai Kaushik S (2025CSZ8470) & Yosef Ro (2025ANZ8223)

## 1. Problem description

- Application: Smith-Waterman Algorithm
- Input sizes tested: N = 256, 512, 1024, 2048, 4096, 8192
- Thread sizes tested: P = 1, 2, 4, 8, 16, 32, 48, 64, 72, 88, 96, 128, 144, 160, 176, 192, 208, 224, 240, 256

## 2. Baseline

```bash
python3 src/baseline/sw_baseline.py 1024 4
```

## 3. Optimizations implemented

### Compilation

You must compile with:

1.  **Target architecture flags**: `-march=native` (to enable AVX2/AVX-512)
2.  **OpenMP enabled**: `-fopenmp`
3.  **Fused Multiply-Add enabled**: `-mfma`
4.  **High optimization level**: `-O3`

## 4. Experimental methodology

# Smith-Waterman Algorithm Benchmarking & Profiling Framework

This project provides a unified `run.sh` script that manages all experiments, comparisons, and hardware performance profiling for the Smith-Waterman Algorithm baseline and optimized implementations.

The script supports:

- Running the Python baseline
- Running the C++ optimized version
- Correctness comparison between Python, C++ baseline, and C++ optimized
- Performance counter analysis using `perf stat`

All modes perform one run only (no averaging or median calculations).

#### Events Collected

- cycles
- instructions
- branches
- branch-misses
- cache-references (LLC)
- cache-misses (LLC)
- L1-dcache-loads
- L1-dcache-load-misses
- L1-icache-loads
- L1-icache-load-misses

#### Derived Metrics

- IPC – Instructions Per Cycle
- BrMiss% – Branch Miss Rate
- LLCMiss% – Last-Level Cache Miss Rate
- L1DMiss% – L1 Data-Cache Miss Rate
- L1IMiss% – L1 Instruction-Cache Miss Rate
- GFLOPS – Giga-Floating-Point Operations per Second

A summary table is printed comparing Python vs C++ baseline vs optimized.

## 5. Results

Performance metrics measured for the different scripts to compare:

Comparing the baseline Python codes and optimized C++ codes with different sequence lengths as mentioned and below are the plots generated.

![Plot of time taken to complete the Smith-Waterman algorithm against the different sizes](plots/time_vs_size.png)
![Plot of speedup of the Smith-Waterman algorithm against the different sizes](plots/speedup_vs_size.png)

Comparing the baseline Python codes and optimized C++ codes with different threads as mentioned.

![Plot of time taken to complete the Smith-Waterman algorithm against the different thread count](plots/time_vs_threads.png)
![Plot of speedup of the Smith-Waterman algorithm against the different thread count](plots/speedup_vs_threads.png)

## 6. Analysis

## 7. Reproducibility

- Run the setup script to install the required packages.

```bash
./setup.sh
```

- Enable the python virtual environment where the required packages are installed.

```bash
source .venv/bin/activate
```

- Run the make script to build the C++ binaries.

```bash
make clean && make all
```

- Run the run script with the specific modes, sequence length and the number of threads.

```bash
./run.sh MODE N P
```

- `MODE` = `baseline`, `optimized`, `compare`, or `perf`
- `N` = sequence length
- `P` = process/thread count

### 1. Baseline (Python)

Runs the reference Python Smith-Waterman Algorithm implementation.

**Command:**

```bash
./run.sh baseline N P
```

### 2. Optimized (C++)

Runs the optimized C++ Smith-Waterman Algorithm binary.

**Command:**

```bash
./run.sh optimized N P
```

### 3. Comparison & Verification

Runs:

1. Python baseline
2. C++ baseline (checksum only)
3. C++ optimized version

Verifies checksums and reports speedup using `/usr/bin/time`.

**Command:**

```bash
./run.sh compare N P
```

### 4. Performance Counter Analysis (perf)

Runs all three implementations under `perf stat` and collects detailed CPU hardware metrics.

**Command:**

```bash
./run.sh perf N P
```
