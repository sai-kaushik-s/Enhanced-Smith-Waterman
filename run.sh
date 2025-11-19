#!/usr/bin/env bash
set -euo pipefail

# -----------------------
# Configurable variables
# -----------------------
VENV_ACTIVATE=".venv/bin/activate"
PYTHON_BIN="python3"
TIME_BIN="/usr/bin/time"
PERF_BIN="perf"

PY_BASELINE="src/baseline/sw_baseline.py"
C_BASELINE="bin/sw_baseline"
OPTIMIZED_BIN="bin/sw_opt"
SCORER_SCRIPT="scorer.py"

MODE=${1:-baseline}
N=${2:-1024}
P=${3:-4}

# -----------------------
# Helpers
# -----------------------
err() { echo "Error: $*" >&2; exit 1; }
require_file() {
    [ -f "$1" ] || err "$1 not found"
}
require_exec() {
    [ -x "$1" ] || err "$1 not found or not executable. Did you run 'make'?"
}

# -----------------------
# Modes
# -----------------------
if [ "$MODE" = "baseline" ]; then
    echo "Running baseline Python implementation"
    if [ -f "$VENV_ACTIVATE" ]; then
        source "$VENV_ACTIVATE"
    fi
    "$PYTHON_BIN" "$PY_BASELINE" "$N" "$P"

elif [ "$MODE" = "baseline_cpp" ]; then
    echo "Running baseline CPP implementation"
    require_exec "$C_BASELINE"
    "$C_BASELINE" "$N" "$P"

elif [ "$MODE" = "optimized" ]; then
    echo "Running optimized binary"
    require_exec "$OPTIMIZED_BIN"
    "$OPTIMIZED_BIN" "$N" "$P"

elif [ "$MODE" = "compare" ]; then
    echo "Comparing baseline and optimized implementations for N=$N, P=$P"

    BASE_PERF_BIN="$PYTHON_BIN $PY_BASELINE"
    BASE_PERF_FILE="$PY_BASELINE"
    BASE_BIN="$C_BASELINE"
    OPT_BIN="$OPTIMIZED_BIN"

    require_file "$BASE_PERF_FILE"
    require_exec "$BASE_BIN"
    require_exec "$OPT_BIN"

    echo "----------------------------------------"
    echo "1/3 Running Python baseline (for performance)..."
    BASE_PERF_OUT=$( { $TIME_BIN -p $BASE_PERF_BIN "$N" "$P"; } 2>&1 )

    echo "2/3 Running C baseline (for checksum verification)..."
    BASE_OUT=$( { $TIME_BIN -p "$BASE_BIN" "$N" "$P"; } 2>&1 )

    echo "3/3 Running Optimized binary (for performance & checksum)..."
    OPT_OUT=$( { $TIME_BIN -p "$OPT_BIN" "$N" "$P"; } 2>&1 )

    echo "----------------------------------------"
    echo "Analyzing results..."

    CHECK_BASE=$(echo "$BASE_OUT" | awk '/Smith-Waterman score:/ {print $NF}')
    CHECK_OPT=$(echo "$OPT_OUT" | awk '/Smith-Waterman score:/ {print $NF}')

    TIME_BASE_INT=$(echo "$BASE_PERF_OUT" | awk '/Execution time:/ {print $3}')
    TIME_BASE_REAL=$(echo "$BASE_PERF_OUT" | awk '/^real/ {print $2}')
    TIME_OPT_INT=$(echo "$OPT_OUT" | awk '/Execution time:/ {print $3}')
    TIME_OPT_REAL=$(echo "$OPT_OUT" | awk '/^real/ {print $2}')

    if [ -z "$CHECK_BASE" ] || [ -z "$CHECK_OPT" ]; then
        echo "Error: Could not parse checksums."
        echo "Baseline (C) output: $BASE_OUT"
        echo "Optimized output:    $OPT_OUT"
        exit 1
    fi

    if [ "$CHECK_BASE" != "$CHECK_OPT" ]; then
        echo "FAILED: Checksum mismatch!"
        echo "Baseline (C): $CHECK_BASE"
        echo "Optimized:    $CHECK_OPT"
        exit 1
    else
        echo "SUCCESS: Checksums match ($CHECK_BASE)"
    fi

    echo "----------------------------------------"
    echo "Performance Comparison..."

    if [ -z "$TIME_BASE_INT" ] || [ -z "$TIME_OPT_INT" ]; then
        echo "Error: Could not parse timing data."
        exit 1
    fi

    echo -e "\033[1;32mInternal Running Time:\033[0m"
    echo -e "  \033[1mPython\033[0m:    ${TIME_BASE_INT}s"
    echo -e "  \033[1mOptimized\033[0m: ${TIME_OPT_INT}s"
    echo -e "\033[1;34mTotal Running Time:\033[0m"
    echo -e "  \033[1mPython\033[0m:    ${TIME_BASE_REAL}s"
    echo -e "  \033[1mOptimized\033[0m: ${TIME_OPT_REAL}s"
    echo "----------------------------------------"

    SPEEDUP_INT=$(awk "BEGIN {if ($TIME_OPT_INT > 0) printf \"%.2f\", $TIME_BASE_INT / $TIME_OPT_INT; else print \"N/A\"}")
    SPEEDUP_REAL=$(awk "BEGIN {if ($TIME_OPT_REAL > 0) printf \"%.2f\", $TIME_BASE_REAL / $TIME_OPT_REAL; else print \"N/A\"}")

    echo -e "\033[1;33mSpeedup (Internal):\033[0m  ${SPEEDUP_INT}x"
    echo -e "\033[1;33mSpeedup (Real Time):\033[0m ${SPEEDUP_REAL}x"

elif [ "$MODE" = "compare_cpp" ]; then
    echo "Comparing C++ baseline and optimized implementations for N=$N, P=$P"

    BASE_BIN="$C_BASELINE"
    OPT_BIN="$OPTIMIZED_BIN"

    require_exec "$BASE_BIN"
    require_exec "$OPT_BIN"

    echo "----------------------------------------"
    echo "1/2 Running Baseline Binary..."
    BASE_OUT=$( { $TIME_BIN -p "$BASE_BIN" "$N" "$P"; } 2>&1 )

    echo "2/2 Running Optimized Binary..."
    OPT_OUT=$( { $TIME_BIN -p "$OPT_BIN" "$N" "$P"; } 2>&1 )

    echo "----------------------------------------"
    echo "Analyzing results..."

    CHECK_BASE=$(echo "$BASE_OUT" | awk '/Smith-Waterman score:/ {print $NF}')
    CHECK_OPT=$(echo "$OPT_OUT" | awk '/Smith-Waterman score:/ {print $NF}')

    TIME_BASE_INT=$(echo "$BASE_OUT" | awk '/Execution time:/ {print $3}')
    TIME_BASE_REAL=$(echo "$BASE_OUT" | awk '/^real/ {print $2}')
    TIME_OPT_INT=$(echo "$OPT_OUT" | awk '/Execution time:/ {print $3}')
    TIME_OPT_REAL=$(echo "$OPT_OUT" | awk '/^real/ {print $2}')

    if [ -z "$CHECK_BASE" ] || [ -z "$CHECK_OPT" ]; then
        echo "Error: Could not parse checksums."
        echo "Baseline (C) output: $BASE_OUT"
        echo "Optimized output:    $OPT_OUT"
        exit 1
    fi

    if [ "$CHECK_BASE" != "$CHECK_OPT" ]; then
        echo "FAILED: Checksum mismatch!"
        echo "Baseline (C): $CHECK_BASE"
        echo "Optimized:    $CHECK_OPT"
        exit 1
    else
        echo "SUCCESS: Checksums match ($CHECK_BASE)"
    fi

    echo "----------------------------------------"
    echo "Performance Comparison..."

    if [ -z "$TIME_BASE_INT" ] || [ -z "$TIME_OPT_INT" ]; then
        echo "Error: Could not parse timing data."
        exit 1
    fi

    echo -e "\033[1;32mInternal Running Time:\033[0m"
    echo -e "  \033[1mBaseline\033[0m:    ${TIME_BASE_INT}s"
    echo -e "  \033[1mOptimized\033[0m: ${TIME_OPT_INT}s"
    echo -e "\033[1;34mTotal Running Time:\033[0m"
    echo -e "  \033[1mBaseline\033[0m:    ${TIME_BASE_REAL}s"
    echo -e "  \033[1mOptimized\033[0m: ${TIME_OPT_REAL}s"
    echo "----------------------------------------"

    SPEEDUP_INT=$(awk "BEGIN {if ($TIME_OPT_INT > 0) printf \"%.2f\", $TIME_BASE_INT / $TIME_OPT_INT; else print \"N/A\"}")
    SPEEDUP_REAL=$(awk "BEGIN {if ($TIME_OPT_REAL > 0) printf \"%.2f\", $TIME_BASE_REAL / $TIME_OPT_REAL; else print \"N/A\"}")

    echo -e "\033[1;33mSpeedup (Internal):\033[0m  ${SPEEDUP_INT}x"
    echo -e "\033[1;33mSpeedup (Real Time):\033[0m ${SPEEDUP_REAL}x"

elif [ "$MODE" = "scorer" ]; then
    echo "Running scorer.py for the baseline and optimized implementations for N=$N, P=$P"

    SCORER_BIN="$PYTHON_BIN $SCORER_SCRIPT"
    BASE_PERF_BIN="$PYTHON_BIN $PY_BASELINE"
    BASE_PERF_FILE="$PY_BASELINE"
    OPT_BIN="$OPTIMIZED_BIN"

    require_file "$BASE_PERF_FILE"
    require_exec "$OPT_BIN"

    $PYTHON_BIN "$SCORER_SCRIPT" \
        --baseline "$BASE_PERF_BIN $N $P" \
        --optimized "$OPT_BIN $N $P"

elif [ "$MODE" = "perf" ]; then
    echo "Running perf profiling for N=$N, P=$P"

    BASE_PY="$PYTHON_BIN $PY_BASELINE"
    BASE_C="$C_BASELINE"
    OPT="$OPTIMIZED_BIN"

    require_file "$PY_BASELINE"
    require_exec "$BASE_C"
    require_exec "$OPT"

    run_perf() {
        local CMD="$1"
        $PERF_BIN stat \
            -e cycles,instructions,branches,branch-misses,\
cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,\
L1-icache-loads,L1-icache-load-misses \
            $CMD 2>&1 | awk -v N_DIM="$N" '
                $2 != "seconds" && $3 != "time" {
                    f1 = $1; f2 = $2
                    gsub(",", "", f1); gsub(",", "", f2)
                    if (f1+0 > 0) { val = f1; evt = $2 } else { val = f2; evt = $1 }

                    if (evt == "cycles")                  {cycles=val}
                    if (evt == "instructions")            {ins=val}
                    if (evt == "branches")                {br=val}
                    if (evt == "branch-misses")           {brm=val}
                    if (evt == "cache-references")        {cr=val}
                    if (evt == "cache-misses")            {cm=val}
                    if (evt == "L1-dcache-loads")         {l1d=val}
                    if (evt == "L1-dcache-load-misses")   {l1dm=val}
                    if (evt == "L1-icache-loads")         {l1i=val}
                    if (evt == "L1-icache-load-misses")   {l1im=val}
                }

                $2 == "seconds" && $3 == "time" {time=$1}

                END {
                    ipc = (cycles>0 ? ins/cycles : 0)
                    brp = (br>0 ? (brm/br)*100 : 0)
                    cmp = (cr>0 ? (cm/cr)*100 : 0)
                    l1dp = (l1d>0 ? (l1dm/l1d)*100 : 0)
                    l1ip = (l1i>0 ? (l1im/l1i)*100 : 0)

                    flops = (2 * N_DIM * N_DIM * N_DIM)
                    gflops = (time>0 ? (flops / time) / 1e9 : 0)

                    printf "%s %s %.3f %s %.2f%% %s %.2f%% %.2f%% %.2f%% %.2f",
                        cycles, ins, ipc,
                        brm, brp,
                        cm, cmp,
                        l1dp, l1ip, gflops
                }
            '
    }

    PY=$(run_perf "$BASE_PY $N $P")
    CBASE=$(run_perf "$BASE_C $N $P")
    OPTSTATS=$(run_perf "$OPT $N $P")

    rows=(
        "Python_Baseline $PY"
        "C++_Baseline $CBASE"
        "C++_Optimized $OPTSTATS"
    )

    titles=( "Target" "Cycles" "Instr" "IPC" "BrMiss" "BrMiss%" "LLCMiss" "LLCMiss%" "L1DMiss%" "L1IMiss%" "GFLOPS" )
    col_count=${#titles[@]}

    for ((i=0; i<col_count; i++)); do widths[$i]=${#titles[$i]}; done

    for row in "${rows[@]}"; do
        i=0
        for field in $row; do
            len=${#field}
            (( len > widths[$i] )) && widths[$i]=$len
            ((i++))
        done
    done

    print_separator() {
        echo -n "+"
        for w in "${widths[@]}"; do
            printf -- "%0.s-" $(seq 1 $((w + 2)))
            echo -n "+"
        done
        echo
    }

    print_row() {
        i=0
        echo -n "|"
        for field in $1; do
            printf " %-${widths[$i]}s " "$field"
            echo -n "|"
            ((i++))
        done
        echo
    }

    echo -e "\033[1;34m-----------------------------------------\033[0m"
    echo -e "\033[1;32mPerformance Counters (perf stat) for N=$N\033[0m"
    echo -e "\033[1;34m-----------------------------------------\033[0m"

    print_separator
    header=""
    for title in "${titles[@]}"; do header="$header$title "; done
    print_row "$header"
    print_separator

    for row in "${rows[@]}"; do
        print_row "$row"
    done

    print_separator

else
    echo "Unknown mode: $MODE"
    exit 1
fi
