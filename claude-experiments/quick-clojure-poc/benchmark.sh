#!/bin/bash

# Benchmark script comparing our language vs Clojure
# Supports two modes:
#   1. Cold start: measures full startup + execution time
#   2. Warm: compiles function, warms up, then measures just execution

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUR_LANG="$SCRIPT_DIR/target/release/quick-clojure-poc"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

usage() {
    echo "Usage: $0 [OPTIONS] <mode> <file_or_code>"
    echo ""
    echo "Modes:"
    echo "  cold     Benchmark cold start (includes startup time)"
    echo "  warm     Benchmark warm execution (compile, warmup, then measure)"
    echo ""
    echo "Options:"
    echo "  -n NUM   Number of iterations (default: 10 for cold, 1000 for warm)"
    echo "  -w NUM   Warmup iterations for warm mode (default: 100)"
    echo "  -f       Treat argument as a file path"
    echo "  -c       Treat argument as inline code (default)"
    echo "  -h       Show this help"
    echo ""
    echo "Examples:"
    echo "  # Cold start benchmark with a file"
    echo "  $0 cold -f tests/basic.clj"
    echo ""
    echo "  # Cold start benchmark with inline code"
    echo "  $0 cold '(+ 1 2)'"
    echo ""
    echo "  # Warm benchmark - define a function and call it"
    echo "  $0 warm '(defn fib [n] (if (< n 2) n (+ (fib (- n 1)) (fib (- n 2))))) (fib 20)'"
    echo ""
    echo "  # Warm benchmark with custom iterations"
    echo "  $0 -n 500 -w 50 warm '(+ 1 2 3 4 5)'"
    exit 1
}

# Check if our language is built
check_build() {
    if [ ! -f "$OUR_LANG" ]; then
        echo -e "${YELLOW}Building release version of our language...${NC}"
        cd "$SCRIPT_DIR"
        cargo build --release 2>&1 | grep -v "Compiling\|Finished\|warning" || true
        echo -e "${GREEN}Build complete!${NC}"
    fi
}

# Check if Clojure is available
check_clojure() {
    if ! command -v clojure &> /dev/null; then
        echo -e "${RED}Error: Clojure not found in PATH${NC}"
        echo "Install Clojure: https://clojure.org/guides/install_clojure"
        exit 1
    fi
}

# Time a command and return milliseconds
time_cmd() {
    local cmd="$1"
    local start end elapsed

    # Use gdate on macOS if available for nanosecond precision
    if command -v gdate &> /dev/null; then
        start=$(gdate +%s%N)
        eval "$cmd" > /dev/null 2>&1
        end=$(gdate +%s%N)
        elapsed=$(( (end - start) / 1000000 ))
    else
        # Fallback to milliseconds with date
        start=$(date +%s%3N 2>/dev/null || echo $(($(date +%s) * 1000)))
        eval "$cmd" > /dev/null 2>&1
        end=$(date +%s%3N 2>/dev/null || echo $(($(date +%s) * 1000)))
        elapsed=$((end - start))
    fi

    echo "$elapsed"
}

# Run cold start benchmark
benchmark_cold() {
    local code="$1"
    local is_file="$2"
    local iterations="${3:-10}"

    local tmpfile
    if [ "$is_file" = "true" ]; then
        tmpfile="$code"
    else
        tmpfile=$(mktemp /tmp/bench_XXXXXX.clj)
        echo "$code" > "$tmpfile"
        trap "rm -f $tmpfile" EXIT
    fi

    echo -e "\n${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  COLD START BENCHMARK ($iterations iterations)${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

    if [ "$is_file" = "true" ]; then
        echo -e "File: $code"
    else
        echo -e "Code: $code"
    fi
    echo ""

    # Benchmark our language
    echo -e "${GREEN}Our Language:${NC}"
    local our_times=()
    local our_total=0

    for ((i=1; i<=iterations; i++)); do
        local t=$(time_cmd "\"$OUR_LANG\" \"$tmpfile\"")
        our_times+=("$t")
        our_total=$((our_total + t))
        printf "  Run %2d: %6d ms\n" "$i" "$t"
    done

    local our_avg=$((our_total / iterations))
    echo -e "  ${GREEN}Average: ${our_avg} ms${NC}"

    # Benchmark Clojure
    echo ""
    echo -e "${YELLOW}Clojure:${NC}"
    local clj_times=()
    local clj_total=0

    for ((i=1; i<=iterations; i++)); do
        local t=$(time_cmd "clojure \"$tmpfile\"")
        clj_times+=("$t")
        clj_total=$((clj_total + t))
        printf "  Run %2d: %6d ms\n" "$i" "$t"
    done

    local clj_avg=$((clj_total / iterations))
    echo -e "  ${YELLOW}Average: ${clj_avg} ms${NC}"

    # Summary
    echo ""
    echo -e "${BLUE}───────────────────────────────────────────────────────────${NC}"
    echo -e "${BLUE}  SUMMARY${NC}"
    echo -e "${BLUE}───────────────────────────────────────────────────────────${NC}"
    echo -e "  Our Language: ${GREEN}${our_avg} ms${NC}"
    echo -e "  Clojure:      ${YELLOW}${clj_avg} ms${NC}"

    if [ "$our_avg" -lt "$clj_avg" ]; then
        local speedup=$(echo "scale=2; $clj_avg / $our_avg" | bc)
        echo -e "  ${GREEN}Our language is ${speedup}x faster!${NC}"
    elif [ "$clj_avg" -lt "$our_avg" ]; then
        local speedup=$(echo "scale=2; $our_avg / $clj_avg" | bc)
        echo -e "  ${YELLOW}Clojure is ${speedup}x faster${NC}"
    else
        echo -e "  ${BLUE}Same performance${NC}"
    fi
}

# Run warm benchmark
benchmark_warm() {
    local code="$1"
    local is_file="$2"
    local iterations="${3:-1000}"
    local warmup="${4:-100}"

    local full_code
    if [ "$is_file" = "true" ]; then
        full_code=$(cat "$code")
    else
        full_code="$code"
    fi

    echo -e "\n${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  WARM BENCHMARK (${warmup} warmup, ${iterations} measured)${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

    if [ "$is_file" = "true" ]; then
        echo -e "File: $code"
    else
        echo -e "Code: $code"
    fi
    echo ""

    # For warm benchmarks, we use loop/recur which both languages support
    # This ensures we're measuring actual execution, not parsing/compilation

    # Create Clojure warm benchmark file
    local clj_warm=$(mktemp /tmp/bench_warm_clj_XXXXXX.clj)
    cat > "$clj_warm" << EOF
;; Warm benchmark wrapper for Clojure
$full_code

;; Wrap as a function for repeated calling
(defn bench-target [] $full_code)

(defn run-benchmark []
  (let [warmup-iterations $warmup
        measured-iterations $iterations]
    ;; Warmup phase - call the compiled function
    (loop [i 0]
      (when (< i warmup-iterations)
        (bench-target)
        (recur (inc i))))

    ;; Measured phase
    (let [start (System/nanoTime)]
      (loop [i 0]
        (when (< i measured-iterations)
          (bench-target)
          (recur (inc i))))
      (let [end (System/nanoTime)
            elapsed-ms (/ (- end start) 1000000.0)
            avg-us (/ (* (- end start) 0.001) measured-iterations)]
        (println (format "Total: %.2f ms" elapsed-ms))
        (println (format "Average: %.2f µs per iteration" avg-us))))))

(run-benchmark)
EOF

    # Create our language warm benchmark file
    # Use loop/recur so we compile once and just run the loop
    local our_warm=$(mktemp /tmp/bench_warm_our_XXXXXX.clj)
    cat > "$our_warm" << EOF
;; Warm benchmark wrapper for our language
$full_code

;; Wrap as a function for repeated calling
(def bench-target (fn [] $full_code))

;; Warmup phase using loop/recur (compiled once, runs $warmup times)
(loop [i 0]
  (if (< i $warmup)
    (do
      (bench-target)
      (recur (+ i 1)))
    nil))

;; Signal warmup done - we'll measure from here externally
;; But include measured iterations in same file for fair comparison
(loop [i 0]
  (if (< i $iterations)
    (do
      (bench-target)
      (recur (+ i 1)))
    nil))
EOF

    trap "rm -f $clj_warm $our_warm" EXIT

    # For our language, we need to measure externally but isolate just the measured portion
    # Since we can't time internally, we'll run warmup separately

    echo -e "${GREEN}Our Language:${NC}"

    # First, measure warmup time separately (so we can subtract it)
    local our_warmup_file=$(mktemp /tmp/bench_warmup_our_XXXXXX.clj)
    cat > "$our_warmup_file" << EOF
$full_code
(def bench-target (fn [] $full_code))
(loop [i 0]
  (if (< i $warmup)
    (do
      (bench-target)
      (recur (+ i 1)))
    nil))
EOF

    echo "  Running warmup ($warmup iterations)..."
    local warmup_start warmup_end warmup_elapsed
    if command -v gdate &> /dev/null; then
        warmup_start=$(gdate +%s%N)
        "$OUR_LANG" "$our_warmup_file" > /dev/null 2>&1
        warmup_end=$(gdate +%s%N)
        warmup_elapsed=$((warmup_end - warmup_start))
    else
        warmup_start=$(python3 -c 'import time; print(int(time.time() * 1000000000))')
        "$OUR_LANG" "$our_warmup_file" > /dev/null 2>&1
        warmup_end=$(python3 -c 'import time; print(int(time.time() * 1000000000))')
        warmup_elapsed=$((warmup_end - warmup_start))
    fi

    echo "  Measuring ($iterations iterations)..."

    # Now measure full (warmup + measured)
    local full_start full_end full_elapsed
    if command -v gdate &> /dev/null; then
        full_start=$(gdate +%s%N)
        "$OUR_LANG" "$our_warm" > /dev/null 2>&1
        full_end=$(gdate +%s%N)
        full_elapsed=$((full_end - full_start))
    else
        full_start=$(python3 -c 'import time; print(int(time.time() * 1000000000))')
        "$OUR_LANG" "$our_warm" > /dev/null 2>&1
        full_end=$(python3 -c 'import time; print(int(time.time() * 1000000000))')
        full_elapsed=$((full_end - full_start))
    fi

    # The measured portion is approximately: full - warmup + startup
    # But startup is included in both, so: measured_ns ~= (full - warmup) only for the loop portion
    # Actually, a cleaner approach: just measure both loops in one file and report total
    # Since we're comparing apples to apples (both languages run same # of iterations after warmup)

    local measured_ns=$((full_elapsed - warmup_elapsed))
    # If measured_ns is negative (due to variance), just use a simple approach
    if [ "$measured_ns" -lt 0 ]; then
        measured_ns=$full_elapsed
    fi

    local our_elapsed_ms=$(echo "scale=2; $measured_ns / 1000000" | bc)
    local our_per_iter_us=$(echo "scale=2; $measured_ns / 1000 / $iterations" | bc)

    echo -e "  Total (measured): ${GREEN}${our_elapsed_ms} ms${NC}"
    echo -e "  Per iteration: ${GREEN}${our_per_iter_us} µs${NC}"

    rm -f "$our_warmup_file"

    echo ""
    echo -e "${YELLOW}Clojure:${NC}"
    echo "  Running warmup and measuring..."

    # Run Clojure benchmark (it handles timing internally)
    clojure "$clj_warm" 2>/dev/null | sed 's/^/  /'

    echo ""
    echo -e "${BLUE}───────────────────────────────────────────────────────────${NC}"
    echo -e "${BLUE}  Note: Warm benchmarks measure steady-state performance${NC}"
    echo -e "${BLUE}  after JIT warmup. Compare 'per iteration' times.${NC}"
    echo -e "${BLUE}───────────────────────────────────────────────────────────${NC}"
}

# Parse arguments
ITERATIONS=""
WARMUP=""
IS_FILE="false"
MODE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -n)
            ITERATIONS="$2"
            shift 2
            ;;
        -w)
            WARMUP="$2"
            shift 2
            ;;
        -f)
            IS_FILE="true"
            shift
            ;;
        -c)
            IS_FILE="false"
            shift
            ;;
        -h|--help)
            usage
            ;;
        cold|warm)
            MODE="$1"
            shift
            ;;
        *)
            if [ -z "$MODE" ]; then
                echo -e "${RED}Error: Must specify mode (cold or warm)${NC}"
                usage
            fi
            CODE="$1"
            shift
            ;;
    esac
done

if [ -z "$MODE" ] || [ -z "$CODE" ]; then
    usage
fi

# Check dependencies
check_build
check_clojure

# Run appropriate benchmark
case $MODE in
    cold)
        ITERATIONS="${ITERATIONS:-10}"
        benchmark_cold "$CODE" "$IS_FILE" "$ITERATIONS"
        ;;
    warm)
        ITERATIONS="${ITERATIONS:-1000}"
        WARMUP="${WARMUP:-100}"
        benchmark_warm "$CODE" "$IS_FILE" "$ITERATIONS" "$WARMUP"
        ;;
    *)
        echo -e "${RED}Unknown mode: $MODE${NC}"
        usage
        ;;
esac
