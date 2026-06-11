#!/bin/bash
# Benchmark ai-lang against Rust and Go.
#
# Each program self-times its core workload (JIT/compile/startup time is
# excluded in every language) and prints:  RESULT <name> <ms> ms checksum=<n>
# Checksums must match across all languages or the run is rejected.
#
# Usage:  benchmarks/run.sh [runs]      (default 3 runs, best-of reported)

set -euo pipefail
cd "$(dirname "$0")"
RUNS="${1:-3}"

AI_LANG=../target/release/ai-lang
BENCHES=(fib loop_mix mandelbrot nbody binary_trees)

echo "building ai-lang (release)..."
(cd .. && cargo build --release 2>&1 | tail -1)
echo "building rust benchmarks (release)..."
(cd rust && cargo build --release 2>&1 | tail -1)
echo "building go benchmarks..."
mkdir -p go/bin
for b in "${BENCHES[@]}"; do
    (cd go && go build -o "bin/$b" "./$b")
done

# Ingest each .ail into its own throwaway codebase (cheap, ~quarter second).
for b in "${BENCHES[@]}"; do
    rm -rf ".cb-$b"
    AI_LANG_CODEBASE=".cb-$b" "$AI_LANG" add "ail/$b.ail" > /dev/null
done

# run_one <cmd...> -> echoes "ms checksum"
run_one() {
    local out
    out="$("$@")"
    echo "$out" | sed -n 's/^RESULT [a-z_]* \([0-9-]*\) ms checksum=\(-\{0,1\}[0-9]*\)$/\1 \2/p'
}

printf "\n%-14s %12s %10s %10s %9s %9s\n" \
    "benchmark" "ai-lang(ms)" "rust(ms)" "go(ms)" "vs rust" "vs go"
printf "%-14s %12s %10s %10s %9s %9s\n" \
    "---------" "-----------" "--------" "------" "-------" "-----"

for b in "${BENCHES[@]}"; do
    best_ail=""; best_rust=""; best_go=""
    chk_ail=""; chk_rust=""; chk_go=""
    for ((r = 0; r < RUNS; r++)); do
        read -r ms chk <<< "$(AI_LANG_CODEBASE=".cb-$b" run_one "$AI_LANG" run main)"
        [[ -z "$best_ail" || "$ms" -lt "$best_ail" ]] && best_ail="$ms"
        chk_ail="$chk"
        read -r ms chk <<< "$(run_one "rust/target/release/$b")"
        [[ -z "$best_rust" || "$ms" -lt "$best_rust" ]] && best_rust="$ms"
        chk_rust="$chk"
        read -r ms chk <<< "$(run_one "go/bin/$b")"
        [[ -z "$best_go" || "$ms" -lt "$best_go" ]] && best_go="$ms"
        chk_go="$chk"
    done
    if [[ "$chk_ail" != "$chk_rust" || "$chk_ail" != "$chk_go" ]]; then
        echo "ERROR: $b checksum mismatch: ai-lang=$chk_ail rust=$chk_rust go=$chk_go" >&2
        exit 1
    fi
    r_rust=$(awk -v a="$best_ail" -v r="$best_rust" \
        'BEGIN { if (r == 0) print "inf"; else printf "%.2fx", a / r }')
    r_go=$(awk -v a="$best_ail" -v g="$best_go" \
        'BEGIN { if (g == 0) print "inf"; else printf "%.2fx", a / g }')
    printf "%-14s %12s %10s %10s %9s %9s   checksum=%s\n" \
        "$b" "$best_ail" "$best_rust" "$best_go" "$r_rust" "$r_go" "$chk_ail"
done
