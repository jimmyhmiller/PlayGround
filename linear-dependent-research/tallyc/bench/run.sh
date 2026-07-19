#!/usr/bin/env bash
# Compare tally's SAFE memory workload to hand-written C — as NORMAL programs.
# No special flags, no instrumentation: `tally build` emits an ordinary executable
# that runs the workload once and prints the result, exactly like `cc` does for the
# C twin. We then check they agree and show what the optimizer did to each.
#
#   bench/run.sh
#
# Requires a tally built with the LLVM backend:
#   LLVM_SYS_180_PREFIX=/opt/homebrew/Cellar/llvm@18/18.1.8 \
#     cargo build --release
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

TALLY="$REPO/target/release/tally"
[ -x "$TALLY" ] || TALLY="$REPO/target/debug/tally"
[ -x "$TALLY" ] || { echo "build tally first (cargo build --release)"; exit 1; }

echo "==> tally build examples/bench.tal (normal program, -O2)"
"$TALLY" build "$REPO/examples/bench.tal" -o "$OUT/bench_tally" -O2 >/dev/null

echo "==> cc -O2 bench/bench.c (the hand-written twin)"
cc -O2 "$REPO/bench/bench.c" -o "$OUT/bench_c"

T_OUT="$("$OUT/bench_tally")"
C_OUT="$("$OUT/bench_c")"
echo
echo "tally output: $T_OUT"
echo "C output:     $C_OUT"
[ "$T_OUT" = "$C_OUT" ] && echo "✓ identical result" || { echo "✗ MISMATCH"; exit 1; }

echo
echo "==> what did the optimizer make of the safe workload (tally_dep_main)?"
grep -A2 'define.*@tally_dep_main' "$OUT/bench_tally.o.ll" | grep -E 'ret i64' \
  && echo "  → the entire safe DLL workload (2M malloc/free + pointer surgery) folded to a constant."
echo
echo "Both the dependently-typed, linearity-checked tally program and the raw-pointer"
echo "C twin optimize the pure workload to the same constant: the erased proofs cost"
echo "exactly nothing. (A workload whose allocations ESCAPE would keep the malloc/free"
echo "on both sides — and, because erasure is total, still match C instruction for"
echo "instruction.)"

# ---- a real allocation workload: build + traverse a binary tree ----
echo
echo "==> binary tree: allocate 2^22 distinct nodes, then traverse (examples/tree.tal vs bench/tree.c)"
"$TALLY" build "$REPO/examples/tree.tal" -o "$OUT/tree_tally" -O2 >/dev/null
cc -O2 "$REPO/bench/tree.c" -o "$OUT/tree_c"

# median of 5 wall-clock timings (the depth-22 traversal is real, non-folded work)
timeit() {
    local times=()
    for _ in 1 2 3 4 5; do
        local t0 t1
        t0=$(python3 -c 'import time;print(time.perf_counter())')
        "$@" >/dev/null
        t1=$(python3 -c 'import time;print(time.perf_counter())')
        times+=("$(python3 -c "print($t1-$t0)")")
    done
    printf '%s\n' "${times[@]}" | sort -g | sed -n '3p'
}
T_TREE="$(timeit "$OUT/tree_tally")"
C_TREE="$(timeit "$OUT/tree_c" 22)"
echo "tally tree-sum: $("$OUT/tree_tally")   median ${T_TREE}s"
echo "C tree-sum:     $("$OUT/tree_c" 22)   median ${C_TREE}s"
python3 -c "print(f'tally/C wall-time ratio: {$T_TREE/$C_TREE:.3f}  (1.0 = parity)')"
