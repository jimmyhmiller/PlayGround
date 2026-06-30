#!/usr/bin/env bash
# Compile-speed-at-scale: the honest §12 risk (docs/FUTURE_WORK.md) — whole-program
# monomorphization + a comptime macro/elaboration pass. The repo's largest program is
# ~450 LOC, so this risk has been UNMEASURED. This generates Coil programs of growing
# size and charts the compiler's wall-time, separating the front end (read → expand →
# check → mono → IR, via `emit-ir`) from a full optimized `build` (adds the LLVM -O3
# pipeline + link).
#
#   bench/compile_scale.sh                 # default sizes
#   bench/compile_scale.sh 500 2000 8000   # custom function counts
set -euo pipefail
cd "$(dirname "$0")/.."
export LLVM_SYS_211_PREFIX="${LLVM_SYS_211_PREFIX:-$(echo /opt/homebrew/Cellar/llvm@21/* 2>/dev/null | tr ' ' '\n' | tail -1)}"
COIL=target/release/coil
echo "building the coil compiler (release-ish dev build)..."
cargo build --release -q

SIZES=("$@"); [ ${#SIZES[@]} -eq 0 ] && SIZES=(250 1000 4000 8000)
T="$(mktemp -d)"

# Generate a program with N functions. A mix that exercises the scaling risks:
#  - a reachable chain of plain functions (codegen volume),
#  - a generic `gid<k>` instantiated at i64 AND f64 every 10th function (mono blowup),
#  - a `(when …)` macro use every 5th function (the comptime elaborator).
# Every function is reachable from main (chained), so nothing is dead-stripped.
gen() { # <N> <out.coil>
  local n="$1" out="$2"
  {
    echo "(module scale)"
    echo "(defn g0 [(x :i64)] (-> :i64) x)"
    for ((i = 1; i < n; i++)); do
      local p=$((i - 1))
      if (( i % 10 == 0 )); then
        # generic, used at two types below — stresses monomorphization
        echo "(defn gid$i [T] [(v T)] (-> T) v)"
        echo "(defn g$i [(x :i64)] (-> :i64) (iadd (cast :i64 (gid$i (cast :f64 x))) (iadd (gid$i x) (g$p x))))"
      elif (( i % 5 == 0 )); then
        # macro use (when) — stresses the comptime macro elaborator
        echo "(defn g$i [(x :i64)] (-> :i64) (iadd x (when (icmp-gt x 0) (g$p x) 0)))"
      else
        echo "(defn g$i [(x :i64)] (-> :i64) (iadd (imul x $i) (g$p x)))"
      fi
    done
    echo "(defn main [] (-> :i64) (g$((n - 1)) 1))"
  } > "$out"
}

# Best (min) of 3 wall-clock seconds for: <cmd...>.
timeit() { # <cmd...>
  local t1 t2 t3
  t1=$(_one "$@"); t2=$(_one "$@"); t3=$(_one "$@")
  python3 -c "print(f'{min($t1,$t2,$t3):.3f}')"
}
# One timed run (seconds, float). The command must succeed.
_one() { # <cmd...>
  local s e
  s=$(python3 -c 'import time;print(time.time())')
  "$@" >/dev/null 2>&1 || { echo "compile FAILED: $*" >&2; exit 1; }
  e=$(python3 -c 'import time;print(time.time())')
  python3 -c "print($e-$s)"
}

RESULTS="bench/COMPILE_SCALE.md"
{
  echo "# Compile speed at scale"
  echo
  echo "Host: \`$(uname -msr)\`. Generated programs: a reachable chain of \`gN\` functions"
  echo "(~1 fn/line), 10% generic (instantiated at i64+f64), 20% using the \`when\` macro."
  echo "Best of 3 runs. **front-end** = \`coil emit-ir\` (read→expand→check→mono→IR, no"
  echo "optimizer); **build** = full \`coil build\` (adds the LLVM -O3 pipeline + link)."
  echo
  echo "| functions | LOC | front-end (s) | build (s) | front-end ms/fn | build ms/fn |"
  echo "|---:|---:|---:|---:|---:|---:|"
} > "$RESULTS"

printf "%8s %8s %12s %10s\n" "fns" "LOC" "front-end(s)" "build(s)"
for n in "${SIZES[@]}"; do
  src="$T/scale_$n.coil"
  gen "$n" "$src"
  loc=$(wc -l < "$src" | tr -d ' ')
  fe=$(timeit "$COIL" emit-ir "$src")
  bd=$(timeit "$COIL" build "$src" -o "$T/scale_$n")
  fems=$(python3 -c "print(f'{1000*$fe/$n:.3f}')")
  bdms=$(python3 -c "print(f'{1000*$bd/$n:.3f}')")
  printf "%8s %8s %12s %10s\n" "$n" "$loc" "$fe" "$bd"
  echo "| $n | $loc | $fe | $bd | $fems | $bdms |" >> "$RESULTS"
done

echo
echo "wrote $RESULTS"
