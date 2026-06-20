#!/usr/bin/env bash
# Benchmark Coil's compiled output against C (clang) on matched programs.
#
#   Coil  : coil build  (LLVM -O3 pipeline, the default since the optimizer landed)
#   C -O3 : the fair baseline -- same LLVM backend, same opt tier as Coil
#
# Each program prints one integer/float; the harness checks Coil and C agree
# before timing (a fast wrong answer is meaningless). Timing is via hyperfine.
#
#   bench/run.sh            # run every benchmark
#   bench/run.sh fib loop   # run a subset
set -euo pipefail
cd "$(dirname "$0")/.."
export LLVM_SYS_180_PREFIX="${LLVM_SYS_180_PREFIX:-/opt/homebrew/Cellar/llvm@18/18.1.8}"
CC="${CC:-cc}"
COIL=target/debug/coil

command -v hyperfine >/dev/null || { echo "need hyperfine (brew install hyperfine)"; exit 1; }
echo "building the coil compiler..."
cargo build -q

BENCHES=("$@"); [ ${#BENCHES[@]} -eq 0 ] && BENCHES=(fib tak loop float memory)
T="$(mktemp -d)"
RESULTS="bench/RESULTS.md"
{
  echo "# Coil vs C benchmarks"
  echo
  echo "Host: \`$(uname -msr)\`, $($CC --version | head -1)."
  echo "Coil = \`coil build\` (LLVM -O3). All programs verified to print identical results."
  echo
} > "$RESULTS"

for b in "${BENCHES[@]}"; do
  echo "=== $b ==="
  "$COIL" build "bench/$b.coil" -o "$T/$b.coil" >/dev/null
  "$CC" -O3 "bench/$b.c" -o "$T/$b.c3"

  oc=$("$T/$b.coil"); o3=$("$T/$b.c3")
  if [ "$oc" != "$o3" ]; then
    echo "MISMATCH in $b: coil=$oc  c-O3=$o3"; exit 1
  fi
  echo "result = $oc (coil == c-O3)"

  hyperfine -N --warmup 2 --runs 8 \
    -n "C -O3"    "$T/$b.c3" \
    -n "Coil -O3" "$T/$b.coil" \
    --export-markdown "$T/$b.md"

  { echo "## \`$b\`  (result \`$oc\`)"; echo; cat "$T/$b.md"; echo; } >> "$RESULTS"
done

echo
echo "wrote $RESULTS"
