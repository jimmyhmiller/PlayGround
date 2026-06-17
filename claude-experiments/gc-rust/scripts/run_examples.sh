#!/usr/bin/env bash
# Run every example program and assert it produces its known-good result. A
# local regression check (no CI required). Pass `--gc-stress` to additionally
# collect at every allocation, proving the GC stays correct under relocation.
#
#   ./scripts/run_examples.sh            # normal run
#   ./scripts/run_examples.sh --gc-stress
set -uo pipefail
cd "$(dirname "$0")/.."

STRESS=""
if [ "${1:-}" = "--gc-stress" ]; then
  STRESS="--gc-stress"
fi

# example name -> expected final stdout line (the program's i64 result)
declare -A EXPECT=(
  [atom]=20000
  [channel]=109900
  [binary_trees]=5242840
  [ffi]=1066
  [ffi_struct]=1
  [ffi_bytes]=42
  [ffi_buffer]=14
  [ffi_callback]=1050
  [fib]=2178309
  [mandelbrot]=86906
  [match]=47
  [mutability]=33
  [nbody]=921463
  [nbody_vec3]=1457652585
  [prelude_demo]=42
  [shapes]=47
  [stdlib]=414
  [threads]=37492500
  [strings]=35
  [types]=206
  [vec]=386
  [vec_prelude]=285
)

# Build once so the per-example runs don't each recompile.
cargo build --quiet --bin gcr || { echo "build failed"; exit 1; }
GCR="./target/debug/gcr"

fail=0
for name in "${!EXPECT[@]}"; do
  want="${EXPECT[$name]}"
  got=$("$GCR" run "examples/$name.gcr" $STRESS 2>/dev/null | tail -1)
  if [ "$got" = "$want" ]; then
    printf "  ok   %-16s = %s\n" "$name" "$got"
  else
    printf "  FAIL %-16s expected %s, got %s\n" "$name" "$want" "$got"
    fail=1
  fi
done

# The multi-file project example (driven by its directory entry).
got=$("$GCR" run examples/modproj/main.gcr $STRESS 2>/dev/null | tail -1)
if [ "$got" = "32" ]; then
  printf "  ok   %-16s = %s\n" "modproj" "$got"
else
  printf "  FAIL %-16s expected 32, got %s\n" "modproj" "$got"
  fail=1
fi

if [ "$fail" = "0" ]; then
  echo "all examples passed${STRESS:+ (under --gc-stress)}"
else
  echo "some examples FAILED"
fi
exit $fail
