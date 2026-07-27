#!/bin/sh
# wasm32 (MVP) C bootstrap for the Coil compiler — the sibling of build.sh.
#
# build.sh bootstraps the committed memory64 seed (coilc.wasm) with runtime.c.
# This script bootstraps a *wasm32* build with runtime32.c. Its purpose is
# width-correctness verification: a machine word wrongly narrowed to i32 in the
# compiler is invisible on 64-bit native (isize==i64) but MISBEHAVES on wasm32
# (isize==i32). Running a full wasm32 compiler end to end catches that — the
# build can't.
#
# Unlike build.sh there is no committed wasm32 seed; we regenerate one from
# source. The catch that this must get right: the committed ./coil is a bootstrap
# SEED and is frequently STALE relative to selfhost/src — using it DIRECTLY to
# emit coilc32.wasm bakes in whatever the seed's source said, not the current
# tree. That is exactly how the 1 MiB→256 MiB shadow-stack fix (wasm.coil,
# wasm-stack-size) went missing: a pre-fix ./coil emits a 1 MiB shadow stack, the
# wasm-hosted compiler overflows it into the data/heap, and it mis-parses valid
# code ("nests more than 2000 levels", garbled symbols) — a phantom "wasm32-only
# corruption" that is really a stale-seed artifact, not a width bug.
#
# The fix: FIRST rebuild a CURRENT native compiler from source with $COIL (a stale
# seed still compiles current source into a correct current compiler — output is a
# function of the source, not the seed), THEN use that fresh compiler to emit the
# wasm32 seed. So the verification always reflects the CURRENT tree. Steps:
#   0. $COIL       ->  main.coil      ->  coil-seed32     (a CURRENT native coil)
#   1. coil-seed32 ->  main_wasm.coil ->  coilc32.wasm    (wasm32 seed)
#   2. cc wasm2c.c ->  wasm2c ; wasm2c coilc32.wasm coilc32.c little
#   3. cc coilc32.c runtime32.c -> coil-bootstrap32
#
# Pass COIL_SEED32=/path/to/current-coil to skip step 0 (e.g. reuse a /tmp/coil-i
# you already built from the current tree).
#
# main_wasm.coil (not main_a64.coil) is the entry: its comptime is pure
# interpretation, so the wasm module never calls the dead thread/JIT/mmap
# imports. Verified: coil-bootstrap32 builds examples/fib.coil -> 55 and
# reproduces a __TEXT,__text byte-identical `main_a64.coil --backend arm64`.
set -e
cd "$(dirname "$0")"
ROOT=..

CC="${CC:-cc}"
COIL="${COIL:-$ROOT/coil}"
OPT="${OPT:--O1}"

# Step 0: obtain a CURRENT native compiler. Reuse $COIL_SEED32 if the caller built
# one from the current tree; otherwise rebuild it from source with $COIL so a stale
# committed ./coil can never leak old behavior (e.g. the 1 MiB shadow stack) into
# the wasm32 seed.
if [ -n "$COIL_SEED32" ] && [ -x "$COIL_SEED32" ]; then
    SEED="$COIL_SEED32"
    echo "[0/3] using caller-provided current compiler: $SEED"
else
    [ -x "$COIL" ] || { echo "need a native coil at \$COIL ($COIL)"; exit 2; }
    SEED="$(pwd)/coil-seed32"
    echo "[0/3] rebuilding a CURRENT native compiler from source (guards vs a stale \$COIL)"
    "$COIL" build "$ROOT/selfhost/src/main.coil" -o "$SEED" $("$ROOT/selfhost/llvm-link-flags.sh" dynamic)
fi

echo "[1/3] current coil -> wasm32 seed (main_wasm.coil -> coilc32.wasm)"
# The compiler recurses deep — opt into a larger shadow stack than the 16 MiB default
# (wasm.coil). 64 MiB self-builds main_a64 byte-identically (a 1 MiB stack once
# overflowed into the data segment and mis-parsed valid source).
"$SEED" build "$ROOT/selfhost/src/main_wasm.coil" --target wasm32-unknown-unknown --wasm-stack-size=64 -o coilc32.wasm

echo "[2/3] building wasm2c translator + translating coilc32.wasm -> coilc32.c"
$CC -O2 -o wasm2c wasm2c.c
./wasm2c coilc32.wasm coilc32.c little

echo "[3/3] compiling coilc32.c + runtime32.c -> coil-bootstrap32 ($OPT)"
$CC $OPT -w -o coil-bootstrap32 coilc32.c runtime32.c -lm

echo "done: $(pwd)/coil-bootstrap32"
echo "verify:  ./coil-bootstrap32 build $ROOT/examples/fib.coil -o /tmp/fib && /tmp/fib; echo \$?   # -> 55"
