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
# source with a native compiler, so this needs a working ./coil (or one passed
# as COIL=...). Steps, all with plain cc:
#   1. native coil  ->  main_wasm.coil  ->  coilc32.wasm   (wasm32 seed)
#   2. cc wasm2c.c  ->  wasm2c ; wasm2c coilc32.wasm coilc32.c little
#   3. cc coilc32.c runtime32.c -> coil-bootstrap32
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

echo "[1/3] native coil -> wasm32 seed (main_wasm.coil -> coilc32.wasm)"
[ -x "$COIL" ] || { echo "need a native coil at \$COIL ($COIL)"; exit 2; }
"$COIL" build "$ROOT/selfhost/src/main_wasm.coil" --target wasm32-unknown-unknown -o coilc32.wasm

echo "[2/3] building wasm2c translator + translating coilc32.wasm -> coilc32.c"
$CC -O2 -o wasm2c wasm2c.c
./wasm2c coilc32.wasm coilc32.c little

echo "[3/3] compiling coilc32.c + runtime32.c -> coil-bootstrap32 ($OPT)"
$CC $OPT -w -o coil-bootstrap32 coilc32.c runtime32.c -lm

echo "done: $(pwd)/coil-bootstrap32"
echo "verify:  ./coil-bootstrap32 build $ROOT/examples/fib.coil -o /tmp/fib && /tmp/fib; echo \$?   # -> 55"
