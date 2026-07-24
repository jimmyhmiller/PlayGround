#!/bin/sh
# Platform-agnostic C bootstrap for the Coil compiler.
#
# Turns the committed memory64 seed `coilc.wasm` into a native `coil-bootstrap`
# executable using nothing but a C compiler:
#   1. build our extended wasm2c translator   (cc wasm2c.c)
#   2. translate coilc.wasm -> coilc.c        (self-contained C, no deps)
#   3. compile coilc.c + runtime.c            (cc -> coil-bootstrap)
#
# No Node, no Rust, no wasm engine. See README.md.
set -e
cd "$(dirname "$0")"

CC="${CC:-cc}"
WASM="${1:-coilc.wasm}"
OPT="${OPT:--O1}"

echo "[1/3] building wasm2c translator"
$CC -O2 -o wasm2c wasm2c.c

echo "[2/3] translating $WASM -> coilc.c"
./wasm2c "$WASM" coilc.c little

echo "[3/3] compiling coilc.c + runtime.c -> coil-bootstrap ($OPT)"
# The generated coilc.c is large (~900k lines); -O1 keeps compile time sane.
$CC $OPT -w -o coil-bootstrap coilc.c runtime.c -lm

echo "done: $(pwd)/coil-bootstrap"
