#!/usr/bin/env bash
# Exercises the interp-meta-IN-WASM path so it can't rot: build the compiler to a
# wasm64 module from main_wasm.coil (comptime runs via interp.coil, no side-modules)
# and have the Node host run it to self-CHECK the whole compiler source in-sandbox.
# Asserts the litmus that makes the wasm bootstrap simple: it self-checks, calls
# meta_run_wasm ZERO times (comptime is pure interpretation), hits ZERO Wall-1 traps,
# and coilc.wasm is a SINGLE static module.
#
# Requires: node (v>=? memory64) + wasm-tools. Skips (exit 0) if either is absent, so
# it never blocks a toolchain-minimal bootstrap.
#
# usage: gate-wasm.sh <coil-self-bin>
set -uo pipefail
cd "$(dirname "$0")/../.."
BIN="${1:?usage: gate-wasm.sh <coil-self-bin>}"

command -v node >/dev/null 2>&1 || { echo "  gate-wasm: SKIP (no node)"; exit 0; }
command -v wasm-tools >/dev/null 2>&1 || { echo "  gate-wasm: SKIP (no wasm-tools)"; exit 0; }

W=/tmp/gate-wasm-coilc.wasm
# The self-hosted compiler recurses deep, so it opts into a larger shadow stack than
# the modest 16 MiB default (wasm.coil). 64 MiB self-builds main_a64 byte-identically.
"$BIN" build selfhost/src/main_wasm.coil --target wasm64-unknown-unknown --wasm-stack-size=64 -o "$W" >/dev/null 2>&1 \
  || { echo "  FAIL: cannot build the wasm compiler from main_wasm.coil"; exit 1; }
wasm-tools validate --features=memory64 "$W" >/dev/null 2>&1 \
  || { echo "  FAIL: coilc.wasm is not a valid memory64 module"; exit 1; }
mods=$(wasm-tools print "$W" 2>/dev/null | grep -cE '^\(module')
[ "$mods" = "1" ] || { echo "  FAIL: coilc.wasm is not a single static module ($mods modules)"; exit 1; }

ERR=/tmp/gate-wasm.err
COIL_WASM_META_TRACE=1 node wasm-host/run-coil-wasm.mjs "$W" check selfhost/src/main_a64.coil >/dev/null 2>"$ERR"
rc=$?
[ "$rc" -eq 0 ]                            || { echo "  FAIL: wasm self-check exit $rc"; tail -3 "$ERR"; exit 1; }
[ "$(grep -c meta_run_wasm "$ERR")" -eq 0 ] || { echo "  FAIL: comptime called meta_run_wasm (should be pure interpretation)"; exit 1; }
[ "$(grep -c WALL1 "$ERR")" -eq 0 ]         || { echo "  FAIL: hit a Wall-1 trap (mmap/dlopen/pthread) during self-check"; exit 1; }
echo "  gate-wasm: PASS (single static module; self-checks in-sandbox; 0 meta_run_wasm, 0 Wall-1 traps)"
