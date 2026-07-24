#!/usr/bin/env bash
# Keeps BOTH comptime engines alive and honest, so the eventual choice between them
# (or shipping both) stays open:
#   * the DEFAULT compiled metaprogram engine (metaengine.coil -> dylib/JIT), and
#   * the INTERPRETER engine (COIL_META_INTERP=1 -> interp.coil runs the metaprogram).
# Either can rot silently if only one is exercised — the standard gates only run the
# default. This gate proves the two are INTERCHANGEABLE:
#   [1] the interpreter engine passes the whole behavioral corpus (reuse arm64/gate-run
#       under the flag: build each corpus program with interp-meta, run, diff stdout+exit
#       against the SAME LLVM reference), and
#   [2] both engines compile main_a64.coil to a byte-identical compiler (__text) — a
#       divergence in EITHER engine's metaprogram results changes the emitted code.
#
# usage: gate-meta-engines.sh <coil-self-bin>
set -uo pipefail
cd "$(dirname "$0")/../.."
BIN="${1:?usage: gate-meta-engines.sh <coil-self-bin>}"

echo "  [1/2] interp-meta behavioral corpus (COIL_META_INTERP=1 gate-run) ..."
COIL_META_INTERP=1 ./selfhost/oracle/arm64/gate-run.sh "$BIN" >/dev/null 2>&1 \
  || { echo "  FAIL: interp-meta diverges from the reference on the behavioral corpus"; exit 1; }

echo "  [2/2] compiled-meta and interp-meta build a byte-identical compiler ..."
"$BIN" build selfhost/src/main_a64.coil --backend arm64 -o /tmp/gme-compiled >/dev/null 2>&1 \
  || { echo "  FAIL: compiled-meta cannot self-build main_a64.coil"; exit 1; }
COIL_META_INTERP=1 "$BIN" build selfhost/src/main_a64.coil --backend arm64 -o /tmp/gme-interp >/dev/null 2>&1 \
  || { echo "  FAIL: interp-meta cannot self-build main_a64.coil"; exit 1; }
# otool -X strips address prefixes so the compare is of raw section bytes, not layout.
A=$(otool -X -s __TEXT __text /tmp/gme-compiled 2>/dev/null | shasum | awk '{print $1}')
B=$(otool -X -s __TEXT __text /tmp/gme-interp   2>/dev/null | shasum | awk '{print $1}')
[ -n "$A" ] && [ "$A" = "$B" ] \
  || { echo "  FAIL: compiled-meta and interp-meta produce DIFFERENT compilers (__text $A vs $B)"; exit 1; }
echo "  gate-meta-engines: PASS (interp-meta == compiled-meta: full corpus + byte-identical self-build)"
