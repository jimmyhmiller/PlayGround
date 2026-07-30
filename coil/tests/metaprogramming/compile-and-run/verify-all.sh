#!/usr/bin/env bash
# THE FULL PROOF, one command. Run from the repo root:
#   metaprog-poc/compile-and-run/verify-all.sh [coil-binary]
#
# 1. rebootstrap  — the compiler rebuilds ITSELF through the compiled metaprogram
#                   engine (its default): FIXPOINT byte-identical + gate-full
#                   byte-exact + arm64 runtime gate.
# 2. (retired)   — engine parity: the interpreter was deleted (decision 7), so the
#                  compiled engine is the only engine and there is nothing to diff
#                   between the compiled engine and the interpreter.
# 3. run.sh       — every mechanism end to end: dylib callbacks, compiled
#                   metaprograms over real Sexp, the codelib vocabulary, the
#                   TOWER (macros in macro bodies, both engines), ARBITRARY code
#                   (generics/HashMap/StrBuf/malloc/libc-FFI at expansion time),
#                   and the compile-time Mandelbrot GUI on the main thread.
set -uo pipefail
cd "$(dirname "$0")/../.."
COIL=${1:-./coil}

echo "=========== 1/3 rebootstrap (fixpoint + gates, engine = default) ==========="
./selfhost/rebootstrap.sh >/tmp/verify-reboot.log 2>&1 || { tail -5 /tmp/verify-reboot.log; echo "REBOOTSTRAP FAILED"; exit 1; }
tail -6 /tmp/verify-reboot.log

echo "=========== 2/3 engine parity — retired (single engine since decision 7) ==========="

echo "=========== 3/3 all mechanisms (run.sh) ==========="
metaprog-poc/compile-and-run/run.sh "$COIL" || { echo "MECHANISMS FAILED"; exit 1; }

echo ""
echo "=========== EVERYTHING VERIFIED ==========="
