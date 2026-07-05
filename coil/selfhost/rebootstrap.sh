#!/usr/bin/env bash
# THE EASY BOOTSTRAP — rebuild and VERIFY the self-host Coil compiler with NO Rust toolchain.
#
# stage0 is chosen automatically:
#   1. $STAGE0 if you set it explicitly
#   2. ./target/debug/coil  (the Rust reference, if you happen to have built it)
#   3. selfhost/seed/coil-seed  (the committed, prebuilt self-host compiler)
# On a fresh checkout only #3 exists — which is the whole point: you never need cargo/rustc/inkwell.
#
# The seed is NEVER trusted blindly. Every run re-derives the compiler from source and proves
# the result faithful two independent ways, so a stale or tampered seed cannot slip through:
#   * FIXPOINT : stage2.o == stage3.o byte-identical  (the arm64 backend is fully deterministic)
#   * GATES    : gate-full  (emitted IR byte-exact vs the reference snapshot, whole corpus)
#                arm64 gate-run  (built programs produce identical stdout+exit)
#
# Requirements: libLLVM.dylib (brew install llvm) + a C compiler (cc). That's it.
# (The compiler embeds an LLVM backend, so its binary links libLLVM even when the arm64
#  backend does the codegen. Only the Rust *build* toolchain is eliminated, not libLLVM.)
#
# Usage: selfhost/rebootstrap.sh [install-dest]      (default dest: ./coil)
#        STAGE0=/path/to/coil selfhost/rebootstrap.sh
set -uo pipefail
cd "$(dirname "$0")/.."                 # repo root
SRC=selfhost/src/main.coil
SEED=selfhost/seed/coil-seed
LF=(--link-flag -L/opt/homebrew/opt/llvm/lib --link-flag -lLLVM)

if   [ -n "${STAGE0:-}" ];        then :
elif [ -x ./target/debug/coil ];  then STAGE0=./target/debug/coil
elif [ -x "$SEED" ];              then STAGE0="$SEED"
else echo "no stage0: need ./target/debug/coil (cargo build) or a committed $SEED"; exit 1
fi
echo "stage0 = $STAGE0"

echo "=== stage1: stage0 builds the self-host compiler (default LLVM backend) ==="
"$STAGE0"     build "$SRC" -o /tmp/coil-rb1                "${LF[@]}" || { echo "stage1 FAILED"; exit 1; }
echo "=== stage2: stage1 rebuilds it with --backend arm64 ==="
/tmp/coil-rb1 build "$SRC" -o /tmp/coil-rb2 --backend arm64 "${LF[@]}" || { echo "stage2 FAILED"; exit 1; }
echo "=== stage3: stage2 rebuilds it with --backend arm64 ==="
/tmp/coil-rb2 build "$SRC" -o /tmp/coil-rb3 --backend arm64 "${LF[@]}" || { echo "stage3 FAILED"; exit 1; }

echo "=== FIXPOINT: stage2.o vs stage3.o ==="
cmp /tmp/coil-rb2.o /tmp/coil-rb3.o || { echo "FIXPOINT FAIL — arm64 objects differ (nondeterminism)"; exit 2; }
echo "  ok — byte-identical, the compiler reproduces itself"

echo "=== GATES ==="
./selfhost/oracle/gate-full.sh /tmp/coil-rb2 >/dev/null      || { echo "gate-full FAIL — not a faithful compiler"; exit 1; }
echo "  gate-full:      PASS (IR byte-exact vs reference)"
./selfhost/oracle/arm64/gate-run.sh /tmp/coil-rb2 >/dev/null || { echo "arm64 gate-run FAIL — runtime divergence"; exit 1; }
echo "  arm64 gate-run: PASS (programs run identically)"

DEST="${1:-./coil}"
cp /tmp/coil-rb2 "$DEST"
# Re-sign after copy: macOS invalidates a Mach-O's ad-hoc signature on cp, and the
# kernel SIGKILLs a mis-signed binary. Re-sign so the installed compiler runs.
codesign -s - --force "$DEST" >/dev/null 2>&1 || true
echo "=== VERIFIED self-host compiler installed -> $DEST ==="
