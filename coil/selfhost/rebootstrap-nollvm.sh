#!/usr/bin/env bash
# THE LLVM-FREE BOOTSTRAP — rebuild + verify the self-host Coil compiler with NO
# LLVM and NO Rust toolchain. The produced compiler (from selfhost/src/main_a64.coil)
# omits the LLVM backend entirely: it links only libSystem and needs only `cc` to
# link native objects. Nothing here touches libLLVM at build or run time.
#
# stage0 is chosen automatically:
#   1. $STAGE0 if set
#   2. selfhost/seed/coil-seed-nollvm   (the committed LLVM-free self-host compiler)
#   (the Rust reference compiler has been removed; the seed is fully self-sufficient)
# On a fresh checkout only #2 exists — the point: no cargo/rustc/inkwell AND no libLLVM.
#
# The seed is re-verified from source on every run, three independent ways, so a
# stale/tampered seed cannot slip through:
#   * NO-LLVM : otool proves stage2 links no libLLVM
#   * FIXPOINT: stage2.o == stage3.o byte-identical (arm64 backend is deterministic)
#   * GATE    : arm64 gate-run — every corpus program runs identically to the
#               LLVM-reference. (gate-full/emit-ir is N/A: this build has no LLVM IR.)
#
# Requirements: a C compiler (cc). That's the whole toolchain.
#
# Usage: selfhost/rebootstrap-nollvm.sh [install-dest]     (default dest: ./coil-nollvm)
#        STAGE0=/path/to/coil selfhost/rebootstrap-nollvm.sh
set -uo pipefail
cd "$(dirname "$0")/.."                 # repo root
SRC=selfhost/src/main_a64.coil
SEED=selfhost/seed/coil-seed-nollvm

if   [ -n "${STAGE0:-}" ];        then :
elif [ -x "$SEED" ];              then STAGE0="$SEED"
else echo "no stage0: need a committed $SEED (or set STAGE0=/path/to/coil)"; exit 1
fi
echo "stage0 = $STAGE0"

echo "=== stage1: stage0 builds the LLVM-free compiler ==="
"$STAGE0"     build "$SRC" -o /tmp/coil-nl1                 || { echo "stage1 FAILED"; exit 1; }
echo "=== stage2: stage1 rebuilds it with --backend arm64 ==="
/tmp/coil-nl1 build "$SRC" -o /tmp/coil-nl2 --backend arm64 || { echo "stage2 FAILED"; exit 1; }
echo "=== stage3: stage2 rebuilds it with --backend arm64 ==="
/tmp/coil-nl2 build "$SRC" -o /tmp/coil-nl3 --backend arm64 || { echo "stage3 FAILED"; exit 1; }

echo "=== NO-LLVM: stage2 must link no libLLVM ==="
if otool -L /tmp/coil-nl2 | grep -qi LLVM; then
  echo "  FAIL — libLLVM is linked:"; otool -L /tmp/coil-nl2 | grep -i LLVM; exit 3
fi
echo "  ok — links only:$(otool -L /tmp/coil-nl2 | tail -n +2 | awk '{printf " %s", $1}')"

echo "=== FIXPOINT: stage2.o vs stage3.o ==="
cmp /tmp/coil-nl2.o /tmp/coil-nl3.o || { echo "FIXPOINT FAIL — arm64 objects differ"; exit 2; }
echo "  ok — byte-identical, the compiler reproduces itself"

echo "=== GATE: arm64 behavioral gate-run ==="
./selfhost/oracle/arm64/gate-run.sh /tmp/coil-nl2 >/dev/null 2>&1 || { echo "arm64 gate-run FAIL"; exit 1; }
echo "  arm64 gate-run: PASS (programs run identically to the LLVM reference)"

DEST="${1:-./coil-nollvm}"
cp /tmp/coil-nl2 "$DEST"
echo "=== VERIFIED LLVM-free compiler installed -> $DEST ==="
