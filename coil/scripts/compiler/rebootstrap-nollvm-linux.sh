#!/usr/bin/env bash
# THE LLVM-FREE BOOTSTRAP, LINUX x86-64 — rebuild + verify the self-host Coil
# compiler with NO LLVM and NO Rust toolchain. The Linux sibling of
# rebootstrap-nollvm.sh (which does the same on macOS/arm64).
#
# The produced compiler (from src/compiler/main_x64.coil) omits the LLVM backend
# entirely: it links only libc/libm and needs only `cc` to link native objects.
# Nothing here touches libLLVM at build or run time.
#
# stage0 is chosen automatically:
#   1. $STAGE0 if set
#   2. bootstrap/seeds/native/coil-seed-nollvm-linux-x86_64  (the committed LLVM-free seed)
#   3. build/bin/coil                                       (the repository launcher)
#
# The result is re-verified from source on every run, three independent ways, so a
# stale/tampered seed cannot slip through:
#   * NO-LLVM : ldd proves stage2 links no libLLVM
#   * FIXPOINT: stage2.o == stage3.o byte-identical (the x64 backend is deterministic)
#   * GATE    : x64 gate-run — every corpus program runs identically to the
#               LLVM-reference. (gate-full/emit-ir is N/A: this build has no LLVM IR.)
#
# Requirements: a C compiler (cc). That's the whole toolchain.
#
# Usage: scripts/compiler/rebootstrap-nollvm-linux.sh [install-dest]  (default: build/bin/coil-nollvm)
#        STAGE0=/path/to/coil scripts/compiler/rebootstrap-nollvm-linux.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
SRC=src/compiler/main_x64.coil
SEED=bootstrap/seeds/native/coil-seed-nollvm-linux-x86_64

if   [ -n "${STAGE0:-}" ];   then :
elif [ -x "$SEED" ];         then STAGE0="$SEED"
elif [ -x build/bin/coil ];          then STAGE0=build/bin/coil
else echo "no stage0: need $SEED or build/bin/coil (or set STAGE0=/path/to/coil)"; exit 1
fi
echo "stage0 = $STAGE0"

# stage1 may come from the LLVM-backed compiler, whose own `build` defaults to the
# LLVM backend and therefore needs the libLLVM link line. Once stage1 exists it is
# LLVM-free and every later stage links nothing extra.
S1FLAGS=()
if ldd "$STAGE0" 2>/dev/null | grep -qi llvm; then
  libdir="${COIL_LLVM_LIBDIR:-}"
  if [ -z "$libdir" ]; then
    for d in /usr/src/stdlib/llvm-21/lib /usr/src/stdlib/x86_64-linux-gnu; do
      { [ -e "$d/libLLVM.so" ] || [ -e "$d/libLLVM-21.so" ]; } && { libdir="$d"; break; }
    done
  fi
  [ -n "$libdir" ] || { echo "stage0 needs libLLVM but none found (set COIL_LLVM_LIBDIR)"; exit 1; }
  S1FLAGS=(--link-flag "-L$libdir" --link-flag "-Wl,-rpath,$libdir" --link-flag -lLLVM
           --link-flag -lstdc++ --link-flag -lm --link-flag -lpthread --link-flag -ldl)
fi

echo "=== stage1: stage0 builds the LLVM-free compiler ==="
"$STAGE0"     build "$SRC" -o /tmp/coil-nlx1 "${S1FLAGS[@]}"  || { echo "stage1 FAILED"; exit 1; }
echo "=== stage2: stage1 rebuilds it with the x64 backend ==="
/tmp/coil-nlx1 build "$SRC" -o /tmp/coil-nlx2                 || { echo "stage2 FAILED"; exit 1; }
echo "=== stage3: stage2 rebuilds it with the x64 backend ==="
/tmp/coil-nlx2 build "$SRC" -o /tmp/coil-nlx3                 || { echo "stage3 FAILED"; exit 1; }

echo "=== NO-LLVM: stage2 must link no libLLVM ==="
if ldd /tmp/coil-nlx2 | grep -qi LLVM; then
  echo "  FAIL — libLLVM is linked:"; ldd /tmp/coil-nlx2 | grep -i LLVM; exit 3
fi
echo "  ok — links only:$(ldd /tmp/coil-nlx2 | awk '{printf " %s", $1}')"

echo "=== FIXPOINT: stage2.o vs stage3.o ==="
cmp /tmp/coil-nlx2.o /tmp/coil-nlx3.o || { echo "FIXPOINT FAIL — x64 objects differ"; exit 2; }
echo "  ok — byte-identical, the compiler reproduces itself"

echo "=== GATE: x64 behavioral gate-run ==="
python3 scripts/oracle.py runtime gate x64 --compiler /tmp/coil-nlx2 >/dev/null 2>&1 \
  || { echo "x64 gate-run FAIL (run it directly to see which programs)"; exit 1; }
echo "  x64 gate-run: PASS (programs run identically to the LLVM reference)"

DEST="${1:-build/bin/coil-nollvm}"
mkdir -p "$(dirname "$DEST")"
cp /tmp/coil-nlx2 "$DEST"
echo "=== VERIFIED LLVM-free compiler installed -> $DEST ==="
