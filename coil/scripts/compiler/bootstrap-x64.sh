#!/usr/bin/env bash
# Self-host the compiler THROUGH THE NATIVE x86-64 BACKEND, and prove it.
#
# The x86-64 sibling of the arm64 bootstrap. Three independent checks, because
# "it compiled" is not evidence that a backend is correct:
#
#   FIXPOINT : stage2.o == stage3.o byte-identical. A compiler built by the x64
#              backend, rebuilding itself with the x64 backend, must land on the
#              same bytes. This catches nondeterminism and most miscompiles of
#              the compiler's own code — a subtly wrong stage2 almost never
#              reproduces itself exactly.
#   GATE     : the x64-built compiler passes the behavioral corpus. It is not
#              enough that it runs; it has to be a working compiler.
#   CABI     : struct passing/returning agrees with a gcc-compiled object. The
#              eightbyte rules only bite at the C boundary (internal calls pass
#              aggregates by pointer), so this is the only check that reaches
#              them — and the bug that broke the first self-host lived here.
#   ENCODE   : every instruction the encoder emits still matches llvm-mc.
#
# stage1 comes from whatever compiler you point at (default build/bin/coil, the
# LLVM-backed build) — only stages 2 and 3 go through the x64 backend.
#
# Usage: scripts/compiler/bootstrap-x64.sh [install-dest]   (default dest: build/bin/coil-x64)
#        STAGE0=/path/to/coil scripts/compiler/bootstrap-x64.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
SRC=src/compiler/main.coil

STAGE0="${STAGE0:-build/bin/coil}"
[ -x "$STAGE0" ] || { echo "no stage0: $STAGE0 is not executable (set STAGE0=/path/to/coil)"; exit 1; }
echo "stage0 = $STAGE0"

# libLLVM link line: main.coil still CONTAINS the LLVM backend (this bootstrap
# proves the x64 backend can BUILD it, not that the result is LLVM-free — that
# is rebootstrap-nollvm-linux.sh, which builds main_x64.coil instead).
libdir="${COIL_LLVM_LIBDIR:-}"
if [ -z "$libdir" ]; then
  for d in /usr/src/stdlib/llvm-21/lib /usr/src/stdlib/x86_64-linux-gnu; do
    [ -e "$d/libLLVM.so" ] || [ -e "$d/libLLVM-21.so" ] && { libdir="$d"; break; }
  done
fi
[ -n "$libdir" ] || { echo "no libLLVM.so found (set COIL_LLVM_LIBDIR)"; exit 1; }
LF=(--link-flag "-L$libdir" --link-flag "-Wl,-rpath,$libdir" --link-flag -lLLVM
    --link-flag -lstdc++ --link-flag -lm --link-flag -lpthread --link-flag -ldl)

echo "=== stage1: stage0 builds the compiler (its own backend) ==="
"$STAGE0" build "$SRC" -o /tmp/coil-x64-s1 "${LF[@]}" || { echo "stage1 FAILED"; exit 1; }
echo "=== stage2: stage1 rebuilds it with --backend x64 ==="
/tmp/coil-x64-s1 build "$SRC" -o /tmp/coil-x64-s2 --backend x64 "${LF[@]}" || { echo "stage2 FAILED"; exit 1; }
echo "=== stage3: stage2 rebuilds it with --backend x64 ==="
/tmp/coil-x64-s2 build "$SRC" -o /tmp/coil-x64-s3 --backend x64 "${LF[@]}" || { echo "stage3 FAILED"; exit 1; }

echo "=== FIXPOINT: stage2.o vs stage3.o ==="
cmp /tmp/coil-x64-s2.o /tmp/coil-x64-s3.o || { echo "FIXPOINT FAIL — x64 objects differ"; exit 2; }
echo "  ok — byte-identical, the compiler reproduces itself through the x64 backend"

echo "=== GATE: the x64-built compiler must itself pass the corpus ==="
python3 scripts/oracle.py runtime gate x64 --compiler /tmp/coil-x64-s2 >/dev/null 2>&1 \
  || { echo "x64 gate-run FAIL (run it directly to see which programs)"; exit 1; }
echo "  ok — every corpus program runs identically to the LLVM reference"

echo "=== CABI: the SysV rules agree with a gcc-compiled translation unit ==="
./scripts/compiler/oracle/x64/gate-cabi.sh /tmp/coil-x64-s2 >/dev/null 2>&1 \
  || { echo "x64 gate-cabi FAIL (run it directly to see which case)"; exit 1; }
echo "  ok — struct passing/returning matches gcc across the C boundary"

echo "=== ENCODE: the encoder still agrees with llvm-mc ==="
if command -v llvm-mc >/dev/null 2>&1; then
  ./scripts/compiler/oracle/x64/gate-encode.sh /tmp/coil-x64-s2 >/dev/null 2>&1 \
    || { echo "x64 gate-encode FAIL"; exit 1; }
  echo "  ok — every encoder case matches llvm-mc"
else
  echo "  skipped — no llvm-mc on PATH"
fi

DEST="${1:-build/bin/coil-x64}"
mkdir -p "$(dirname "$DEST")"
cp /tmp/coil-x64-s2 "$DEST"
echo "=== VERIFIED x64-self-hosted compiler installed -> $DEST ==="
