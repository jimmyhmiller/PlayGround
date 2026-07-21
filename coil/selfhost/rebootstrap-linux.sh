#!/usr/bin/env bash
# THE EASY BOOTSTRAP, Linux x86-64 edition — rebuild and VERIFY the self-host Coil
# compiler on an ELF host. Mirrors rebootstrap.sh's shape with two differences:
#
#   * every stage uses the DEFAULT (LLVM) backend — the native arm64 backend emits
#     Mach-O and never runs here, so the fixpoint is the LLVM-backend one
#     (stage2.o == stage3.o, byte-identical; the LLVM emission is deterministic).
#   * the gates are the Linux oracle: gate-full (IR byte-exact vs the Linux
#     snapshot in selfhost/oracle/linux/full-reference), gate-run (stdout+exit vs
#     the shared behavioral snapshot), and gate-cli.
#
# stage0 is chosen automatically:
#   1. $STAGE0 if you set it explicitly
#   2. selfhost/seed/coil-seed-linux-x86_64  (the committed ELF seed) — DEFAULT
#
# Requirements: cc/clang and libLLVM 21 (Ubuntu: apt.llvm.org llvm-21 packages).
# The libdir is discovered via llvm-config-21/llvm-config; override with
# COIL_LLVM_LIBDIR if yours lives elsewhere. If the committed seed's libLLVM
# doesn't match your system, rebuild a stage0 from the shipped IR instead — see
# selfhost/seed/linux-ir/NOTES.md.
#
# Usage: selfhost/rebootstrap-linux.sh [install-dest]   (default dest: ./coil-linux
#        — NOT ./coil, which is the committed macOS binary)
set -uo pipefail
cd "$(dirname "$0")/.."                 # repo root
SRC=selfhost/src/main.coil
SEED=selfhost/seed/coil-seed-linux-x86_64

libdir="${COIL_LLVM_LIBDIR:-}"
if [ -z "$libdir" ]; then
  for lc in llvm-config-21 /usr/lib/llvm-21/bin/llvm-config llvm-config; do
    if command -v "$lc" >/dev/null 2>&1; then libdir="$("$lc" --libdir)"; break; fi
  done
fi
if [ -z "$libdir" ] || [ ! -e "$libdir/libLLVM.so" ]; then
  echo "no libLLVM.so found (install LLVM 21 from apt.llvm.org, or set COIL_LLVM_LIBDIR)"; exit 1
fi
LF=(--link-flag "-L$libdir" --link-flag "-Wl,-rpath,$libdir" --link-flag -lLLVM
    --link-flag -lstdc++ --link-flag -lm --link-flag -lpthread --link-flag -ldl)

if   [ -n "${STAGE0:-}" ];        then :
elif [ -x "$SEED" ];              then STAGE0="$SEED"
else echo "no stage0: need a committed $SEED (or set STAGE0=/path/to/coil)"; exit 1; fi
echo "stage0 = $STAGE0   (libLLVM: $libdir)"

echo "=== stage1: stage0 builds the self-host compiler ==="
"$STAGE0"        build "$SRC" -o /tmp/coil-lrb1 "${LF[@]}" || { echo "stage1 FAILED"; exit 1; }
echo "=== stage2: stage1 rebuilds it ==="
/tmp/coil-lrb1   build "$SRC" -o /tmp/coil-lrb2 "${LF[@]}" || { echo "stage2 FAILED"; exit 1; }
echo "=== stage3: stage2 rebuilds it ==="
/tmp/coil-lrb2   build "$SRC" -o /tmp/coil-lrb3 "${LF[@]}" || { echo "stage3 FAILED"; exit 1; }

echo "=== FIXPOINT: stage2.o vs stage3.o ==="
cmp /tmp/coil-lrb2.o /tmp/coil-lrb3.o || { echo "FIXPOINT FAIL — objects differ (nondeterminism)"; exit 2; }
echo "  ok — byte-identical, the compiler reproduces itself"

echo "=== GATES ==="
./selfhost/oracle/linux/gate-full.sh /tmp/coil-lrb2 >/dev/null || { echo "linux gate-full FAIL"; exit 1; }
echo "  linux gate-full: PASS (IR byte-exact vs the Linux snapshot)"
./selfhost/oracle/linux/gate-run.sh /tmp/coil-lrb2 >/dev/null  || { echo "linux gate-run FAIL"; exit 1; }
echo "  linux gate-run:  PASS (programs run identically)"
./selfhost/oracle/gate-cli.sh /tmp/coil-lrb2 >/dev/null        || { echo "gate-cli FAIL"; exit 1; }
echo "  gate-cli:        PASS (argv, exit codes, fmt)"

DEST="${1:-./coil-linux}"
cp /tmp/coil-lrb2 "$DEST"
echo "=== VERIFIED self-host compiler installed -> $DEST ==="
