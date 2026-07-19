#!/usr/bin/env bash
# THE EASY BOOTSTRAP — rebuild and VERIFY the self-host Coil compiler with NO Rust toolchain.
#
# stage0 is chosen automatically (NO Rust in the default path — the self-host
# compiler bootstraps and verifies itself):
#   1. $STAGE0 if you set it explicitly
#   2. selfhost/seed/coil-seed  (the committed, prebuilt self-host compiler) — DEFAULT
#   (the Rust reference compiler has been removed; the seed is fully self-sufficient)
# You never need cargo/rustc/inkwell; the seed re-derives the whole compiler from source.
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
# ---- THE THREE BUILDS --------------------------------------------------------
#
#   flavour        script                            LLVM            links
#   -------------  --------------------------------  --------------  -------------------------
#   dynamic-LLVM   rebootstrap.sh                    libLLVM.dylib   + Homebrew libLLVM  ~3.5MB
#   static-LLVM    COIL_LLVM_LINK=static  ditto      linked in       macOS /usr/lib only  ~92MB
#   no-LLVM        rebootstrap-nollvm.sh             none            libSystem only       ~3.2MB
#
# DYNAMIC is the default: it is what the committed seed expects and it is what you
# want while developing. The compiler it produces will NOT run without Homebrew's
# libLLVM.dylib.
#
# STATIC is for shipping a compiler to someone else. rustc and zig both take this
# route — a rustup toolchain has no system libLLVM anywhere, it is statically
# linked into a ~200MB librustc_driver — and the trade is the same one they make:
# ~26x the binary for a compiler that runs on a bare machine.
#
# NO-LLVM is the most self-contained of the three (only libSystem, needs only `cc`)
# and is verified as such by its own gate. Its arm64 backend still has gaps the
# LLVM backend does not — notably `export-c` with a by-value struct parameter — so
# it cannot yet build every program the other two can.
#
# The link line lives in ONE place, selfhost/llvm-link-flags.sh.
LF=($(./selfhost/llvm-link-flags.sh "${COIL_LLVM_LINK:-dynamic}")) \
  || { echo "cannot compute LLVM link flags"; exit 1; }

if   [ -n "${STAGE0:-}" ];        then :
elif [ -x "$SEED" ];              then STAGE0="$SEED"
else echo "no stage0: need a committed $SEED (or set STAGE0=/path/to/coil)"; exit 1
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

# The LLVM backend must be able to build the compiler too, and reach its own
# fixpoint. Stages 1-3 above do NOT cover this: stage1 uses stage0 (the committed
# seed, which predates whatever you just changed) and stages 2-3 both pass
# --backend arm64. So a codegen.coil change that breaks the LLVM backend on the
# compiler's own source passes every check above. One did: a C-ABI guard added to
# call-ptr rejected driver.coil's (fnptr c [… (slice u8) …] i64) backend hooks, and
# fixpoint + all three gates stayed green while `coil build main.coil` was broken.
echo "=== LLVM-BACKEND SELF-BUILD: stage1 rebuilds the compiler, twice ==="
/tmp/coil-rb1 build "$SRC" -o /tmp/coil-rl2 "${LF[@]}" >/dev/null \
  || { echo "LLVM self-build FAIL — the new compiler cannot build the compiler with the LLVM backend"; exit 2; }
/tmp/coil-rl2 build "$SRC" -o /tmp/coil-rl3 "${LF[@]}" >/dev/null \
  || { echo "LLVM self-build FAIL — stage rl2 cannot rebuild the compiler"; exit 2; }
cmp /tmp/coil-rl2.o /tmp/coil-rl3.o || { echo "LLVM FIXPOINT FAIL — LLVM-backend objects differ"; exit 2; }
echo "  ok — byte-identical, the LLVM backend reproduces the compiler too"

echo "=== GATES ==="
./selfhost/oracle/gate-full.sh /tmp/coil-rb2 >/dev/null      || { echo "gate-full FAIL — not a faithful compiler"; exit 1; }
echo "  gate-full:      PASS (IR byte-exact vs reference)"
./selfhost/oracle/arm64/gate-run.sh /tmp/coil-rb2 >/dev/null || { echo "arm64 gate-run FAIL — runtime divergence"; exit 1; }
echo "  arm64 gate-run: PASS (programs run identically)"
./selfhost/oracle/gate-cli.sh /tmp/coil-rb2 >/dev/null      || { echo "gate-cli FAIL — the CLI contract regressed"; exit 1; }
echo "  gate-cli:       PASS (argv, exit codes, fmt)"

DEST="${1:-./coil}"
cp /tmp/coil-rb2 "$DEST"
# Re-sign after copy: macOS invalidates a Mach-O's ad-hoc signature on cp, and the
# kernel SIGKILLs a mis-signed binary. Re-sign so the installed compiler runs.
codesign -s - --force "$DEST" >/dev/null 2>&1 || true
echo "=== VERIFIED self-host compiler installed -> $DEST ==="
