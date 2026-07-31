#!/usr/bin/env bash
# THE EASY BOOTSTRAP — rebuild and VERIFY the self-host Coil compiler with NO Rust toolchain.
#
# stage0 is chosen automatically (NO Rust in the default path — the self-host
# compiler bootstraps and verifies itself):
#   1. $STAGE0 if you set it explicitly
#   2. bootstrap/seeds/native/coil-seed  (the committed, prebuilt self-host compiler) — DEFAULT
#   (the Rust reference compiler has been removed; the seed is fully self-sufficient)
# You never need cargo/rustc/inkwell; the seed re-derives the whole compiler from source.
#
# The seed is NEVER trusted blindly. Every run re-derives the compiler from source and proves
# the result faithful two independent ways, so a stale or tampered seed cannot slip through:
#   * FIXPOINT : stage2.o == stage3.o byte-identical  (the arm64 backend is fully deterministic)
#   * GATES    : gate-full  (emitted IR byte-exact vs the reference snapshot, whole corpus)
#                arm64 gate-run  (built programs produce identical stdout+exit)
#                gate-meta-engines (compiled-meta == interp-meta: corpus + byte-identical self-build)
#                gate-wasm  (interp-meta running in a single static wasm module; skips w/o node)
#
# Requirements: libLLVM.dylib (brew install llvm) + a C compiler (cc). That's it.
# (The compiler embeds an LLVM backend, so its binary links libLLVM even when the arm64
#  backend does the codegen. Only the Rust *build* toolchain is eliminated, not libLLVM.)
#
# Usage: scripts/compiler/rebootstrap.sh [install-dest]      (default dest: build/bin/coil)
#        STAGE0=/path/to/coil scripts/compiler/rebootstrap.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
SRC=src/compiler/main.coil
SEED=bootstrap/seeds/native/coil-seed
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
# The link line lives in ONE place, scripts/compiler/llvm-link-flags.sh.
LF=($(./scripts/compiler/llvm-link-flags.sh "${COIL_LLVM_LINK:-dynamic}")) \
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
python3 scripts/oracle.py gate all --compiler /tmp/coil-rb2 >/dev/null \
  || { echo "compiler snapshot gates FAIL; run: python3 scripts/oracle.py gate all --compiler /tmp/coil-rb2 --verbose"; exit 1; }
echo "  snapshot gates: PASS (read/ast/load/resolve/check/expand/mono/ir/diag/x86/full byte-exact)"
# Compiler-free consistency check on the SHARED corpus manifest: an entry blessed for
# only one platform breaks the other platform's gate-full with a missing-file error,
# which is exactly how fs_lib.coil landed (Linux reference only, macOS gate dead).
python3 scripts/oracle.py coverage >/dev/null \
  || { echo "snapshot coverage FAIL"; python3 scripts/oracle.py coverage; exit 1; }
echo "  corpus coverage: PASS (every corpus entry blessed on both platforms)"
python3 scripts/oracle.py runtime gate arm64 --compiler /tmp/coil-rb2 >/dev/null || { echo "arm64 runtime gate FAIL — runtime divergence"; exit 1; }
echo "  arm64 gate-run: PASS (programs run identically)"
./scripts/compiler/oracle/gate-cli.sh /tmp/coil-rb2 >/dev/null      || { echo "gate-cli FAIL — the CLI contract regressed"; exit 1; }
echo "  gate-cli:       PASS (argv, exit codes, fmt)"
# Both comptime engines must stay interchangeable: the default COMPILED engine (above
# gates run it) AND the INTERPRETER engine (COIL_META_INTERP / interp.coil). Keeps the
# choice open and neither from rotting.
python3 scripts/dev.py test meta --compiler /tmp/coil-rb2       || { echo "meta-engine gate FAIL — compiled-meta and interp-meta diverge"; exit 1; }
# interp-meta running IN wasm (single static module, wasm2c-ready). Skips if no node.
python3 scripts/dev.py test wasm --compiler /tmp/coil-rb2               || { echo "wasm gate FAIL — interp-meta-in-wasm self-check regressed"; exit 1; }

# The gates above all ran against rb2, the ARM64-backend build. rl3 is the one that
# gets installed, so it is gated too — same source, but different machine code, and
# "derived from verified source" is not the same as "verified".
python3 scripts/oracle.py gate full --compiler /tmp/coil-rl3 >/dev/null || { echo "gate-full FAIL on the LLVM build (the installed binary)"; exit 1; }
./scripts/compiler/oracle/gate-cli.sh  /tmp/coil-rl3 >/dev/null || { echo "gate-cli FAIL on the LLVM build (the installed binary)"; exit 1; }
echo "  installed binary: PASS (gate-full + gate-cli on the LLVM build)"

DEST="${1:-build/bin/coil}"
# Install the LLVM-BUILT compiler, not the arm64 one. Both are derived from the same
# source at the same depth — rb1 builds rb2 with the arm64 backend and rl2 with LLVM,
# and each then reproduces itself byte-identically — but the arm64 backend does no
# optimisation, so the compiler it produces runs about 11x slower: 10.2s vs 0.9s to
# compile the compiler, 11.0s vs 1.7s to lint it. Installing rb2 made every build,
# every lint and every gate in this repo pay that, which is why a `coil lint` over the
# tree took minutes. The arm64 build is still produced and still gated above — it is
# what proves that backend faithful — it is just not what anyone runs.
mkdir -p "$(dirname "$DEST")"
cp /tmp/coil-rl3 "$DEST"
# Re-sign after copy: macOS invalidates a Mach-O's ad-hoc signature on cp, and the
# kernel SIGKILLs a mis-signed binary. Re-sign so the installed compiler runs.
codesign -s - --force "$DEST" >/dev/null 2>&1 || true
echo "=== VERIFIED self-host compiler installed -> $DEST ==="
