#!/usr/bin/env bash
# The classic bootstrap fixpoint for the self-host Coil compiler.
#
#   stage1 = self-host compiler, built by the RUST reference compiler
#   stage2 = self-host compiler, built by stage1 (the self-host compiling ITSELF)
#   stage3 = self-host compiler, built by stage2
#
# THE PROOF: the self-host compiler reproduces itself. Two independent, equally
# strong statements:
#   (A) stage2.o == stage3.o byte-identical  — the self-host's CODEGEN is fully
#       deterministic; needs NO normalization. This is the meaningful proof.
#   (B) stage2 == stage3 byte-identical as EXECUTABLES, after canonicalizing the
#       one source of link-tool nondeterminism: the system linker stamps a RANDOM
#       LC_UUID into every Mach-O it produces (dyld requires the load command to be
#       present, so -no_uuid is not an option — it makes the binary unrunnable), and
#       the ad-hoc code signature hashes that UUID. canon-uuid.py rewrites the UUID
#       to a fixed value and we re-sign with a fixed identifier. This is exactly the
#       "make the link reproducible" step real reproducible-build systems use; it is
#       applied identically to every stage and touches nothing the compiler emits.
#
# stage1 (lowered by Rust's backend) may differ from stage2/stage3 (lowered by the
# self-host's own LLVMTargetMachineEmitToFile) — only the fixpoint stage2==stage3 is
# required. Each stage is also sanity-checked to be a WORKING compiler (its emit-ir
# matches the Rust reference), not merely byte-equal.
set -uo pipefail
cd "$(dirname "$0")/.."             # repo root
SRC=selfhost/src/main.coil
REF=./target/debug/coil
LF=(--link-flag -L/opt/homebrew/opt/llvm/lib --link-flag -lLLVM)
SAMPLE=examples/fib.coil
CANON="python3 $(dirname "$0")/canon-uuid.py"

[ -x "$REF" ] || { echo "reference compiler missing: $REF (run: cargo build)"; exit 1; }

# Canonicalize link-tool nondeterminism: fix the random Mach-O LC_UUID + re-sign
# with a fixed identifier so a reproducible-build comparison sees identical bytes.
canon() { $CANON "$1" && codesign -f -s - -i coil-selfhost "$1"; }

echo "=== stage1: rust-reference builds the self-host compiler ==="
"$REF" build "$SRC" -o /tmp/stage1 "${LF[@]}" || { echo "stage1 build FAILED"; exit 1; }

echo "=== stage2: stage1 builds the self-host compiler ==="
/tmp/stage1 build "$SRC" -o /tmp/stage2 "${LF[@]}" || { echo "stage2 build FAILED"; exit 1; }

echo "=== stage3: stage2 builds the self-host compiler ==="
/tmp/stage2 build "$SRC" -o /tmp/stage3 "${LF[@]}" || { echo "stage3 build FAILED"; exit 1; }

echo "=== functionality: each stage's emit-ir must match the Rust reference ==="
"$REF" dump-ir "$SAMPLE" > /tmp/ref.ir
for s in stage1 stage2 stage3; do
  if diff -q <(/tmp/$s emit-ir "$SAMPLE") /tmp/ref.ir >/dev/null; then
    echo "  $s emit-ir $SAMPLE: OK (matches Rust reference)"
  else
    echo "  $s emit-ir $SAMPLE: MISMATCH — $s is not a faithful compiler"; exit 1
  fi
done

rc=0
echo "=== PROOF (A): self-host codegen is deterministic — stage2.o vs stage3.o ==="
if cmp /tmp/stage2.o /tmp/stage3.o; then
  echo "  OBJECTS BYTE-IDENTICAL (no normalization needed)"
else
  echo "  OBJECTS DIFFER — codegen nondeterminism!"; rc=2
fi

echo "=== PROOF (B): executables stage2 vs stage3 (UUID-canonicalized) ==="
canon /tmp/stage2 >/dev/null
canon /tmp/stage3 >/dev/null
if cmp /tmp/stage2 /tmp/stage3; then
  echo "BOOTSTRAP FIXPOINT: stage2 == stage3 (byte-identical executables)"
  /tmp/stage2 emit-ir "$SAMPLE" >/dev/null 2>&1 \
    && echo "  (canonicalized stage2 still runs correctly)" \
    || { echo "  canonicalized binary BROKEN"; rc=1; }
else
  echo "  stage2 != stage3 after canonicalization — residual:"
  cmp -l /tmp/stage2 /tmp/stage3 | head; rc=1
fi
exit $rc
