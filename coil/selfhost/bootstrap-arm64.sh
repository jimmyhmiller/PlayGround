#!/usr/bin/env bash
# Bootstrap fixpoint through the NATIVE arm64 backend (no LLVM in codegen):
#   stage1 = self-host compiler built by the Rust reference (LLVM backend)
#   stage2 = self-host compiler built by stage1 with --backend arm64
#   stage3 = self-host compiler built by stage2 with --backend arm64
# PROOF: stage2.o == stage3.o byte-identical (the arm64 backend is fully
# deterministic — no normalization, no UUID canonicalization needed at the
# object level), and each stage passes the full-pipeline oracle gate.
set -uo pipefail
cd "$(dirname "$0")/.."
SRC=selfhost/src/main.coil
REF=./target/debug/coil
LF=(--link-flag -L/opt/homebrew/opt/llvm/lib --link-flag -lLLVM)

[ -x "$REF" ] || { echo "reference compiler missing: $REF (run: cargo build)"; exit 1; }

echo "=== stage1: rust-reference builds the self-host compiler (LLVM backend) ==="
"$REF" build "$SRC" -o /tmp/a64-stage1 "${LF[@]}" || { echo "stage1 FAILED"; exit 1; }

echo "=== stage2: stage1 builds the self-host compiler with --backend arm64 ==="
/tmp/a64-stage1 build "$SRC" -o /tmp/a64-stage2 --backend arm64 "${LF[@]}" || { echo "stage2 FAILED"; exit 1; }

echo "=== stage3: stage2 builds the self-host compiler with --backend arm64 ==="
/tmp/a64-stage2 build "$SRC" -o /tmp/a64-stage3 --backend arm64 "${LF[@]}" || { echo "stage3 FAILED"; exit 1; }

echo "=== functionality: stage2 + stage3 must pass the full-pipeline oracle gate ==="
for s in a64-stage2 a64-stage3; do
  if ./selfhost/oracle/gate-full.sh /tmp/$s >/dev/null 2>&1; then
    echo "  $s: gate-full PASS"
  else
    echo "  $s: gate-full FAIL — not a faithful compiler"; exit 1
  fi
done

echo "=== PROOF: the arm64 backend reproduces itself — stage2.o vs stage3.o ==="
if cmp /tmp/a64-stage2.o /tmp/a64-stage3.o; then
  echo "ARM64 BOOTSTRAP FIXPOINT: stage2.o == stage3.o (byte-identical objects)"
else
  echo "objects differ — codegen nondeterminism!"; exit 2
fi
