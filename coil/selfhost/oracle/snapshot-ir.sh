#!/usr/bin/env bash
# Snapshot the REFERENCE codegen output (Rust `coil dump-ir`) over the codegen
# corpus. `dump-ir` = `emit-ir` (textual LLVM IR via LLVMPrintModuleToString) run
# through `normalize_ir::normalize`, which cancels codegen's run-to-run
# nondeterminism (positional global numbering @g/@str/@cstr/@__coil_llvm_ir_N,
# attribute-group #N numbering, top-level emission order) so a byte-diff gate is
# meaningful. The self-host codegen prints with the SAME LLVMPrintModuleToString
# and the SAME normalization, so both sides are compared on equal footing.
#
# The corpus is a curated SMALL subset to start (the codegen port is the
# multi-day frontier): minimal fixtures under selfhost/oracle/ir/fixtures/ plus a
# few real .coil files listed in seeds below. NOTE: every program drags the
# auto-loaded `coil.core` prelude (~50 functions) into its IR — there is no
# genuinely tiny whole-module target, so a gate PASS requires lowering the whole
# prelude closure. That is the honest shape of codegen first-green.
set -uo pipefail
cd "$(dirname "$0")/../.."          # repo root
COIL=${COIL_REF_BIN:-./target/debug/coil}
IR=selfhost/oracle/ir
REF=$IR/reference
LIST=$IR/corpus.txt

# Real .coil files included in the codegen corpus (kept small; emit-ir must succeed).
SEEDS=(
  examples/allocation.coil examples/allocators.coil examples/args.coil
  examples/bitfields.coil examples/closure.coil examples/explicit-layout.coil
  examples/extern.coil examples/fib.coil examples/generics.coil
  examples/inference.coil examples/io.coil examples/layout.coil
  examples/lockfree.coil examples/mem.coil examples/references.coil
  examples/structs.coil examples/sums.coil examples/threads.coil
  examples/vector.coil examples/widths.coil
  apps/chip8/objc.coil
  lib/alloc.coil lib/arraylist.coil lib/atomic.coil lib/closure.coil
  lib/control.coil lib/derive.coil lib/dyn.coil lib/fmt.coil lib/hashmap.coil
  lib/match.coil lib/mem.coil lib/mmio.coil lib/print.coil lib/result.coil
  lib/slice.coil lib/thread.coil lib/try.coil
)

[ -x "$COIL" ] || { echo "reference compiler not found: $COIL (run: cargo build)"; exit 1; }

rm -rf "$REF"; mkdir -p "$REF"
: > "$LIST"

snap() {
  local f="$1"
  [ -e "$f" ] || { echo "MISSING corpus input: $f"; exit 1; }
  if ! "$COIL" dump-ir "$f" > "$REF/$(echo "$f" | tr '/' '_').dump" 2>/tmp/snap_ir_err; then
    echo "SKIP (emit-ir error) $f: $(head -1 /tmp/snap_ir_err)"
    rm -f "$REF/$(echo "$f" | tr '/' '_').dump"
    return
  fi
  echo "$f" >> "$LIST"
}

n=0
for f in "$IR"/fixtures/*.coil; do [ -e "$f" ] || continue; snap "$f"; n=$((n+1)); done
for f in "${SEEDS[@]}"; do snap "$f"; n=$((n+1)); done

sort -o "$LIST" "$LIST"
echo "snapshot-ir: $(wc -l < "$LIST" | tr -d ' ') files in corpus -> $REF"
