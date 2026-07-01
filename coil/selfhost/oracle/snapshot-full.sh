#!/usr/bin/env bash
# Snapshot the REFERENCE for the FULL integrated pipeline (read -> load ->
# expand-stage3 -> resolve -> check -> mono -> codegen -> print -> normalize).
# Reference = Rust `coil dump-ir <RAW f>`, which is exactly `emit_ir` (textual LLVM
# IR via LLVMPrintModuleToString) run through `normalize_ir::normalize`. The
# self-host `selfhost/src/main.coil emit-ir` prints with the SAME LLVMPrintModuleToString
# and the SAME normalization, so a byte-diff gate is meaningful.
#
# UNLIKE the codegen-only gate (snapshot-ir.sh), the corpus here is the RAW,
# un-expanded source: the merged compiler must EXPAND macros itself before lowering.
# The corpus starts from the codegen corpus's real .coil seeds + minimal fixtures.
# Any file whose Rust `dump-ir` errors is SKIPPED (logged); the integration gate
# only covers files the whole reference pipeline accepts.
set -uo pipefail
cd "$(dirname "$0")/../.."          # repo root
COIL=${COIL_REF_BIN:-./target/debug/coil}
FULL=selfhost/oracle/full
REF=$FULL/reference
LIST=$FULL/corpus.txt

# Same seeds as the codegen corpus (snapshot-ir.sh), fed RAW through expansion.
SEEDS=(
  examples/allocation.coil examples/allocators.coil examples/args.coil
  examples/bitfields.coil examples/closure.coil examples/explicit-layout.coil
  examples/extern.coil examples/fib.coil examples/generics.coil
  examples/inference.coil examples/io.coil examples/layout.coil
  examples/lockfree.coil examples/mem.coil examples/references.coil
  examples/structs.coil examples/sums.coil examples/threads.coil
  examples/vector.coil examples/widths.coil
  examples/calc.coil examples/json.coil examples/hashmap.coil
  apps/chip8/objc.coil
  lib/alloc.coil lib/arraylist.coil lib/atomic.coil lib/closure.coil
  lib/control.coil lib/derive.coil lib/dyn.coil lib/fmt.coil lib/hashmap.coil
  lib/match.coil lib/mem.coil lib/mmio.coil lib/print.coil lib/result.coil
  lib/slice.coil lib/thread.coil lib/try.coil
  selfhost/src/main.coil
  # --- feature corpora: each exercises a stubbed self-host feature so it becomes
  # part of the contract (currently failing on the self-host, green on Rust). ---
  examples/dyn_write.coil                     # trait objects / dyn dispatch (lib/dyn.coil)
  examples/simd.coil lib/simd.coil            # SIMD / vector types (lib/simd.coil)
  selfhost/oracle/features/meta_stage3.coil   # (meta …) Stage-3 staged macros
  selfhost/oracle/features/export_c.coil      # (export-c …) C ABI export thunks
  selfhost/oracle/features/x86_sysv_abi.coil  # struct-by-value (host lowering; x86 gate below)
  examples/conventions.coil                    # custom calling convention -> LLVM fastcc
  examples/per-arch.coil                       # per-arch defcc (fastcc, informational regs)
  examples/shim.coil                           # :shim convention (naked trampoline + inline asm)
  examples/everything.coil                     # variadic externs + :shim + fastcc + aligned layout
)

[ -x "$COIL" ] || { echo "reference compiler not found: $COIL (run: cargo build)"; exit 1; }

mkdir -p "$FULL"
rm -rf "$REF"; mkdir -p "$REF"
: > "$LIST"

snap() {
  local f="$1"
  [ -e "$f" ] || { echo "MISSING corpus input: $f"; exit 1; }
  if ! "$COIL" dump-ir "$f" > "$REF/$(echo "$f" | tr '/' '_').dump" 2>/tmp/snap_full_err; then
    echo "SKIP (dump-ir error) $f: $(head -1 /tmp/snap_full_err)"
    rm -f "$REF/$(echo "$f" | tr '/' '_').dump"
    return
  fi
  echo "$f" >> "$LIST"
}

n=0
for f in selfhost/oracle/ir/fixtures/*.coil; do [ -e "$f" ] || continue; snap "$f"; n=$((n+1)); done
for f in "${SEEDS[@]}"; do snap "$f"; n=$((n+1)); done

sort -o "$LIST" "$LIST"
echo "snapshot-full: $(wc -l < "$LIST" | tr -d ' ') files in corpus -> $REF"
