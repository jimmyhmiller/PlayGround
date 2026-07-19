#!/usr/bin/env bash
# Snapshot the REFERENCE x86-64 SysV-ABI IR (Rust `coil dump-ir --target
# x86_64-apple-macosx11.0.0`) for the struct-by-value corpus. Reference =
# normalize(emit_ir_for(x86)) — the SAME normalization as the host `dump-ir`, so
# the self-host's `emit-ir --target …` is gated on equal footing.
#
# This is a CROSS-TARGET gate: it exercises the x86-64 SysV struct-by-value
# coercion (small ints coalesced into a register, big/HFA rules) from an arm64
# host — a code path distinct from the host AAPCS64 lowering the full gate covers.
set -euo pipefail
cd "$(dirname "$0")/../.."          # repo root
COIL=${COIL_REF_BIN:-./coil}
TARGET=x86_64-apple-macosx11.0.0
REF=selfhost/oracle/x86/reference
LIST=selfhost/oracle/x86/corpus.txt

[ -x "$COIL" ] || { echo "reference compiler not found: $COIL (run: cargo build)"; exit 1; }

rm -rf "$REF"; mkdir -p "$REF"
find selfhost/oracle/features -name '*x86*.coil' 2>/dev/null | sort > "$LIST"

n=0
while IFS= read -r f; do
  [ -z "$f" ] && continue
  out="$REF/$(echo "$f" | tr '/' '_').dump"
  "$COIL" dump-ir "$f" --target "$TARGET" > "$out"
  n=$((n+1))
done < "$LIST"
echo "snapshot-x86: $n files (target $TARGET) -> $REF"
