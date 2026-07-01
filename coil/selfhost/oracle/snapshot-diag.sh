#!/usr/bin/env bash
# Snapshot the REFERENCE human-diagnostic renderer (Rust `coil emit-ir`) over the
# error corpus. Unlike the pass dumps (which compare a canonical `(error@… "msg")`
# form), this gate compares the FULL rendered STDERR the Rust compiler shows a
# human: the `error: <msg>` line, the `--> file:line:col` frame, the source line +
# caret, the `note: in expansion of macro …` provenance trace, and the multi-error
# `N errors` summary (src/span.rs render/render_all + src/main.rs report/print_error).
#
# The self-host currently only emits `(error@lo:hi "msg")` — it has NO renderer —
# so gate-diag.sh fails wholesale against it. That failure IS the contract the Port
# phase must turn green by porting the diagnostic renderer.
#
# Determinism: every input is a pure front-end (parse/resolve/check) error, so
# `emit-ir` fails before codegen/link — no temp files, no linker, no LLVM. The one
# imported-file case resolves its import to an ABSOLUTE path; we normalize the repo
# root away (identically here and in the gate) so the snapshot is repo-relative and
# regenerates byte-identically regardless of where the repo lives.
set -euo pipefail
cd "$(dirname "$0")/../.."          # repo root
ROOT=$(pwd)
REF=selfhost/oracle/diag/reference
LIST=selfhost/oracle/diag/corpus.txt
COIL=${COIL_REF_BIN:-./target/debug/coil}

[ -x "$COIL" ] || { echo "reference compiler not found: $COIL (run: cargo build)"; exit 1; }

rm -rf "$REF"; mkdir -p "$REF"
find selfhost/oracle/diag/inputs -name '*.coil' 2>/dev/null | sort > "$LIST"

n=0
while IFS= read -r f; do
  [ -z "$f" ] && continue
  out="$REF/$(echo "$f" | tr '/' '_').diag"
  # Combined stdout+stderr (stdout is empty for an error input); strip the repo-root
  # prefix so an imported file's absolute path becomes repo-relative.
  "$COIL" emit-ir "$f" 2>&1 | sed "s|$ROOT/||g" > "$out" || true
  n=$((n+1))
done < "$LIST"
echo "snapshot-diag: $n error inputs -> $REF"
