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
echo "snapshot-diag: $n emit-ir error inputs -> $REF"

# ---- build-path inputs -------------------------------------------------------
# These drive `coil build` (not emit-ir) to pin the CLI/build control flow around
# diagnostics: (1) a compile error MUST halt before the link step (no `cc … .o`
# junk), (2) a link failure prints Rust's exact `linker (cc) failed with exit
# status: N`, (3) a failing static-assert renders a real diagnostic. We capture
# BOTH the combined stdout+stderr AND the process exit code (a `.exit` sidecar),
# since faithfulness here is as much about exit codes as about text.
#
# The `-o` is EXTENSIONLESS and lives in a scratch dir: Rust's `with_extension("o")`
# and the self-host's obj-path then agree on the `<basename>.o` name, so any linker
# diagnostic that prints the object is byte-identical. The linker only ever prints
# the object BASENAME (derived from the input filename), never the scratch dir, so
# the snapshot is location-independent; we still strip $ROOT for good measure.
BLIST=selfhost/oracle/diag/build-corpus.txt
find selfhost/oracle/diag/build-inputs -name '*.coil' 2>/dev/null | sort > "$BLIST"
TMPD=$(mktemp -d)
b=0
while IFS= read -r f; do
  [ -z "$f" ] && continue
  out="$REF/$(echo "$f" | tr '/' '_').diag"
  ec="$REF/$(echo "$f" | tr '/' '_').exit"
  o="$TMPD/$(basename "$f" .coil)"
  # `|| code=$?` captures the (nonzero) exit without tripping `set -e`.
  code=0
  "$COIL" build "$f" -o "$o" > "$TMPD/raw" 2>&1 || code=$?
  # Strip $ROOT and the scratch dir so a build-SUCCESS `wrote <TMPD>/<name>` line
  # (and any object path) is location-independent and reproducible in the gate.
  sed "s|$ROOT/||g; s|$TMPD/||g" "$TMPD/raw" > "$out"
  echo "$code" > "$ec"
  b=$((b+1))
done < "$BLIST"
rm -rf "$TMPD"
echo "snapshot-diag: $b build error inputs -> $REF"
