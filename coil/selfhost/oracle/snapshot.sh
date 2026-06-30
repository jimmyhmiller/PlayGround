#!/usr/bin/env bash
# Snapshot the REFERENCE reader output (Rust `coil dump-read`) over the whole
# corpus. This is the oracle the self-hosted reader must reproduce byte-for-byte.
# Re-run whenever the corpus or the canonical dump format changes.
set -euo pipefail
cd "$(dirname "$0")/../.."          # repo root
REF=selfhost/oracle/reference
LIST=selfhost/oracle/corpus.txt
COIL=${COIL_REF_BIN:-./target/debug/coil}

[ -x "$COIL" ] || { echo "reference compiler not found: $COIL (run: cargo build)"; exit 1; }

rm -rf "$REF"; mkdir -p "$REF"

# Corpus = every real .coil file in the tree (reader is semantics-agnostic, so
# all of them are valid inputs) plus the curated negative/edge fixtures.
{ find examples lib apps src freestanding -name '*.coil' 2>/dev/null
  find selfhost/oracle/negative -name '*.coil' 2>/dev/null
} | sort > "$LIST"

n=0
while IFS= read -r f; do
  out="$REF/$(echo "$f" | tr '/' '_').dump"
  "$COIL" dump-read "$f" > "$out"
  n=$((n+1))
done < "$LIST"
echo "snapshot: $n files -> $REF"
