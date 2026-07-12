#!/usr/bin/env bash
# Snapshot the REFERENCE parser output (Rust `coil dump-ast`) over the corpus.
# The corpus = the `coil expand` output of every real .coil file (the parser
# consumes POST-MACRO core forms), plus curated parser error/edge fixtures.
# This is the oracle the self-hosted parser must reproduce byte-for-byte.
set -uo pipefail
cd "$(dirname "$0")/../.."          # repo root
COIL=${COIL_REF_BIN:-./coil}
AST=selfhost/oracle/ast
CORP=$AST/corpus
REF=$AST/reference
LIST=$AST/corpus.txt
EXCL=$AST/EXCLUDED.txt

[ -x "$COIL" ] || { echo "reference compiler not found: $COIL (run: cargo build)"; exit 1; }

rm -rf "$CORP" "$REF"; mkdir -p "$CORP" "$REF"
: > "$LIST"; : > "$EXCL"

n=0; ex=0
while IFS= read -r f; do
  mangled=$(echo "$f" | tr '/' '_')
  exp="$CORP/$mangled"
  if "$COIL" expand "$f" > "$exp" 2>/tmp/ast_expand_err; then
    "$COIL" dump-ast "$exp" > "$REF/$(echo "$exp" | tr '/' '_').dump"
    echo "$exp" >> "$LIST"
    n=$((n+1))
  else
    echo "$f : $(head -1 /tmp/ast_expand_err)" >> "$EXCL"
    rm -f "$exp"
    ex=$((ex+1))
  fi
done < <(find examples lib apps src freestanding -name '*.coil' 2>/dev/null | sort)

# curated negative/edge fixtures (hand-written core forms with parser errors),
# fed to dump-ast directly (they are already core forms).
for f in "$AST"/negative/*.coil; do
  [ -e "$f" ] || continue
  "$COIL" dump-ast "$f" > "$REF/$(echo "$f" | tr '/' '_').dump"
  echo "$f" >> "$LIST"
  n=$((n+1))
done

sort -o "$LIST" "$LIST"
echo "snapshot-ast: $n files in corpus ($ex excluded -> $EXCL) -> $REF"
