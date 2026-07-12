#!/usr/bin/env bash
# Snapshot the REFERENCE loader output (Rust `coil dump-load`) over the corpus.
# The corpus = RAW real .coil files (the loader runs on PRE-macro forms, so unlike
# the parser there is no `coil expand` step), plus curated edge fixtures. A file
# whose load FAILS (missing import, non-module file, …) exits non-zero and is
# EXCLUDED + logged — only cleanly-loading files become corpus + reference.
# This is the oracle the self-hosted loader must reproduce byte-for-byte.
set -uo pipefail
cd "$(dirname "$0")/../.."          # repo root
COIL=${COIL_REF_BIN:-./coil}
LOAD=selfhost/oracle/load
REF=$LOAD/reference
LIST=$LOAD/corpus.txt
EXCL=$LOAD/EXCLUDED.txt

[ -x "$COIL" ] || { echo "reference compiler not found: $COIL (run: cargo build)"; exit 1; }

rm -rf "$REF"; mkdir -p "$REF"
: > "$LIST"; : > "$EXCL"

n=0; ex=0
snap() {  # snap <file>
  local f="$1"
  if "$COIL" dump-load "$f" > "$REF/$(echo "$f" | tr '/' '_').dump" 2>/tmp/load_err; then
    echo "$f" >> "$LIST"
    n=$((n+1))
  else
    echo "$f : $(head -1 /tmp/load_err)" >> "$EXCL"
    rm -f "$REF/$(echo "$f" | tr '/' '_').dump"
    ex=$((ex+1))
  fi
}

while IFS= read -r f; do
  snap "$f"
done < <(find examples lib apps src freestanding -name '*.coil' 2>/dev/null | sort)

# curated edge fixtures (module + import :as + :use [names] + export)
for f in "$LOAD"/fixtures/*.coil; do
  [ -e "$f" ] || continue
  snap "$f"
done

sort -o "$LIST" "$LIST"
echo "snapshot-load: $n files in corpus ($ex excluded -> $EXCL) -> $REF"
