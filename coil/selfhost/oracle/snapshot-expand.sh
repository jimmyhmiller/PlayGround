#!/usr/bin/env bash
# Snapshot the REFERENCE macro-expander output (Rust `coil dump-expand`) over the
# corpus. The corpus = RAW real .coil files (the expander runs on PRE-macro forms,
# like the loader — so unlike the parser there is no `coil expand` step), plus any
# curated edge fixtures. A file whose LOAD fails (missing import, non-module file,
# …) exits non-zero and is EXCLUDED + logged. A file that loads but whose macro
# EXPANSION errors is dumped canonically (`(error@lo:hi "msg")`, exit 0) and KEPT —
# error-path parity is part of the contract. This is the oracle the self-hosted
# expander must reproduce byte-for-byte.
set -uo pipefail
cd "$(dirname "$0")/../.."          # repo root
COIL=${COIL_REF_BIN:-./coil}
EXP=selfhost/oracle/expand
REF=$EXP/reference
LIST=$EXP/corpus.txt
EXCL=$EXP/EXCLUDED.txt

[ -x "$COIL" ] || { echo "reference compiler not found: $COIL (run: cargo build)"; exit 1; }

rm -rf "$REF"; mkdir -p "$REF"
: > "$LIST"; : > "$EXCL"

n=0; ex=0
snap() {  # snap <file>
  local f="$1"
  if "$COIL" dump-expand "$f" > "$REF/$(echo "$f" | tr '/' '_').dump" 2>/tmp/expand_err; then
    echo "$f" >> "$LIST"
    n=$((n+1))
  else
    echo "$f : $(head -1 /tmp/expand_err)" >> "$EXCL"
    rm -f "$REF/$(echo "$f" | tr '/' '_').dump"
    ex=$((ex+1))
  fi
}

while IFS= read -r f; do
  snap "$f"
done < <(find examples lib apps src freestanding -name '*.coil' 2>/dev/null | sort)

# curated edge fixtures (if any)
for f in "$EXP"/fixtures/*.coil; do
  [ -e "$f" ] || continue
  snap "$f"
done

sort -o "$LIST" "$LIST"
echo "snapshot-expand: $n files in corpus ($ex excluded -> $EXCL) -> $REF"
