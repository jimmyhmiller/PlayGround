#!/bin/bash
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
RESULTS=/tmp/ours.tsv
CORPUS="$ROOT/conformance/corpus.edn"
N=$(grep -c '' "$CORPUS")
rm -f "$RESULTS"
next=0; guard=0
while [ "$next" -lt "$N" ] && [ "$guard" -lt 120 ]; do
  guard=$((guard+1))
  START=$next CORPUS="$CORPUS" RESULTS="$RESULTS" \
    cargo test -p clojure-jvm --test oracle_runner run -- --nocapture >/dev/null 2>&1
  if grep -q "^DONE" "$RESULTS" 2>/dev/null; then break; fi
  last=$(grep -oE '^[0-9]+' "$RESULTS" 2>/dev/null | sort -n | tail -1)
  if [ -z "$last" ] || [ "$last" -lt "$next" ]; then
    printf '%s\tABORT\t(non-unwinding crash)\n' "$next" >> "$RESULTS"
    next=$((next+1))
  else
    next=$((last+1))
  fi
done
echo "HARNESS DONE: $(grep -cE '^[0-9]+' "$RESULTS") results"
