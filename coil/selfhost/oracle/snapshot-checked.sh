#!/usr/bin/env bash
# Snapshot the REFERENCE type-checker output (Rust `coil dump-checked`) over the
# corpus. The check pass runs read -> load -> resolve -> CHECK, so it consumes the
# SAME pre-expanded core forms the resolver does. We therefore REUSE the resolver
# corpus inputs verbatim:
#   (a) the `coil expand` output of every cleanly-expanding real .coil file
#       (selfhost/oracle/resolved/corpus/*), and
#   (b) the curated multi-module RAW fixtures (selfhost/oracle/resolved/fixtures/*).
# Both are listed in selfhost/oracle/resolved/corpus.txt — we read that list.
# Plus (c) curated TYPE-ERROR fixtures under selfhost/oracle/checked/fixtures/
# (bad bound, arity mismatch, unknown field, …) so the check error path is gated.
#
# A load/resolve ERROR is NOT possible here: the (a)/(b) inputs already resolve
# cleanly (that's what put them in the resolver corpus). A CHECK error is NOT
# excluded: dump-checked dumps it canonically (`(error@lo:hi "msg")`), using the
# FIRST diagnostic, so error-path parity is gated too.
set -uo pipefail
cd "$(dirname "$0")/../.."          # repo root
COIL=${COIL_REF_BIN:-./coil}
CHK=selfhost/oracle/checked
REF=$CHK/reference
LIST=$CHK/corpus.txt
RESLIST=selfhost/oracle/resolved/corpus.txt

[ -x "$COIL" ]   || { echo "reference compiler not found: $COIL (run: cargo build)"; exit 1; }
[ -f "$RESLIST" ] || { echo "no resolver corpus; run snapshot-resolved.sh first"; exit 1; }

rm -rf "$REF"; mkdir -p "$REF"
: > "$LIST"

n=0
# (a)+(b) reuse the resolver corpus inputs verbatim.
while IFS= read -r f; do
  [ -e "$f" ] || { echo "MISSING corpus input: $f"; exit 1; }
  "$COIL" dump-checked "$f" > "$REF/$(echo "$f" | tr '/' '_').dump"
  echo "$f" >> "$LIST"
  n=$((n+1))
done < "$RESLIST"

# (c) curated type-error fixtures, fed to dump-checked directly.
for f in "$CHK"/fixtures/*.coil; do
  [ -e "$f" ] || continue
  "$COIL" dump-checked "$f" > "$REF/$(echo "$f" | tr '/' '_').dump"
  echo "$f" >> "$LIST"
  n=$((n+1))
done

sort -o "$LIST" "$LIST"
echo "snapshot-checked: $n files in corpus -> $REF"
