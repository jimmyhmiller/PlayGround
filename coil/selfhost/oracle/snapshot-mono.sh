#!/usr/bin/env bash
# Snapshot the REFERENCE monomorphizer output (Rust `coil dump-mono`) over the
# corpus. The mono pass runs read -> load -> resolve -> check -> MONO, so it
# consumes the SAME pre-expanded core forms the checker does. We therefore REUSE
# the checker corpus inputs verbatim:
#   (a) the `coil expand` output of every cleanly-expanding real .coil file
#       (selfhost/oracle/resolved/corpus/*), and
#   (b) the curated multi-module RAW fixtures (selfhost/oracle/resolved/fixtures/*),
#   (c) the curated TYPE-ERROR fixtures (selfhost/oracle/checked/fixtures/*).
# All three are listed in selfhost/oracle/checked/corpus.txt — we read that list.
# Plus (d) curated MONO-ERROR fixtures under selfhost/oracle/mono/fixtures/ (if any).
#
# A load/resolve/check ERROR is dumped canonically by dump-mono (the front-end
# Diag, FIRST diagnostic) — mono never runs in that case. A MONO error is dumped
# spanless (`(error@D:D "msg")`). So every error path is gated too.
set -uo pipefail
cd "$(dirname "$0")/../.."          # repo root
COIL=${COIL_REF_BIN:-./target/debug/coil}
MONO=selfhost/oracle/mono
REF=$MONO/reference
LIST=$MONO/corpus.txt
CHKLIST=selfhost/oracle/checked/corpus.txt

[ -x "$COIL" ]    || { echo "reference compiler not found: $COIL (run: cargo build)"; exit 1; }
[ -f "$CHKLIST" ] || { echo "no checker corpus; run snapshot-checked.sh first"; exit 1; }

rm -rf "$REF"; mkdir -p "$REF"
: > "$LIST"

n=0
# (a)+(b)+(c) reuse the checker corpus inputs verbatim.
while IFS= read -r f; do
  [ -e "$f" ] || { echo "MISSING corpus input: $f"; exit 1; }
  "$COIL" dump-mono "$f" > "$REF/$(echo "$f" | tr '/' '_').dump"
  echo "$f" >> "$LIST"
  n=$((n+1))
done < "$CHKLIST"

# (d) curated mono-error fixtures, fed to dump-mono directly.
for f in "$MONO"/fixtures/*.coil; do
  [ -e "$f" ] || continue
  "$COIL" dump-mono "$f" > "$REF/$(echo "$f" | tr '/' '_').dump"
  echo "$f" >> "$LIST"
  n=$((n+1))
done

sort -o "$LIST" "$LIST"
echo "snapshot-mono: $n files in corpus -> $REF"
