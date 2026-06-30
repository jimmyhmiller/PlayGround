#!/usr/bin/env bash
# Snapshot the REFERENCE resolver output (Rust `coil dump-resolved`) over the
# corpus. Two corpus halves:
#   (a) the `coil expand` output of every real .coil file (resolve consumes
#       POST-MACRO core forms — `parse_program` cannot parse a raw macro call —
#       so, exactly like the parser oracle, the corpus is the expanded text). A
#       file whose EXPAND fails is EXCLUDED + logged.
#   (b) curated multi-module RAW fixtures under fixtures/ (module + import
#       :as/:use */:use [names] + export + cross-module calls + the real stdlib),
#       fed to dump-resolved directly — they use only core forms so they resolve
#       without an expand step, and exercise the import/alias/export paths the
#       expanded single-file corpus does not.
# A load/resolve ERROR is NOT excluded: dump-resolved dumps it canonically
# (`(error@lo:hi "msg")`), like dump-ast, so error-path parity is gated too.
# This is the oracle the self-hosted resolver must reproduce byte-for-byte.
set -uo pipefail
cd "$(dirname "$0")/../.."          # repo root
COIL=${COIL_REF_BIN:-./target/debug/coil}
RES=selfhost/oracle/resolved
CORP=$RES/corpus
REF=$RES/reference
LIST=$RES/corpus.txt
EXCL=$RES/EXCLUDED.txt

[ -x "$COIL" ] || { echo "reference compiler not found: $COIL (run: cargo build)"; exit 1; }

rm -rf "$CORP" "$REF"; mkdir -p "$CORP" "$REF"
: > "$LIST"; : > "$EXCL"

n=0; ex=0
# (a) expand each real file, then dump-resolved the expanded core forms.
while IFS= read -r f; do
  mangled=$(echo "$f" | tr '/' '_')
  exp="$CORP/$mangled"
  if "$COIL" expand "$f" > "$exp" 2>/tmp/res_expand_err; then
    "$COIL" dump-resolved "$exp" > "$REF/$(echo "$exp" | tr '/' '_').dump"
    echo "$exp" >> "$LIST"
    n=$((n+1))
  else
    echo "$f : $(head -1 /tmp/res_expand_err)" >> "$EXCL"
    rm -f "$exp"
    ex=$((ex+1))
  fi
done < <(find examples lib apps src freestanding -name '*.coil' 2>/dev/null | sort)

# (b) curated multi-module RAW fixtures, fed to dump-resolved directly.
for f in "$RES"/fixtures/*.coil; do
  [ -e "$f" ] || continue
  "$COIL" dump-resolved "$f" > "$REF/$(echo "$f" | tr '/' '_').dump"
  echo "$f" >> "$LIST"
  n=$((n+1))
done

sort -o "$LIST" "$LIST"
echo "snapshot-resolved: $n files in corpus ($ex excluded -> $EXCL) -> $REF"
