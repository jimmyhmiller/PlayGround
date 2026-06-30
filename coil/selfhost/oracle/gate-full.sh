#!/usr/bin/env bash
# THE GATE (full integrated pipeline). Diff the self-hosted FULL compiler
# (`selfhost/coil.coil emit-ir FILE` = read -> load -> expand-stage3 -> resolve ->
# check -> mono -> codegen -> print -> normalize, on RAW source) against the Rust
# reference snapshot (`coil dump-ir FILE`) across the corpus. Exit 0 iff
# byte-identical for EVERY corpus file. Both sides are NORMALIZED textual LLVM IR
# (see snapshot-full.sh). A hacky/partial integration produces a non-empty diff
# somewhere and fails here — there is nowhere to hide.
#
# Usage: gate-full.sh <coil-self-binary>
#   COIL_SELF_ARGS  extra args to pass before `emit-ir` (default none)
#   VERBOSE=1       print the first differing hunk for each failure
set -uo pipefail
cd "$(dirname "$0")/../.."          # repo root
BIN=${1:?usage: gate-full.sh <coil-self-binary>}
REF=selfhost/oracle/full/reference
LIST=selfhost/oracle/full/corpus.txt

[ -x "$BIN" ]  || { echo "GATE FAIL: self-host binary not executable: $BIN"; exit 2; }
[ -f "$LIST" ] || { echo "GATE FAIL: no corpus snapshot; run snapshot-full.sh first"; exit 2; }

pass=0; fail=0; first_fail=""
while IFS= read -r f; do
  [ -z "$f" ] && continue
  ref="$REF/$(echo "$f" | tr '/' '_').dump"
  got=$("$BIN" ${COIL_SELF_ARGS:-} emit-ir "$f" 2>/tmp/coil_self_full_err); rc=$?
  if [ $rc -ne 0 ]; then
    fail=$((fail+1)); [ -z "$first_fail" ] && first_fail="$f (exit $rc): $(head -1 /tmp/coil_self_full_err)"
    [ "${VERBOSE:-}" = 1 ] && echo "FAIL(crash) $f: $(head -1 /tmp/coil_self_full_err)"
    continue
  fi
  if [ "$got" = "$(cat "$ref")" ]; then
    pass=$((pass+1))
  else
    fail=$((fail+1)); [ -z "$first_fail" ] && first_fail="$f (output mismatch)"
    if [ "${VERBOSE:-}" = 1 ]; then
      echo "FAIL(diff)  $f"
      diff <(cat "$ref") <(printf '%s' "$got") | head -12
    fi
  fi
done < "$LIST"

echo "gate-full: $pass pass, $fail fail"
if [ $fail -ne 0 ]; then
  echo "first failure: $first_fail"
  exit 1
fi
echo "GATE PASS — self-host FULL pipeline is byte-identical to reference across corpus"
