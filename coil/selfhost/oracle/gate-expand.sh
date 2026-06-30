#!/usr/bin/env bash
# THE GATE (macro expander). Diff a self-hosted expander against the reference
# snapshot across the whole corpus. Exit 0 iff `<coil-self> dump-expand FILE` is
# byte-identical to the reference for EVERY corpus file. This is the contract: a
# hacky/partial port produces a non-empty diff somewhere and fails here — there is
# nowhere to hide.
#
# Usage: gate-expand.sh <coil-self-binary>
#   COIL_SELF_ARGS  extra args to pass before `dump-expand` (default none)
#   VERBOSE=1       print the first differing line for each failure
set -uo pipefail
cd "$(dirname "$0")/../.."          # repo root
BIN=${1:?usage: gate-expand.sh <coil-self-binary>}
REF=selfhost/oracle/expand/reference
LIST=selfhost/oracle/expand/corpus.txt

[ -x "$BIN" ]  || { echo "GATE FAIL: self-host binary not executable: $BIN"; exit 2; }
[ -f "$LIST" ] || { echo "GATE FAIL: no corpus snapshot; run snapshot-expand.sh first"; exit 2; }

pass=0; fail=0; first_fail=""
while IFS= read -r f; do
  ref="$REF/$(echo "$f" | tr '/' '_').dump"
  got=$("$BIN" ${COIL_SELF_ARGS:-} dump-expand "$f" 2>/tmp/coil_self_expand_err); rc=$?
  if [ $rc -ne 0 ]; then
    fail=$((fail+1)); [ -z "$first_fail" ] && first_fail="$f (exit $rc): $(head -1 /tmp/coil_self_expand_err)"
    [ "${VERBOSE:-}" = 1 ] && echo "FAIL(crash) $f: $(head -1 /tmp/coil_self_expand_err)"
    continue
  fi
  if [ "$got" = "$(cat "$ref")" ]; then
    pass=$((pass+1))
  else
    fail=$((fail+1)); [ -z "$first_fail" ] && first_fail="$f (output mismatch)"
    if [ "${VERBOSE:-}" = 1 ]; then
      echo "FAIL(diff)  $f"
      diff <(cat "$ref") <(printf '%s' "$got") | head -8
    fi
  fi
done < "$LIST"

echo "gate-expand: $pass pass, $fail fail"
if [ $fail -ne 0 ]; then
  echo "first failure: $first_fail"
  exit 1
fi
echo "GATE PASS — self-host expander is byte-identical to reference across corpus"
