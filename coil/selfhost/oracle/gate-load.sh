#!/usr/bin/env bash
# THE GATE (loader). Diff a self-hosted loader against the reference snapshot
# across the whole corpus. Exit 0 iff `<coil-self> dump-load FILE` is
# byte-identical to the reference for EVERY corpus file. This is the contract: a
# hacky/partial port produces a non-empty diff somewhere and fails here — there
# is nowhere to hide.
#
# Usage: gate-load.sh <coil-self-binary>
#   COIL_SELF_ARGS  extra args to pass before `dump-load` (default none)
#   VERBOSE=1       print the first differing line for each failure
set -uo pipefail
cd "$(dirname "$0")/../.."          # repo root
BIN=${1:?usage: gate-load.sh <coil-self-binary>}
REF=selfhost/oracle/load/reference
LIST=selfhost/oracle/load/corpus.txt

[ -x "$BIN" ]  || { echo "GATE FAIL: self-host binary not executable: $BIN"; exit 2; }
[ -f "$LIST" ] || { echo "GATE FAIL: no corpus snapshot; run snapshot-load.sh first"; exit 2; }

pass=0; fail=0; first_fail=""
while IFS= read -r f; do
  ref="$REF/$(echo "$f" | tr '/' '_').dump"
  got=$("$BIN" ${COIL_SELF_ARGS:-} dump-load "$f" 2>/tmp/coil_self_load_err); rc=$?
  if [ $rc -ne 0 ]; then
    fail=$((fail+1)); [ -z "$first_fail" ] && first_fail="$f (exit $rc): $(head -1 /tmp/coil_self_load_err)"
    [ "${VERBOSE:-}" = 1 ] && echo "FAIL(crash) $f: $(head -1 /tmp/coil_self_load_err)"
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

echo "gate-load: $pass pass, $fail fail"
if [ $fail -ne 0 ]; then
  echo "first failure: $first_fail"
  exit 1
fi
echo "GATE PASS — self-host loader is byte-identical to reference across corpus"
