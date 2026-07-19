#!/usr/bin/env bash
# THE x86-64 SysV-ABI GATE. Diff the self-host's `emit-ir --target
# x86_64-apple-macosx11.0.0` against the Rust reference (normalized x86 IR) across
# the struct-by-value corpus. Exit 0 iff byte-identical for EVERY input.
#
# The self-host currently IGNORES --target (it always lowers for the arm64 host),
# so this gate fails: the emitted `target triple`/`target datalayout` and the SysV
# struct coercion differ from the x86 reference. That failure is the 'x86' gap —
# the Port must teach the self-host codegen the requested target + SysV ABI.
#
# Usage: gate-x86.sh <coil-self-binary>
#   COIL_SELF_ARGS  extra args before `emit-ir` (default none)
#   VERBOSE=1       print the first differing hunk for each failure
set -uo pipefail
cd "$(dirname "$0")/../.."          # repo root
BIN=${1:?usage: gate-x86.sh <coil-self-binary>}
TARGET=x86_64-apple-macosx11.0.0
REF=selfhost/oracle/x86/reference
LIST=selfhost/oracle/x86/corpus.txt

[ -x "$BIN" ]  || { echo "GATE FAIL: self-host binary not executable: $BIN"; exit 2; }
[ -f "$LIST" ] || { echo "GATE FAIL: no x86 snapshot; run snapshot-x86.sh first"; exit 2; }

pass=0; fail=0; first_fail=""
while IFS= read -r f; do
  [ -z "$f" ] && continue
  ref="$REF/$(echo "$f" | tr '/' '_').dump"
  got=$("$BIN" ${COIL_SELF_ARGS:-} emit-ir "$f" --target "$TARGET" 2>/tmp/coil_self_x86_err); rc=$?
  if [ $rc -ne 0 ]; then
    fail=$((fail+1)); [ -z "$first_fail" ] && first_fail="$f (exit $rc): $(head -1 /tmp/coil_self_x86_err)"
    [ "${VERBOSE:-}" = 1 ] && echo "FAIL(crash) $f: $(head -1 /tmp/coil_self_x86_err)"
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

echo "gate-x86: $pass pass, $fail fail"
if [ $fail -ne 0 ]; then
  echo "first failure: $first_fail"
  exit 1
fi
echo "GATE PASS — self-host x86-64 SysV IR is byte-identical to the Rust reference"
