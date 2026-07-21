#!/usr/bin/env bash
# IR gate for the Linux x86-64 port: `emit-ir` every gate-full corpus file and diff
# byte-exact against the LINUX reference snapshot (this directory's full-reference/).
# The macOS snapshot (selfhost/oracle/full/reference) cannot serve here — triple,
# datalayout and the x86 musttail downgrade legitimately differ — so the port keeps
# its own, blessed by snapshot-full.sh from a verified compiler.
#
# Usage: selfhost/oracle/linux/gate-full.sh <coil-binary>
#   VERBOSE=1  print the first differing hunk for each failure
set -uo pipefail
cd "$(dirname "$0")/../../.."
BIN=${1:?usage: gate-full.sh <coil-binary>}
REF=selfhost/oracle/linux/full-reference
LIST=selfhost/oracle/full/corpus.txt

[ -x "$BIN" ]  || { echo "GATE FAIL: binary not executable: $BIN"; exit 2; }
[ -d "$REF" ]  || { echo "GATE FAIL: no Linux reference; run selfhost/oracle/linux/snapshot-full.sh first"; exit 2; }

ARM64_ONLY="examples/shim.coil examples/everything.coil"
pass=0; fail=0; first_fail=""
while IFS= read -r f; do
  [ -z "$f" ] && continue
  case " $ARM64_ONLY " in *" $f "*)
    # arm64-register shim conventions: on x86 the per-arch diagnostic must fire.
    out=$("$BIN" emit-ir "$f" 2>&1) && rc=0 || rc=$?
    if [ "$rc" != 0 ] && echo "$out" | grep -q "not a general-purpose register on the target architecture"; then
      pass=$((pass+1))
    else
      fail=$((fail+1)); [ -z "$first_fail" ] && first_fail="$f (want per-arch error, rc=$rc)"
    fi
    continue;;
  esac
  ref="$REF/$(echo "$f" | tr '/' '_').dump"
  got=$("$BIN" ${COIL_SELF_ARGS:-} emit-ir "$f" 2>/tmp/coil_linux_full_err); rc=$?
  if [ $rc -ne 0 ]; then
    fail=$((fail+1)); [ -z "$first_fail" ] && first_fail="$f (exit $rc): $(head -1 /tmp/coil_linux_full_err)"
    [ "${VERBOSE:-}" = 1 ] && echo "FAIL(crash) $f: $(head -1 /tmp/coil_linux_full_err)"
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
echo "linux gate-full: $pass passed, $fail failed"
[ -n "$first_fail" ] && echo "first failure: $first_fail"
[ "$fail" -eq 0 ]
