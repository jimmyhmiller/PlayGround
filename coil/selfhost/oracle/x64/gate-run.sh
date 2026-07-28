#!/usr/bin/env bash
# Behavioral gate for the x86-64 backend: build every corpus program with
# `--backend x64`, run it, and diff stdout+exit against the LLVM-backend
# reference snapshot. Runtime equality is the contract between backends.
#
# usage: gate-run.sh <coil-self-bin> [--verbose]
set -uo pipefail
cd "$(dirname "$0")/../../.."
BIN="${1:?usage: gate-run.sh <coil-self-bin>}"
VERBOSE="${2:-}"
HERE=selfhost/oracle/x64
REF="$HERE/reference"
pass=0; fail=0
while IFS= read -r line; do
  [ -z "$line" ] && continue
  case "$line" in \#*) continue;; esac
  set -- $line
  f="$1"; shift
  id=$(echo "$f" | tr '/.' '__')
  exe="/tmp/coil-x64-gate-$id"
  if ! timeout 120 "$BIN" build "$f" -o "$exe" --backend x64 >/dev/null 2>"/tmp/coil-x64-gate-$id.buildlog"; then
    echo "FAIL(build) $f"; [ -n "$VERBOSE" ] && head -3 "/tmp/coil-x64-gate-$id.buildlog"
    fail=$((fail+1)); continue
  fi
  cp "$exe" /tmp/coil-x64-fixed-$id
  timeout 30 /tmp/coil-x64-fixed-$id "$@" </dev/null >"/tmp/coil-x64-gate-$id.out" 2>/dev/null
  code=$?
  refcode=$(cat "$REF/$id.exit" 2>/dev/null)
  if cmp -s "/tmp/coil-x64-gate-$id.out" "$REF/$id.stdout" && [ "$code" = "$refcode" ]; then
    pass=$((pass+1)); [ -n "$VERBOSE" ] && echo "ok  $f"
  else
    echo "FAIL(run) $f  exit=$code want=$refcode"
    if [ -n "$VERBOSE" ]; then
      diff "/tmp/coil-x64-gate-$id.out" "$REF/$id.stdout" | head -6
    fi
    fail=$((fail+1))
  fi
done < "$HERE/corpus.txt"
echo "x64 gate: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
