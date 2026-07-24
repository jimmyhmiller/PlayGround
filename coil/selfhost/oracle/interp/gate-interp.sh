#!/usr/bin/env bash
# Behavioral gate for the BYTECODE INTERPRETER (selfhost/src/interp.coil). For
# every corpus program, `coil interp <prog>` runs it through the SAME front-end
# the backends use (parse -> resolve -> check -> mono) and INTERPRETS the mono'd
# Program, then we diff its stdout+exit against the LLVM-backend reference
# snapshot. Runtime equality with the compiled program is the contract: the
# interpreter must produce byte-identical output and the same exit code.
#
# Modeled on selfhost/oracle/arm64/gate-run.sh, but instead of build+run it does
# `coil interp`. It reuses the SAME corpus and the SAME reference snapshots as the
# arm64 gate (selfhost/oracle/arm64/{corpus.txt,reference}).
#
# usage: gate-interp.sh <coil-self-bin> [--verbose]
set -uo pipefail
cd "$(dirname "$0")/../../.."
BIN="${1:?usage: gate-interp.sh <coil-self-bin>}"
VERBOSE="${2:-}"
ARM=selfhost/oracle/arm64
REF="$ARM/reference"
pass=0; fail=0
while IFS= read -r line; do
  [ -z "$line" ] && continue
  case "$line" in \#*) continue;; esac
  set -- $line
  # A leading `R ` marks programs whose reference came from the Rust compiler
  # (inline-asm streamer cases); interpret them the same way.
  if [ "$1" = "R" ]; then shift; fi
  f="$1"; shift
  id=$(echo "$f" | tr '/.' '__')
  outf="/tmp/coil-interp-gate-$id.out"
  timeout 60 "$BIN" interp "$f" "$@" </dev/null >"$outf" 2>/dev/null
  code=$?
  refcode=$(cat "$REF/$id.exit" 2>/dev/null)
  if cmp -s "$outf" "$REF/$id.stdout" && [ "$code" = "$refcode" ]; then
    pass=$((pass+1)); [ -n "$VERBOSE" ] && echo "ok  $f"
  else
    echo "FAIL(interp) $f  exit=$code want=$refcode"
    if [ -n "$VERBOSE" ]; then
      diff "$outf" "$REF/$id.stdout" 2>/dev/null | head -6
    fi
    fail=$((fail+1))
  fi
done < "$ARM/corpus.txt"
echo "interp gate: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
