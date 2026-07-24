#!/usr/bin/env bash
# Contract gate for the BYTECODE INTERPRETER: for every corpus program, prove
# that `coil interp <prog>` produces byte-identical stdout and the same exit code
# as the SAME program COMPILED and run. This is the interpreter's actual contract
# ("interp == compiled"), verified against LIVE compiler output rather than a
# frozen snapshot — the direct, strongest form of "the interpreter matches
# codegen".
#
# argv[0] is an invocation artifact, not program behavior: a compiled binary
# reports its own filesystem path, an interpreter reports the source path it was
# handed. We therefore invoke the compiled binary with `exec -a <srcpath>` so BOTH
# sides see the identical argv[0] (= the source path). This normalization is
# applied UNIFORMLY to every program — for the 54 programs that never read argv[0]
# it changes nothing; for `args`/`everything` (which `puts(argv[0])`) it makes the
# comparison one of program semantics, not of where the binary happens to live.
# Nothing is faked, skipped, or special-cased.
#
# `R`-marked corpus lines use inline-asm `:shim` trampolines; they are built with
# `--backend arm64` exactly as the corpus provenance declares. All other programs
# build with the default (LLVM) backend.
#
# usage: gate-interp-vs-compiled.sh <coil-self-bin> [--verbose]
set -uo pipefail
cd "$(dirname "$0")/../../.."
BIN="${1:?usage: gate-interp-vs-compiled.sh <coil-self-bin>}"
VERBOSE="${2:-}"
CORPUS=selfhost/oracle/arm64/corpus.txt
pass=0; fail=0
while IFS= read -r line; do
  [ -z "$line" ] && continue
  case "$line" in \#*) continue;; esac
  set -- $line
  BK=""
  if [ "$1" = "R" ]; then BK="--backend arm64"; shift; fi
  f="$1"; shift
  id=$(echo "$f" | tr '/.' '__')
  exe="/tmp/coil-ivc-$id"
  if ! timeout 120 "$BIN" build "$f" -o "$exe" $BK >"/tmp/coil-ivc-$id.buildlog" 2>&1; then
    echo "FAIL(build) $f"; [ -n "$VERBOSE" ] && head -3 "/tmp/coil-ivc-$id.buildlog"
    fail=$((fail+1)); continue
  fi
  cout=$(exec -a "$f" "$exe" "$@" </dev/null 2>/dev/null); ccode=$?
  iout=$(timeout 60 "$BIN" interp "$f" "$@" </dev/null 2>/dev/null); icode=$?
  if [ "$cout" = "$iout" ] && [ "$ccode" = "$icode" ]; then
    pass=$((pass+1)); [ -n "$VERBOSE" ] && echo "ok  $f"
  else
    echo "FAIL(diff) $f  interp-exit=$icode compiled-exit=$ccode"
    if [ -n "$VERBOSE" ]; then diff <(printf '%s' "$cout") <(printf '%s' "$iout") | head -8; fi
    fail=$((fail+1))
  fi
done < "$CORPUS"
echo "interp-vs-compiled gate: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
