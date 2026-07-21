#!/usr/bin/env bash
# Behavioral gate for the Linux x86-64 port: build every corpus program with the
# DEFAULT (LLVM) backend on the Linux host, run it, and diff stdout+exit against
# the SAME reference snapshots the arm64 gate uses (selfhost/oracle/arm64/reference)
# — program behavior is the cross-platform contract, so one snapshot serves both.
#
# The executable is copied to the same fixed /tmp/coil-arm64-fixed-<id> path the
# arm64 gate uses because argv[0] appears in reference stdout (examples/args.coil).
#
# Two corpus entries declare arm64-REGISTER shim conventions (examples/shim.coil,
# examples/everything.coil). On an x86 host those must fail with the per-arch
# diagnostic — asserted here, never skipped silently.
#
# usage: selfhost/oracle/linux/gate-run.sh <coil-binary> [--verbose]
set -uo pipefail
cd "$(dirname "$0")/../../.."
BIN="${1:?usage: gate-run.sh <coil-binary>}"
VERBOSE="${2:-}"
HERE=selfhost/oracle/arm64
REF="$HERE/reference"
ARM64_ONLY="examples/shim.coil examples/everything.coil"
pass=0; fail=0
while IFS= read -r line; do
  [ -z "$line" ] && continue
  case "$line" in \#*) continue;; esac
  set -- $line
  if [ "$1" = "R" ]; then shift; fi
  f="$1"; shift
  id=$(echo "$f" | tr '/.' '__')
  exe="/tmp/coil-arm64-fixed-$id"
  case " $ARM64_ONLY " in
    *" $f "*)
      out=$(timeout 120 "$BIN" build "$f" -o "$exe" 2>&1); rc=$?
      if [ "$rc" != 0 ] && echo "$out" | grep -q "not a general-purpose register on the target architecture"; then
        pass=$((pass+1)); [ -n "$VERBOSE" ] && echo "ok  $f (per-arch error, as designed on x86)"
      else
        echo "FAIL(arch-error) $f rc=$rc: $(echo "$out" | head -1)"; fail=$((fail+1))
      fi
      continue;;
  esac
  if ! timeout 120 "$BIN" build "$f" -o "$exe" >/dev/null 2>"/tmp/coil-linux-gate-$id.buildlog"; then
    echo "FAIL(build) $f"; [ -n "$VERBOSE" ] && head -3 "/tmp/coil-linux-gate-$id.buildlog"
    fail=$((fail+1)); continue
  fi
  timeout 30 "$exe" "$@" </dev/null >"/tmp/coil-linux-gate-$id.out" 2>/dev/null
  code=$?
  refcode=$(cat "$REF/$id.exit" 2>/dev/null)
  if cmp -s "/tmp/coil-linux-gate-$id.out" "$REF/$id.stdout" && [ "$code" = "$refcode" ]; then
    pass=$((pass+1)); [ -n "$VERBOSE" ] && echo "ok  $f"
  else
    echo "FAIL(run) $f  exit=$code want=$refcode"
    if [ -n "$VERBOSE" ]; then
      diff "/tmp/coil-linux-gate-$id.out" "$REF/$id.stdout" | head -6
    fi
    fail=$((fail+1))
  fi
done < "$HERE/corpus.txt"
echo "linux gate: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
