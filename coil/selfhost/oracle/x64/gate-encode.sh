#!/usr/bin/env bash
# Encoder gate for the x86-64 backend: every instruction x64.coil emits must be
# byte-identical to what a real assembler produces for the same mnemonic.
#
# selfhost/src/x64_selftest.coil prints one case per line as
#     <hex>TAB<AT&T assembly>
# We feed the assembly column to llvm-mc --show-encoding and diff its bytes
# against the hex column. A mismatch is a hand-encoding bug in x64.coil.
#
# This gate has teeth precisely because it does not trust the encoder's own
# opinion of what it emitted: llvm-mc is an independent implementation.
#
# Usage: selfhost/oracle/x64/gate-encode.sh <coil-binary>
#   VERBOSE=1  print every case, not just failures
set -uo pipefail
cd "$(dirname "$0")/../../.."
BIN=${1:?usage: gate-encode.sh <coil-binary>}
MC=${LLVM_MC:-llvm-mc}

[ -x "$BIN" ] || { echo "GATE FAIL: binary not executable: $BIN"; exit 2; }
command -v "$MC" >/dev/null 2>&1 || { echo "GATE FAIL: no llvm-mc (set LLVM_MC=/path/to/llvm-mc)"; exit 2; }

CASES=$(mktemp); trap 'rm -f "$CASES" "$CASES.asm" "$CASES.mc"' EXIT
"$BIN" run selfhost/src/x64_selftest.coil > "$CASES" || {
  echo "GATE FAIL: x64_selftest.coil did not run"; exit 2; }

# Assemble every mnemonic in ONE llvm-mc invocation (per-line would be ~180
# process spawns); the encodings come back in order, one per line.
cut -f2 "$CASES" > "$CASES.asm"
"$MC" -triple=x86_64-unknown-linux-gnu --show-encoding < "$CASES.asm" 2>/dev/null \
  | grep -o 'encoding: \[[^]]*\]' \
  | sed 's/encoding: \[//; s/\]//; s/0x//g; s/,//g' > "$CASES.mc"

n_cases=$(wc -l < "$CASES"); n_mc=$(wc -l < "$CASES.mc")
if [ "$n_cases" != "$n_mc" ]; then
  echo "GATE FAIL: $n_cases cases but llvm-mc returned $n_mc encodings (an mnemonic did not assemble)"
  # show which line llvm-mc choked on
  "$MC" -triple=x86_64-unknown-linux-gnu --show-encoding < "$CASES.asm" 2>&1 >/dev/null | head -5
  exit 1
fi

pass=0; fail=0; first=""
while IFS= read -r line && IFS= read -r want <&3; do
  got=$(printf '%s' "$line" | cut -f1)
  asm=$(printf '%s' "$line" | cut -f2)
  if [ "$got" = "$want" ]; then
    pass=$((pass+1))
    [ "${VERBOSE:-}" = 1 ] && printf '  ok   %-34s %s\n' "$asm" "$got"
  else
    fail=$((fail+1))
    [ -z "$first" ] && first="$asm (encoder: $got, llvm-mc: $want)"
    printf '  FAIL %-34s encoder=%s llvm-mc=%s\n' "$asm" "$got" "$want"
  fi
done < "$CASES" 3< "$CASES.mc"

echo "x64 gate-encode: $pass passed, $fail failed"
[ "$fail" = 0 ] || { echo "first failure: $first"; exit 1; }
