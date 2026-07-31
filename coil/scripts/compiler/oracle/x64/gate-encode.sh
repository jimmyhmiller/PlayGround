#!/usr/bin/env bash
# Encoder gate for the x86-64 backend: every instruction x64.coil emits must be
# byte-identical to what a real assembler produces for the same mnemonic.
#
# src/compiler/x64_selftest.coil prints one case per line as
#     <hex>TAB<AT&T assembly>
# We feed the assembly column to llvm-mc --show-encoding and diff its bytes
# against the hex column. A mismatch is a hand-encoding bug in x64.coil.
#
# This gate has teeth precisely because it does not trust the encoder's own
# opinion of what it emitted: llvm-mc is an independent implementation.
#
# Usage: scripts/compiler/oracle/x64/gate-encode.sh <coil-binary>
#   VERBOSE=1  print every case, not just failures
set -uo pipefail
cd "$(dirname "$0")/../../../.."
BIN=${1:?usage: gate-encode.sh <coil-binary>}
MC=${LLVM_MC:-llvm-mc}

[ -x "$BIN" ] || { echo "GATE FAIL: binary not executable: $BIN"; exit 2; }
command -v "$MC" >/dev/null 2>&1 || { echo "GATE FAIL: no llvm-mc (set LLVM_MC=/path/to/llvm-mc)"; exit 2; }

CASES=$(mktemp); trap 'rm -f "$CASES" "$CASES.asm" "$CASES.mc" "$CASES.bin"' EXIT
"$BIN" run src/compiler/x64_selftest.coil > "$CASES" || {
  echo "GATE FAIL: x64_selftest.coil did not run"; exit 2; }

# Assemble every mnemonic in ONE llvm-mc invocation (per-line would be ~180
# process spawns); the encodings come back in order, one per line.
sed 's/#wide$//' "$CASES" | cut -f2 > "$CASES.asm"
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

# A few encoders deliberately emit a WIDER form than llvm-mc's canonical choice:
# the frame-prologue instructions are patched in place after the body is emitted,
# so their length must not depend on the immediate's value. For those, byte
# equality against llvm-mc is the wrong test — what matters is that the bytes
# DISASSEMBLE to the intended instruction. They are marked with a "#wide" suffix
# on the assembly column and checked with objdump instead.
# Compare an encoding against an expected mnemonic by DISASSEMBLING it. objdump
# and our AT&T spelling differ in inessential ways (hex vs decimal immediates,
# the `q` size suffix, spacing), so both sides are normalised to
# `mnemonic operand,operand` with immediates in decimal before comparing.
disasm_matches() {  # disasm_matches <hex> <att-asm>
  printf '%s' "$1" | sed 's/../\\x&/g' | xargs -0 printf > "$CASES.bin" 2>/dev/null
  d=$(objdump -D -b binary -m i386:x86-64 -M att "$CASES.bin" 2>/dev/null \
      | grep -E '^\s+0:' | sed 's/.*\t//')
  norm() {
    # strip the AT&T size suffix while the mnemonic is still a separate word,
    # THEN remove spacing/sigils — doing it in the other order leaves `subq`
    # glued to its operands and the suffix can no longer be matched.
    printf '%s' "$1" \
      | sed 's/^\([a-z][a-z]*\)q /\1 /' \
      | tr -d ' %$' \
      | awk '{ while (match($0, /0x[0-9a-f]+/)) {
                 h = substr($0, RSTART, RLENGTH);
                 v = strtonum(h);
                 $0 = substr($0,1,RSTART-1) v substr($0,RSTART+RLENGTH);
               } print }'
  }
  [ "$(norm "$d")" = "$(norm "$2")" ]
}

pass=0; fail=0; first=""
while IFS= read -r line && IFS= read -r want <&3; do
  got=$(printf '%s' "$line" | cut -f1)
  asm=$(printf '%s' "$line" | cut -f2)
  case "$asm" in
    *"#wide")
      asm=${asm%%#wide}; asm=${asm% }
      if disasm_matches "$got" "$asm"; then
        pass=$((pass+1))
        [ "${VERBOSE:-}" = 1 ] && printf '  ok   %-34s %s (wide form)\n' "$asm" "$got"
      else
        fail=$((fail+1)); [ -z "$first" ] && first="$asm (wide form did not disassemble back)"
        printf '  FAIL %-34s encoder=%s does not disassemble to it\n' "$asm" "$got"
      fi
      continue;;
  esac
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
