#!/usr/bin/env bash
# THE DIAGNOSTIC-PARITY GATE. Diff a self-hosted compiler's rendered human
# diagnostic (the STDERR of `emit-ir FILE`) against the Rust reference snapshot
# across the error corpus. Exit 0 iff byte-identical for EVERY input.
#
# This gate is what makes "the self-host renders errors the same way Rust does"
# mechanically enforced. The Rust side renders `error: … / --> file:line:col /
# source line / ^^^ / note: in expansion of macro … / N errors`; the self-host
# currently only prints `(error@lo:hi "msg")`, so this gate FAILS wholesale until
# the diagnostic renderer (src/span.rs render/render_all + main.rs report) is
# ported. The failing set is the 'diagnostics' gap.
#
# Usage: gate-diag.sh <coil-self-binary>
#   COIL_SELF_ARGS  extra args to pass before `emit-ir` (default none)
#   VERBOSE=1       print the first differing hunk for each failure
set -uo pipefail
cd "$(dirname "$0")/../.."          # repo root
ROOT=$(pwd)
BIN=${1:?usage: gate-diag.sh <coil-self-binary>}
REF=selfhost/oracle/diag/reference
LIST=selfhost/oracle/diag/corpus.txt

[ -x "$BIN" ]  || { echo "GATE FAIL: self-host binary not executable: $BIN"; exit 2; }
[ -f "$LIST" ] || { echo "GATE FAIL: no diag snapshot; run snapshot-diag.sh first"; exit 2; }

pass=0; fail=0; first_fail=""
while IFS= read -r f; do
  [ -z "$f" ] && continue
  ref="$REF/$(echo "$f" | tr '/' '_').diag"
  got=$("$BIN" ${COIL_SELF_ARGS:-} emit-ir "$f" 2>&1 | sed "s|$ROOT/||g")
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

# ---- build-path inputs: compare BOTH combined output AND exit code -----------
# Drives `coil build` for the halt-on-error / link-failure / static-assert cases,
# with an extensionless -o in a scratch dir (see snapshot-diag.sh for why the
# object name — hence any linker diagnostic — stays byte-identical). A build gate
# case passes iff the self-host's stdout+stderr AND its exit code both match the
# Rust reference exactly.
BLIST=selfhost/oracle/diag/build-corpus.txt
if [ -f "$BLIST" ]; then
  TMPD=$(mktemp -d)
  while IFS= read -r f; do
    [ -z "$f" ] && continue
    ref="$REF/$(echo "$f" | tr '/' '_').diag"
    refec="$REF/$(echo "$f" | tr '/' '_').exit"
    o="$TMPD/$(basename "$f" .coil)"
    "$BIN" ${COIL_SELF_ARGS:-} build "$f" -o "$o" > "$TMPD/raw" 2>&1
    code=$?
    got=$(sed "s|$ROOT/||g; s|$TMPD/||g" "$TMPD/raw")
    want=$(cat "$ref")
    wantec=$(cat "$refec" 2>/dev/null)
    if [ "$got" = "$want" ] && [ "$code" = "$wantec" ]; then
      pass=$((pass+1))
    else
      fail=$((fail+1)); [ -z "$first_fail" ] && first_fail="$f (build mismatch)"
      if [ "${VERBOSE:-}" = 1 ]; then
        echo "FAIL(build) $f  exit got=$code want=$wantec"
        diff <(printf '%s' "$want") <(printf '%s' "$got") | head -14
      fi
    fi
  done < "$BLIST"
  rm -rf "$TMPD"
fi

echo "gate-diag: $pass pass, $fail fail"
if [ $fail -ne 0 ]; then
  echo "first failure: $first_fail"
  exit 1
fi
echo "GATE PASS — self-host diagnostics are byte-identical to the Rust reference"
