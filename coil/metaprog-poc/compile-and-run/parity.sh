#!/usr/bin/env bash
# ENGINE PARITY: the compiled metaprogram engine (COIL_META=compiled) must produce
# BYTE-IDENTICAL emit-ir output to the interpreter engine for every corpus file.
# Both runs use the SAME binary (pass it as $1; it must be linked with
# -Wl,-export_dynamic) so the only variable is the engine.
# Run from the repo root:  metaprog-poc/compile-and-run/parity.sh /tmp/coil-meta
set -uo pipefail
cd "$(dirname "$0")/../.."
COIL=${1:?usage: parity.sh <engine-enabled coil binary>}
[ -x "$COIL" ] || { echo "not executable: $COIL"; exit 1; }

pass=0; fail=0; skip=0
check() {
  local f="$1"
  local ref cmp rc1 rc2
  ref=$("$COIL" emit-ir "$f" 2>&1); rc1=$?
  cmp=$(COIL_META=compiled "$COIL" emit-ir "$f" 2>&1); rc2=$?
  if [ $rc1 -ne 0 ] && [ $rc2 -ne 0 ]; then
    # both engines reject it — the DIAGNOSTIC must match too
    if [ "$ref" = "$cmp" ]; then pass=$((pass+1)); else
      echo "DIAG-DIFF $f"; diff <(echo "$ref") <(echo "$cmp") | head -6; fail=$((fail+1)); fi
    return
  fi
  if [ $rc1 -ne $rc2 ]; then echo "RC-DIFF $f (interp=$rc1 compiled=$rc2)"; echo "$cmp" | head -3; fail=$((fail+1)); return; fi
  if [ "$ref" = "$cmp" ]; then pass=$((pass+1)); else
    echo "IR-DIFF $f"; diff <(echo "$ref") <(echo "$cmp") | head -6; fail=$((fail+1)); fi
}

for f in examples/*.coil lib/*.coil metaprog-poc/*.coil; do
  [ -e "$f" ] || continue
  check "$f"
done
echo "parity: $pass identical, $fail divergent"
[ $fail -eq 0 ]
