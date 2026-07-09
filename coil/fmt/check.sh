#!/usr/bin/env bash
# fmt/check.sh — safety gate for the Coil formatter.
#
# For every .coil under the given roots (default: examples lib selfhost/src apps):
#   1. token-equivalence : format(f) must read back to the SAME node tree as f
#                          (formatting may only change whitespace/line breaks, never
#                           add/drop/reorder/comment-out a token) — the correctness gate.
#   2. idempotence       : format(format(f)) == format(f).
#
# Run from the repo root:  bash fmt/check.sh
set -u
COIL=${COIL:-./target/release/coil}
ROOTS=${*:-"examples lib selfhost/src apps"}
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
strip() { sed -E 's/\[nl=[0-9]+\] //'; }

pass=0; fail=0
for f in $(find $ROOTS -name '*.coil' 2>/dev/null | sort); do
  if ! $COIL run fmt/fmt.coil -- "$f" > "$TMP/1" 2>"$TMP/err"; then
    echo "FMT-ERR   $f -- $(head -1 "$TMP/err")"; fail=$((fail+1)); continue
  fi
  # idempotence
  $COIL run fmt/fmt.coil -- "$TMP/1" > "$TMP/2" 2>/dev/null
  if ! diff -q "$TMP/1" "$TMP/2" >/dev/null; then
    echo "NOT-IDEMP $f"; fail=$((fail+1)); continue
  fi
  # token-equivalence
  $COIL run fmt/dump.coil -- "$f"     2>/dev/null | strip > "$TMP/a"
  $COIL run fmt/dump.coil -- "$TMP/1" 2>/dev/null | strip > "$TMP/b"
  if ! diff -q "$TMP/a" "$TMP/b" >/dev/null; then
    echo "TOKEN-DIFF $f"; diff "$TMP/a" "$TMP/b" | head -6; fail=$((fail+1)); continue
  fi
  pass=$((pass+1))
done
echo "=== $pass pass, $fail fail ==="
[ "$fail" -eq 0 ]
