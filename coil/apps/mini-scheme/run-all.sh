#!/usr/bin/env bash
# Build the evaluator, then run every tests/*.scm through BOTH mini-scheme and
# Chez and compare. Run from mini-scheme/.
set -e
../coil build scheme.coil -o /tmp/mini-scheme >/dev/null
echo "program            mini-scheme        chez               GC (mini)"
echo "-----------------------------------------------------------------------------"
for t in tests/*.scm; do
  mine=$(/tmp/mini-scheme < "$t" 2>/tmp/gc)
  chez=$(chez --quiet 2>/dev/null <<CHEZ
$(sed '$d' "$t")
(display $(tail -1 "$t"))(newline)
CHEZ
)
  gc=$(sed 's/\[gc\] //' /tmp/gc)
  flag=$([ "$mine" = "$chez" ] && echo "✓" || echo "✗ MISMATCH")
  printf "%-18s %-18s %-18s %s  %s\n" "$(basename $t)" "$mine" "$chez" "$flag" "$gc"
done
