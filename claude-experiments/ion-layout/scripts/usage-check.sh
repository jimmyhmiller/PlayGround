#!/bin/bash
# Real usage numbers straight from Claude Code (same data as the /usage
# screen). A long-running session must STOP once "Current week (all models)"
# crosses 50%.
set -euo pipefail
out="$(claude -p "/usage" 2>&1)"
echo "$out"
pct="$(echo "$out" | sed -n 's/.*Current week (all models): \([0-9]*\)% used.*/\1/p')"
if [ -n "$pct" ] && [ "$pct" -ge 50 ]; then
  echo
  echo "*** WEEKLY USAGE AT ${pct}% — OVER THE 50% STOP LINE. STOP WORKING. ***"
  exit 1
fi
