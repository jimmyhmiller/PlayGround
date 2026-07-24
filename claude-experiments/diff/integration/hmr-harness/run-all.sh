#!/usr/bin/env bash
#
# Run the HMR generality harness across every app config and report a summary.
#
#   integration/hmr-harness/run-all.sh [--offline] [--keep]
#
#   --offline  skip configs that clone from the network (run only in-repo apps);
#              use this for a fast, network-free CI smoke.
#   --keep     keep any cloned app dirs (passed through to hmr-check.sh).
#
# Exit 0 only if every selected app passed.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

OFFLINE=0; KEEP=""
for arg in "$@"; do
  case "$arg" in
    --offline) OFFLINE=1 ;;
    --keep) KEEP="--keep" ;;
    *) echo "unknown flag: $arg" >&2; exit 2 ;;
  esac
done

pass=0; fail=0; skipped=0; failed_apps=""
for conf in "$HERE"/apps/*.conf; do
  name="$(basename "$conf" .conf)"
  if [ "$OFFLINE" = "1" ] && grep -qE '^REPO=' "$conf"; then
    echo ">>> SKIP $name (network app, --offline)"; skipped=$((skipped+1)); continue
  fi
  echo ">>> $name"
  if "$HERE/hmr-check.sh" "$conf" $KEEP; then
    pass=$((pass+1))
  else
    fail=$((fail+1)); failed_apps="$failed_apps $name"
  fi
  echo
done

echo "=== HMR harness summary: $pass passed, $fail failed, $skipped skipped ==="
[ -n "$failed_apps" ] && echo "failed:$failed_apps"
[ "$fail" = "0" ]
