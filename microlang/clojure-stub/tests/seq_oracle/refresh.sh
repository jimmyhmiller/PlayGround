#!/usr/bin/env bash
# Regenerate expected.txt from REAL Clojure. Run this only when probe.clj
# changes -- expected.txt is Clojure's answer, never a hand-edited one.
#
# `clojure` must run from a directory that does not shadow core.clj.
set -euo pipefail
cd "$(dirname "$0")"
HERE="$PWD"

# Write to a temp file and move only on success. A probe form that throws (say,
# calling a ClojureScript-only fn) makes clojure die PART WAY THROUGH, and
# redirecting straight into expected.txt would silently enshrine a truncated
# spec that the test then happily "passes" against.
TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT
if ! (cd /tmp && clojure -M "$HERE/probe.clj") > "$TMP"; then
  echo "clojure exited non-zero -- expected.txt NOT updated." >&2
  echo "Re-run without redirect to see the error:" >&2
  echo "  (cd /tmp && clojure -M $HERE/probe.clj)" >&2
  exit 1
fi
# Every line must be "label<TAB>value"; anything else means a stray print or a
# partial write.
if grep -qvE '^[^	]+	' "$TMP"; then
  echo "probe emitted a line that is not label<TAB>value -- expected.txt NOT updated:" >&2
  grep -nvE '^[^	]+	' "$TMP" >&2
  exit 1
fi
mv "$TMP" "$HERE/expected.txt"
trap - EXIT
echo "expected.txt refreshed: $(wc -l < "$HERE/expected.txt") lines from real Clojure." >&2
