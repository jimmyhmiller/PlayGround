#!/usr/bin/env bash
# Regenerate expected.txt from REAL Clojure. Run this only when probe.clj
# changes -- expected.txt is Clojure's answer, never a hand-edited one.
#
# `clojure` must run from a directory that does not shadow core.clj.
set -euo pipefail
cd "$(dirname "$0")"
HERE="$PWD"
(cd /tmp && clojure -M "$HERE/probe.clj") > "$HERE/expected.txt"
echo "expected.txt refreshed from $(clojure -M -e '(println (clojure-version))' 2>/dev/null || echo clojure)" >&2
