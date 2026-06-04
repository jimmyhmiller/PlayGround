#!/usr/bin/env bash
# Oracle gate for the React-compiler-on-JSLIR port.
#
#   run-oracle.sh                 # summary
#   run-oracle.sh --json          # machine-readable
#   run-oracle.sh --list mismatch # names in a bucket
#   run-oracle.sh --filter useMemo --limit 50
#   run-oracle.sh --regen         # re-extract cache/ from the pinned upstream (tools/extract.js)
#
# Offline by default: diffs against the committed cache/. Needs `node` on PATH (for the
# snap-exact normalizer) and the crate's node_modules (npm install once).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

if [[ "${1:-}" == "--regen" ]]; then
  [[ -d "$HERE/node_modules" ]] || (cd "$HERE" && npm install --no-audit --no-fund)
  node "$HERE/tools/extract.js"
  exit 0
fi

[[ -d "$HERE/node_modules" ]] || (cd "$HERE" && npm install --no-audit --no-fund)
cd "$REPO"
exec cargo run --release -q -p jsir-react-oracle -- "$@"
