#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$ROOT/scripts/install-local.sh" >/dev/null

GVBINDIR="$ROOT/target/graphviz" \
  dot -Kion -Tsvg "$ROOT/examples/diamond.dot" > "$ROOT/target/graphviz/diamond.svg"
GVBINDIR="$ROOT/target/graphviz" \
  dot -Kion -Tsvg "$ROOT/examples/ion-like.dot" > "$ROOT/target/graphviz/ion-like.svg"

echo "Wrote $ROOT/target/graphviz/diamond.svg"
echo "Wrote $ROOT/target/graphviz/ion-like.svg"
