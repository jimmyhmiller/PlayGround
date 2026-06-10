#!/usr/bin/env bash
# Run the parity oracle: diff this port's layout against the original
# essence.ts on examples, real ion dumps, and randomized in-domain graphs.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IONGRAPH="/Users/jimmyhmiller/Documents/Code/open-source/iongraph"

cd "$ROOT"
cargo build --release --quiet --bin ion-dump

# tsx must resolve from a directory that has it; install once into npx cache.
TSX_LOADER="$(find ~/.npm/_npx -path '*node_modules/tsx/dist/loader.mjs' 2>/dev/null | head -1)"
if [ -z "$TSX_LOADER" ]; then
  npx -y tsx --version > /dev/null
  TSX_LOADER="$(find ~/.npm/_npx -path '*node_modules/tsx/dist/loader.mjs' 2>/dev/null | head -1)"
fi

cd "$IONGRAPH"
node --stack-size=8000 --import "$TSX_LOADER" "$ROOT/scripts/parity.ts"
