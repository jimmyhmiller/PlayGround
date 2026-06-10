#!/usr/bin/env bash
# Full verification: build the Rust core + C plugin, run the property test
# suite, then machine-check rendered geometry for every corpus graph.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "── build ──────────────────────────────────────────"
./scripts/build.sh

echo "── property tests (cargo test) ────────────────────"
cargo test --release --quiet

echo "── rendered-geometry checks (dot -Kion -Tjson) ────"
node scripts/verify.mjs

echo
echo "All checks passed."
