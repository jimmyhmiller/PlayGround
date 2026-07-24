#!/usr/bin/env bash
# RSC Slice C / R2 gate: build diffpack natively (Rust), then prove a `"use server"`
# exported function called through the CLIENT stub id round-trips to the SERVER
# implementation and returns its real result, against the pinned
# react-server-dom-webpack runtime. Native transforms; node is only the oracle.
# Exit 0 = gate PASS.
set -euo pipefail

repo="$(cd "$(dirname "$0")/../.." && pwd)"
fixture="${1:-$repo/integration/rsc-action}"

echo "== building diffpack (release) =="
cargo build --release --manifest-path "$repo/Cargo.toml"

if [ ! -d "$fixture/node_modules/react-server-dom-webpack" ]; then
  echo "== installing pinned RSC deps in $fixture =="
  (cd "$fixture" && npm install --no-audit --no-fund)
fi

echo "== node oracle: 'use server' action client→server round-trip =="
node "$repo/scripts/rsc/action-roundtrip.mjs" "$fixture"
