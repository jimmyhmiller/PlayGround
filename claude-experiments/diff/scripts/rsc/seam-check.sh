#!/usr/bin/env bash
# RSC Slice B / R3 gate: build the client graph natively (Rust), then prove a
# client reference resolves through the emitted client-references manifest + the
# `__webpack_*` seam to the real exported component. Native build; node is only the
# oracle. Exit 0 = gate PASS.
# Strict mode, the ERR net (no abort is ever silent) and fail() — see _gate-prelude.sh.
source "$(dirname "$0")/_gate-prelude.sh"

repo="$(cd "$(dirname "$0")/../.." && pwd)"
fixture="${1:-$repo/integration/rsc-seam}"

echo "== building diffpack (release) =="
cargo build --release --manifest-path "$repo/Cargo.toml"

echo "== native client build of $fixture =="
"$repo/target/release/diffpack" build-app "$fixture" client --no-minify

echo "== node oracle: resolve a client reference through the __webpack_* seam =="
node "$repo/scripts/rsc/seam-resolve.mjs" "$fixture/.diffpack-output"
