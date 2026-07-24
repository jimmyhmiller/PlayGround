#!/usr/bin/env bash
# RSC flight gate (node-only, no browser) — build all three RSC graphs natively
# (Rust), then prove diffpack's flight runtime renders the Server Component <Page/>
# to correct SSR HTML against the pinned react-server-dom-webpack:
#   1. the REACT-SERVER render (its own inlined react-server React, in a child)
#      turns <Page/> — which embeds a hook-bearing "use client" island and a
#      "use server" action — into a flight stream, serializing the island as a
#      client reference via diffpack's client-references manifest;
#   2. the SSR bundle (its own inlined React + the island as real code) consumes
#      that flight with createFromReadableStream + the divergent-id
#      serverConsumerManifest (client ids joined to the SSR graph's own ids) and
#      react-dom renders it to HTML.
# Native build + native transforms + native manifests; node is only the oracle.
# This is the fast node-only subset of scripts/rsc/rsc-check.sh (the full browser
# gate). Exit 0 = gate PASS.
set -euo pipefail

repo="$(cd "$(dirname "$0")/../.." && pwd)"
fixture="${1:-$repo/integration/rsc-reference}"
diffpack="$repo/target/release/diffpack"
output="$fixture/.diffpack-output"

echo "== building diffpack (release) =="
cargo build --release --manifest-path "$repo/Cargo.toml"

if [ ! -d "$fixture/node_modules/react-server-dom-webpack" ]; then
  echo "== installing pinned RSC deps in $fixture =="
  (cd "$fixture" && npm install --no-audit --no-fund)
fi

echo "== native build: client, react-server, ssr graphs of $fixture =="
"$diffpack" build-app "$fixture" client --no-minify
"$diffpack" build-app "$fixture" react-server --no-minify
# The react-server and ssr graphs both emit server/server.mjs; snapshot the
# react-server output aside before the ssr build overwrites server/.
rm -rf "$output/rsc-render"
cp -r "$output/server" "$output/rsc-render"
"$diffpack" build-app "$fixture" ssr --no-minify

echo "== node oracle: flight render -> SSR html of <Page/> (server component + use-client island) =="
node "$repo/scripts/rsc/flight-render.mjs" "$fixture"
