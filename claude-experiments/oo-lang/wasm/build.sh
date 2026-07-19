#!/bin/sh
# Build the Scry VM to a browser-ready wasm32 module, then validate it.
# Requires the self-hosted coil (with the C0 shadow-stack + C1 function-table finalizer).
set -e
cd "$(dirname "$0")/.."
coil build src/main.coil --target wasm32-unknown-unknown -o wasm/scry.wasm
if command -v wasm-tools >/dev/null; then wasm-tools validate wasm/scry.wasm && echo "wasm/scry.wasm: VALID"; fi
ls -la wasm/scry.wasm
