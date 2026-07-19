#!/usr/bin/env bash
# Build the browser demo: compile the LLVM-free core + host to wasm32 and
# generate the JS glue next to index.html.
#
# The pinned `wasm-bindgen` crate version must match the `wasm-bindgen` CLI on
# PATH; they share a schema version and refuse to work across a mismatch.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(dirname "$here")"

cargo build --manifest-path "$root/Cargo.toml" \
    -p livetype-wasm --target wasm32-unknown-unknown --release

wasm-bindgen \
    --target web \
    --no-typescript \
    --out-dir "$here/pkg" \
    "$root/target/wasm32-unknown-unknown/release/livetype_wasm.wasm"

echo "built: $here/pkg/livetype_wasm_bg.wasm ($(du -h "$here/pkg/livetype_wasm_bg.wasm" | cut -f1))"
