#!/bin/sh
# Serve the repo so wasm/demo.html can fetch ../viewer/* and ../agent/*.
# Everything is static — no server-side scry, no network at runtime.
set -e
cd "$(dirname "$0")/.."
PORT="${PORT:-8777}"
echo "scry wasm demo:  http://127.0.0.1:$PORT/wasm/demo.html"
echo "viewer only:     http://127.0.0.1:$PORT/wasm/index.html"
exec python3 -m http.server "$PORT" --bind 127.0.0.1
