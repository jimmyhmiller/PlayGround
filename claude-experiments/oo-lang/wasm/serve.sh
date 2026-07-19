#!/bin/sh
# Serve the repo so wasm/demo.html can fetch ../viewer/* and ../agent/*.
# Everything is static — no server-side scry, no network at runtime.
#
# Sends `Cache-Control: no-store` on every response, exactly like the native scry server
# (docs: "so you never get a stale app.js"). Without it the browser happily reuses a cached
# scry.wasm / app.js after a rebuild and you see the OLD build after reloading.
set -e
cd "$(dirname "$0")/.."
PORT="${PORT:-8777}"
echo "scry wasm demo:  http://127.0.0.1:$PORT/wasm/demo.html"
echo "viewer only:     http://127.0.0.1:$PORT/wasm/index.html"
exec python3 - "$PORT" <<'PY'
import sys, functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

class NoStore(SimpleHTTPRequestHandler):
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map, ".wasm": "application/wasm",
                      ".scry": "text/plain", ".mjs": "text/javascript"}
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        super().end_headers()
    def log_message(self, *a): pass

ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), NoStore).serve_forever()
PY
