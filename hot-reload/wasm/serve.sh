#!/usr/bin/env bash
# Serve the demo. `no-store` so a rebuilt .wasm is never served from cache.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
port="${1:-8080}"

cd "$here"
echo "http://localhost:$port/"
exec python3 -c '
import functools, http.server, sys

class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

port = int(sys.argv[1])
http.server.ThreadingHTTPServer(("", port), Handler).serve_forever()
' "$port"
