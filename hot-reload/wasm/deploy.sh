#!/usr/bin/env bash
# Deploy the demo to Vercel. The page is three static files, so nothing is
# built on Vercel's side: build.sh compiles the wasm here, and this stages it
# into the Build Output API directory that `vercel deploy --prebuilt` uploads.
#
# Staging into .vercel/output is what lets `wasm/pkg` stay gitignored — the
# prebuilt upload takes that directory verbatim, ignore files and all.
#
# First run needs `vercel login`, and will ask which project to link.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
out="$here/.vercel/output"

"$here/build.sh"

rm -rf "$out"
mkdir -p "$out/static"
cp "$here/index.html" "$out/static/"
cp -R "$here/pkg" "$out/static/pkg"

# `version: 3` is the Build Output API version, not the demo's. With a bare
# static directory and no routes, Vercel serves the files as-is — including
# .wasm as application/wasm, which the streaming instantiate in the glue needs.
cat > "$out/config.json" <<'JSON'
{
  "version": 3
}
JSON

cd "$here"
exec vercel deploy --prebuilt "$@"
