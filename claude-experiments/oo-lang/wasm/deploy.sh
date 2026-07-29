#!/bin/sh
# Assemble wasm/dist/ — everything the in-browser demo needs and nothing else — ready to
# `vercel deploy` as a static site. There is no build step on the host: the wasm module is
# produced here by ./build.sh (which needs the self-hosted coil with the C0/C1 finalizer),
# so the deployment is prebuilt assets only.
#
# The layout MIRRORS the repo, because demo.html/index.html fetch ../viewer/*, ../examples/*,
# ../agent/core.scry and ../std/json.scry. Keeping the relative shape means the pages deploy
# byte-identical to the ones ./serve.sh serves — no path rewriting, nothing to drift.
set -e
cd "$(dirname "$0")/.."
DIST=wasm/dist

[ -f wasm/scry.wasm ] || { echo "wasm/scry.wasm missing — run wasm/build.sh first" >&2; exit 1; }

# The project link lives in dist/.vercel, but dist is rebuilt from scratch every run — keep it
# across the wipe so re-assembling doesn't orphan the Vercel project and create a second one.
LINK=$(mktemp -d)
[ -d "$DIST/.vercel" ] && cp -R "$DIST/.vercel" "$LINK/"

rm -rf "$DIST"
mkdir -p "$DIST/wasm/vendor" "$DIST/viewer/vendor" "$DIST/examples" "$DIST/agent" "$DIST/std"
[ -d "$LINK/.vercel" ] && cp -R "$LINK/.vercel" "$DIST/"
rm -rf "$LINK"

# the VM + the two pages
cp wasm/scry.wasm wasm/scry-wasm.js wasm/demo.html wasm/index.html wasm/demo.scry "$DIST/wasm/"
cp wasm/vendor/xterm.js wasm/vendor/xterm.css "$DIST/wasm/vendor/"

# the unmodified viewer
cp viewer/app.js viewer/style.css "$DIST/viewer/"
cp viewer/vendor/*.js "$DIST/viewer/vendor/"

# programs the pages load. examples/agent and examples/std are symlinks into the dirs below,
# so copy the .scry files only and let the two real directories serve the imports.
cp examples/*.scry "$DIST/examples/"
cp agent/core.scry "$DIST/agent/"
cp std/json.scry "$DIST/std/"

# `/` must REDIRECT, not rewrite: a rewrite serves demo.html while the URL stays `/`, so the
# page's own relative imports (./scry-wasm.js, ./scry.wasm, vendor/xterm.js) resolve against the
# root and 404. The browser has to actually land on /wasm/demo.html.
# No Content-Type rules here: Vercel already serves .wasm as application/wasm, and a `headers`
# entry can NOT override the content type of a static file anyway — verified against `vercel dev`,
# where the rule demonstrably matches (a marker header lands) while Content-Type stays put. So
# .scry files are served application/octet-stream; harmless, since the pages read them with
# response.text(), which decodes UTF-8 regardless.
cat > "$DIST/vercel.json" <<'JSON'
{
  "$schema": "https://openapi.vercel.sh/vercel.json",
  "cleanUrls": false,
  "redirects": [{ "source": "/", "destination": "/wasm/demo.html", "permanent": false }]
}
JSON

# The CLI falls back to a parent .gitignore when the deploy root has no ignore file, and the
# repo ignores wasm/scry.wasm — which is exactly the file that must ship. An explicit (empty)
# .vercelignore stops that fallback.
printf '# nothing ignored: this directory IS the deployment\n' > "$DIST/.vercelignore"

echo "assembled $DIST ($(du -sh "$DIST" | cut -f1))"
echo "  deploy:  cd $DIST && vercel deploy --prod"
echo "  verify:  SCRY_WASM_ROOT=$PWD/$DIST node wasm/ui-smoke-wasm.mjs"
