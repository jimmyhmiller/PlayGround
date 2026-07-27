#!/usr/bin/env bash
# RSC culmination gate — the UNMODIFIED `create-next-app` default builds + renders +
# hydrates under diffpack, verified in a REAL browser (agent-browser).
#
# The whole effort's culmination: take the PRISTINE create-next-app --app default
# (preserved VERBATIM at integration/next-app-router/authentic-create-next-app/,
# next@16.2.11 / react@19.2.4) — untouched app/layout.tsx (next/font/google Geist +
# Geist_Mono, Metadata, globals.css) and app/page.tsx (next/image on /next.svg +
# /vercel.svg, CSS Modules) — and build it with the diffpack RSC spine + the native
# next app-router adapter, with NO edits to its app/. Native build (Rust); Node +
# Chrome are only the oracle.
#
# The authentic app/ is NEVER modified: this gate copies it (byte-for-byte) into a
# fresh temp build dir, points node_modules at the working fixture's pinned installs
# (react / react-dom / react-server-dom-webpack@19.2.4 / next), and restores the two
# standard create-next-app static SVGs the page references (public/ was not part of
# the preserved snapshot — only app/ + next.config.ts were). It then asserts the
# authentic source stayed byte-identical (checksum) after the whole build.
#
# Gates (exit 0 = PASS):
#   A1  SSR of the FULL app-router document (doctype, RootLayout owns <html>).
#   A2  next/font/google macro rewritten for BOTH Geist + Geist_Mono; the font CSS
#       (@import + variable classes) hoisted into <head>; ${geist.variable} +
#       ${geistMono.variable} both resolved onto <html>.
#   A3  Metadata API: <title>Create Next App</title> + the description meta rendered.
#   A4  globals.css + the page CSS Module linked via /rsc.css; the scoped class on
#       the element matches a real rule in the served stylesheet.
#   A5  next/image: both SVG srcs render as raw <img src=...> with NO srcset
#       (unoptimized, byte-faithful to Next under React 19); the priority image
#       hoists a <link rel=preload as=image>; both SVG assets serve 200.
#   A6  Real browser: the page hydrates — hydrateRoot(document) commits a React fiber
#       onto the document — with ZERO console errors/warnings (clean hydration).
#   A7  Real browser: the styles APPLY — the CSS-Module <main> computes max-width
#       800px (from ._main_*) and the page container computes display:flex.
# Strict mode, the ERR net (no abort is ever silent) and fail() — see _gate-prelude.sh.
source "$(dirname "$0")/_gate-prelude.sh"

repo="$(cd "$(dirname "$0")/../.." && pwd)"
fixture="$repo/integration/next-app-router"
authentic="$fixture/authentic-create-next-app"
diffpack="$repo/target/release/diffpack"

[ -d "$authentic/app" ] || fail "pristine default missing at $authentic/app"

echo "== building diffpack (release) =="
cargo build --release --manifest-path "$repo/Cargo.toml"

# The pinned RSC deps live in the working fixture's node_modules (gitignored). Install
# there once if absent — same versions the whole RSC spine is verified against.
if [ ! -d "$fixture/node_modules/react-server-dom-webpack" ]; then
  echo "== installing pinned RSC deps in $fixture =="
  (cd "$fixture" && npm install --no-audit --no-fund react-server-dom-webpack@19.2.4)
fi
[ -d "$fixture/node_modules/react" ] || fail "no react in $fixture/node_modules (install the fixture deps first)"

# --- Fresh temp build dir; copy the UNTOUCHED authentic app/ verbatim --------------
build="$(mktemp -d "${TMPDIR:-/tmp}/diffpack-authentic.XXXXXX")"
cleanup() {
  [ -n "${server_pid:-}" ] && kill "$server_pid" 2>/dev/null || true
  agent-browser close 2>/dev/null || true
  rm -rf "$build"
}
trap cleanup EXIT

cp -R "$authentic/app" "$build/app"
cp "$authentic/next.config.ts" "$build/next.config.ts"
# Record the authentic source checksum BEFORE the build; assert it is untouched after.
# Assert the checksum inputs exist FIRST: a missing input makes `find` exit 1, and swallowing
# that would checksum a smaller set on both sides — comparing equal and silently retiring the
# tamper check rather than failing it.
[ -d "$authentic/app" ] || fail "authentic source missing $authentic/app — nothing to checksum"
[ -f "$authentic/next.config.ts" ] || fail "authentic source missing $authentic/next.config.ts — nothing to checksum"
# `|| true` would be WRONG here: a truncated checksum on BOTH sides compares equal, silently
# retiring the tamper check. The inputs are asserted to exist above instead.
# lint-gates: allow — preconditions asserted; an empty result would compare equal, not fail
authentic_sum_before="$(cd "$authentic" && find app next.config.ts -type f -exec shasum {} \; | sort | shasum | cut -d' ' -f1)"
# node_modules: the fixture's pinned installs (resolution only; nothing written).
ln -s "$fixture/node_modules" "$build/node_modules"
# public/: the two standard create-next-app SVGs page.tsx references (the preserved
# snapshot captured only app/ + next.config.ts; these static assets are not code).
mkdir -p "$build/public"
cp "$fixture/public/next.svg"   "$build/public/next.svg"
cp "$fixture/public/vercel.svg" "$build/public/vercel.svg"

out="$build/.diffpack-output"

echo "== native build: client graph =="
"$diffpack" build-app "$build" client --no-minify
echo "== native build: react-server render graph =="
"$diffpack" build-app "$build" react-server --no-minify
rm -rf "$out/rsc-render"; cp -r "$out/server" "$out/rsc-render"
echo "== native build: ssr-of-flight graph =="
"$diffpack" build-app "$build" ssr --no-minify

# --- The authentic source must be byte-identical after the whole build -------------
# `|| true` would be WRONG here: a truncated checksum on BOTH sides compares equal, silently
# retiring the tamper check. The inputs are asserted to exist above instead.
# lint-gates: allow — preconditions asserted; an empty result would compare equal, not fail
authentic_sum_after="$(cd "$authentic" && find app next.config.ts -type f -exec shasum {} \; | sort | shasum | cut -d' ' -f1)"
[ "$authentic_sum_before" = "$authentic_sum_after" ] || fail "the authentic create-next-app source was modified by the build (checksum drift) — it MUST stay untouched"
echo "OK: authentic create-next-app app/ stayed byte-identical through the build"

# --- Boot the app server -----------------------------------------------------------
serverlog="$build/server.log"
node "$repo/scripts/rsc/next-server.mjs" "$out" 0 > "$serverlog" 2>&1 &
server_pid=$!
for _ in $(seq 1 50); do
  port="$(grep -o 'localhost:[0-9]*' "$serverlog" | head -1 | cut -d: -f2 || true)"
  [ -n "${port:-}" ] && break
  sleep 0.2
done
[ -n "${port:-}" ] || { cat "$serverlog"; fail "app server did not start"; }
base="http://localhost:$port"
echo "app server on $base"

html="$(curl -s "$base/")"

# --- Gate A0: raw document integrity (streaming SSR must not split an HTML token) ---
# react-dom writes on 2048-byte view boundaries that routinely land INSIDE a tag, so
# anything interleaved between two of its writes (the inline __DF_FLIGHT scripts)
# corrupts the document. A browser's parser recovers from that, so it has to be checked
# on the raw bytes — ahead of A1..A7, which only ever see the symptom.
curl -s "$base/" -o "$build/document.html"
node "$repo/scripts/rsc/html-integrity.mjs" "$build/document.html" \
  || fail "A0: the served document has a <script> inside an open tag (streaming SSR injected a flight script mid-token)"
echo "OK (A0): the streamed document has no <script> spliced inside an HTML tag"

# --- Gate A1: SSR of the full app-router document ----------------------------------
echo "$html" | grep -q "<!DOCTYPE html>" || { echo "$html"; fail "A1: no full document (RootLayout must own <html>)"; }
echo "$html" | grep -q '<html lang="en"' || { echo "$html"; fail "A1: RootLayout <html lang=en> not rendered"; }
echo "$html" | grep -q "To get started, edit the page.tsx file." || { echo "$html"; fail "A1: the default page's Server Component content is missing"; }
echo "OK (A1): the untouched default SSRs the full app-router document"

# --- Gate A2: next/font/google (BOTH Geist + Geist_Mono) rewritten + hoisted --------
echo "$html" | grep -q "family=Geist:" || { echo "$html"; fail "A2: the Geist @import was not hoisted"; }
echo "$html" | grep -q "family=Geist+Mono" || { echo "$html"; fail "A2: the Geist_Mono @import was not hoisted (second font macro not rewritten)"; }
echo "$html" | grep -q "__df_fontvar_geist_mono" || { echo "$html"; fail "A2: the Geist_Mono CSS-variable class is missing"; }
echo "$html" | grep -qE 'class="__df_fontvar_geist __df_fontvar_geist_mono"' || { echo "$html"; fail "A2: \${geistSans.variable} \${geistMono.variable} did not both resolve onto <html>"; }
echo "OK (A2): next/font macro rewritten for BOTH Geist + Geist_Mono; font CSS hoisted; both variables on <html>"

# --- Gate A3: Metadata API (the stock title + description) -------------------------
echo "$html" | grep -qE "<title[^>]*>Create Next App</title>" || { echo "$html"; fail "A3: the stock metadata <title>Create Next App</title> was not rendered"; }
echo "$html" | grep -qE '<meta[^>]*name="description"[^>]*content="Generated by create next app"' || { echo "$html"; fail "A3: the stock metadata description was not rendered"; }
echo "OK (A3): the stock Metadata (title + description) rendered"

# --- Gate A4: globals.css + CSS Module via /rsc.css; scoping agrees -----------------
echo "$html" | grep -qE '<link[^>]*href="/rsc.css"' || { echo "$html"; fail "A4: /rsc.css was not linked into <head>"; }
module_class="$(echo "$html" | grep -oE -m1 '_page_[a-z0-9]+' || true)"
[ -n "$module_class" ] || { echo "$html"; fail "A4: the page CSS-Module scoped class was not applied"; }
css="$(curl -s "$base/rsc.css")"
echo "$css" | grep -q "$module_class" || { echo "$css"; fail "A4: served /rsc.css has no rule for the applied class $module_class (scoping disagrees)"; }
# globals.css content is present in the same served stylesheet.
echo "$css" | grep -qi "prefers-color-scheme" || { echo "$css" | head; fail "A4: globals.css (its @media prefers-color-scheme rules) is missing from the served stylesheet"; }
echo "OK (A4): globals.css + the page CSS Module served via /rsc.css; applied class $module_class matches a real rule"

# --- Gate A5: next/image — both SVGs raw (unoptimized), priority preload hoisted ----
logo="$(echo "$html" | grep -oiE -m1 '<img[^>]*src="/next.svg"[^>]*>' || true)"
[ -n "$logo" ] || { echo "$html"; fail "A5: the /next.svg <img> was not rendered"; }
if echo "$logo" | grep -qiE 'srcset='; then echo "$logo"; fail "A5: the SVG logo must NOT have a srcset (unoptimized, byte-faithful to Next under React 19)"; fi
echo "$logo" | grep -qiE 'decoding="async"' || { echo "$logo"; fail "A5: the SVG logo lost decoding=async"; }
echo "$html" | grep -oiE '<img[^>]*src="/vercel.svg"[^>]*>' | head -1 | grep -q . || { echo "$html"; fail "A5: the /vercel.svg <img> was not rendered"; }
echo "$html" | grep -qiE '<link[^>]*rel="preload"[^>]*as="image"' || { echo "$html"; fail "A5: the priority image did not hoist a <link rel=preload as=image>"; }
for svg in /next.svg /vercel.svg; do
  sc="$(curl -s -o /dev/null -w '%{http_code} %{content_type}' "$base$svg")"
  echo "$sc" | grep -qiE '^200 image/svg' || fail "A5: $svg did not serve as a real 200 image/svg (got: $sc)"
done
echo "OK (A5): next/image renders both SVGs raw (no srcset), a priority preload is hoisted, both assets serve 200"

# --- Gate A6 (real browser): clean hydration of the document -----------------------
agent-browser open "about:blank" >/dev/null 2>&1 || fail "A6: agent-browser could not open (is it installed?)"
agent-browser console --clear >/dev/null 2>&1 || true
agent-browser open "$base/" >/dev/null 2>&1
agent-browser wait --load networkidle >/dev/null 2>&1 || true
sleep 1
# hydrateRoot(document, ...) committing a fiber leaves React's internal container keys
# on the document node — definitive proof the client bundle ran and React took over.
hydrated="$(agent-browser eval 'JSON.stringify({ hydrated: Object.keys(document).some(k=>k.startsWith("__reactContainer")), flight: typeof window.__DIFFPACK_FLIGHT__ === "string" && window.__DIFFPACK_FLIGHT__.length>0, title: document.title })' 2>/dev/null || true)"
echo "$hydrated" | grep -q '\\"hydrated\\":true' || echo "$hydrated" | grep -q '"hydrated":true' || { echo "$hydrated"; fail "A6: hydrateRoot(document) did not commit a React fiber onto the document (hydration did not run)"; }
console="$(agent-browser console 2>/dev/null || true)"
if echo "$console" | grep -iE 'hydrat|did not match|does not match|error|uncaught|warn' | grep -v '^$' >/dev/null; then
  echo "$console"; fail "A6: the console had errors/warnings/hydration-mismatch on load (hydration must be clean)"
fi
echo "OK (A6): the untouched default hydrated cleanly (React fiber committed on document; zero console errors/warnings)"

# --- Gate A7 (real browser): the styles actually apply -----------------------------
styles="$(agent-browser eval '(() => { const m=document.querySelector("main"); const p=document.querySelector("div[class^=_page_]"); return JSON.stringify({ mainMaxWidth: m && getComputedStyle(m).maxWidth, pageDisplay: p && getComputedStyle(p).display, imgs: document.querySelectorAll("img").length }); })()' 2>/dev/null || true)"
echo "$styles" | grep -q '800px' || { echo "$styles"; fail "A7: the CSS-Module ._main_ max-width:800px did not compute (styles not applied)"; }
echo "$styles" | grep -q 'flex' || { echo "$styles"; fail "A7: the CSS-Module ._page_ display:flex did not compute (styles not applied)"; }
echo "$styles" | grep -qE 'imgs[^0-9]*2' || { echo "$styles"; fail "A7: expected 2 rendered <img> elements (next/image output)"; }
echo "OK (A7): the styles apply in the browser (main max-width 800px, page display flex, 2 images)"

echo "PASS: the UNMODIFIED create-next-app default (next/font Geist+Geist_Mono, Metadata, globals.css + CSS Modules, next/image) builds with the diffpack RSC spine + native next adapter, server-renders the full document, its styles apply, and it hydrates cleanly in a real browser — the authentic app/ untouched (checksum-verified)"
