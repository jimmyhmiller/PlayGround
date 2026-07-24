#!/usr/bin/env bash
# RSC Slice SSG-1 gate — FULL SSG (build-time static prerender) for a REAL Next.js
# app-router app, served by a DUMB static file server (zero per-request render, zero
# child processes), verified in a REAL browser (agent-browser).
#
# The pipeline diffpack already uses to SERVE (react-server render child -> flight ->
# SSR-of-flight -> full HTML document) is run AHEAD OF TIME by `build-app <root> static`
# and written to `.diffpack-output/static/<route>.html` + `<route>.rsc`. This gate:
#   1. Native build (Rust): client -> react-server (cp -> rsc-render) -> ssr -> static.
#   2. FILES ON DISK: index/about/products a+b .html + .rsc exist (full documents +
#      raw flight); dynamic routes (blog/[slug], go, error-demo) are SKIPPED and
#      recorded in prerender-manifest.json with a reason (never silently dropped).
#   3. STRUCTURAL: the dumb server imports NEITHER RSC bundle and spawns NO child.
#   4. DUMB SERVE: curl byte-equals the on-disk files; ?__rsc=1 is raw flight;
#      a dynamic path 501s (never index HTML, never a render); its .rsc 404s.
#   5. REAL BROWSER: a prerendered page HYDRATES with ZERO console errors, its island
#      is interactive (count 5 -> 6), and a next/link soft-navigates to the prerendered
#      /about (?__rsc=1 diff-render, #app-shell preserved), history.back restores /.
# Native build (Rust); the prerender + serve run the app's own React / plain fs (the
# allowed oracle). Exit 0 = gate PASS.
set -euo pipefail

repo="$(cd "$(dirname "$0")/../.." && pwd)"
fixture="${1:-$repo/integration/next-app-router}"
diffpack="$repo/target/release/diffpack"
output="$fixture/.diffpack-output"
static="$output/static"

fail() { echo "FAIL: $*" >&2; exit 1; }

echo "== building diffpack (release) =="
cargo build --release --manifest-path "$repo/Cargo.toml"

if [ ! -d "$fixture/node_modules/react-server-dom-webpack" ]; then
  echo "== installing pinned RSC deps in $fixture =="
  (cd "$fixture" && npm install --no-audit --no-fund react-server-dom-webpack@19.2.4)
fi

echo "== native build: client -> react-server (cp -> rsc-render) -> ssr -> static =="
"$diffpack" build-app "$fixture" client --no-minify >/dev/null
"$diffpack" build-app "$fixture" react-server --no-minify >/dev/null
rm -rf "$output/rsc-render"
cp -r "$output/server" "$output/rsc-render"
"$diffpack" build-app "$fixture" ssr --no-minify >/dev/null
"$diffpack" build-app "$fixture" static

# --- Gate 1: files on disk ------------------------------------------------------
for f in index about products/a products/b; do
  [ -s "$static/$f.html" ] || fail "missing prerendered HTML $static/$f.html"
  grep -q "<!DOCTYPE html>" "$static/$f.html" || fail "$f.html is not a full document (no doctype)"
  grep -q 'id="app-shell"' "$static/$f.html" || fail "$f.html missing root-layout wrapper (#app-shell)"
  grep -q "__DIFFPACK_FLIGHT__" "$static/$f.html" || fail "$f.html has no inlined flight (__DIFFPACK_FLIGHT__)"
done
sed 's/<!--[^>]*-->//g' "$static/products/a.html" | grep -q "product: a" || fail "products/a.html missing 'product: a'"
sed 's/<!--[^>]*-->//g' "$static/products/b.html" | grep -q "product: b" || fail "products/b.html missing 'product: b'"
for f in index about products/a products/b; do
  [ -s "$static/$f.rsc" ] || fail "missing raw flight $static/$f.rsc"
  head -c 20 "$static/$f.rsc" | grep -qi "<!DOCTYPE" && fail "$f.rsc is HTML, not raw flight"
done
# Exactly two products files (one per generateStaticParams combo); no literal [id].
[ -f "$static/products/[id].html" ] && fail "a literal products/[id].html was written (SSG enumeration broken)"
prod_count="$(ls "$static/products"/*.html | wc -l | tr -d ' ')"
[ "$prod_count" = "2" ] || fail "expected 2 products html files, found $prod_count"
# Dynamic routes NOT prerendered.
[ -e "$static/go.html" ] && fail "go.html was prerendered (force-dynamic must be skipped)"
[ -e "$static/error-demo.html" ] && fail "error-demo.html was prerendered (force-dynamic must be skipped)"
ls "$static/blog"/*.html >/dev/null 2>&1 && fail "a blog/*.html was prerendered (dynamic must be skipped)"
# Manifest records the skipped dynamic routes with a reason.
manifest="$static/prerender-manifest.json"
[ -s "$manifest" ] || fail "prerender-manifest.json missing"
for p in "/blog/\[slug\]" "/go" "/error-demo"; do
  grep -q "$p" "$manifest" || fail "prerender-manifest.json does not record dynamic route $p"
done
grep -q "reason" "$manifest" || fail "prerender-manifest.json dynamic entries carry no reason"
# public/ colocated.
[ -f "$static/rsc.css" ] || fail "public/ not colocated into static/ (rsc.css missing)"
echo "OK (gate 1): static+ssg .html+.rsc on disk (index/about/products a,b); dynamic skipped in manifest; public colocated"

# --- Gate 1b: classification PRECEDENCE — generateStaticParams does NOT beat a request read
# /blog/[slug] EXPORTS generateStaticParams (the fixture) yet ALSO reads cookies(). Next
# classifies it ƒ Dynamic (verified: `next build` reports ƒ, not ●) because a request-state
# read opts the whole route into dynamic rendering. diffpack must reproduce that precedence:
# blog is SKIPPED at prerender (no static .html) and recorded Dynamic with a reason that
# NAMES the precedence — never falsely "no generateStaticParams", never silently prerendered.
blog_page="$fixture/app/blog/[slug]/page.tsx"
grep -q "generateStaticParams" "$blog_page" || fail "fixture /blog/[slug] must export generateStaticParams (precedence exemplar) — got none in $blog_page"
# It must NOT appear under the prerendered `static` list, and its recorded reason must name
# the precedence over generateStaticParams (proving classification, not an accidental skip).
python3 - "$manifest" <<'PY' || fail "prerender-manifest.json precedence assertion failed"
import json, sys
m = json.load(open(sys.argv[1]))
assert "/blog/[slug]" not in m["static"], "/blog/[slug] must NOT be prerendered static (it reads cookies)"
reason = next(d["reason"] for d in m["dynamic"] if d["path"] == "/blog/[slug]")
assert "despite generateStaticParams" in reason, f"/blog/[slug] reason must name the precedence over generateStaticParams, got: {reason!r}"
# The clean SSG contrast: /products/[id] enumerated one static path per generateStaticParams entry.
assert "/products/a" in m["static"] and "/products/b" in m["static"], "products SSG enumeration missing from static list"
print("precedence OK: /blog/[slug] is Dynamic DESPITE generateStaticParams; /products a,b are SSG")
PY
# And the react-server graph's staticparams op refuses to enumerate a non-Ssg route (blog),
# so a cookie-reading route can never be mistaken for statically enumerable.
sp_err="$(node "$output/rsc-render/server.mjs" staticparams "/blog/[slug]" "$output/client-references-manifest.json" 2>&1 || true)"
echo "$sp_err" | grep -q "is not an Ssg route" || fail "staticparams op must refuse the non-Ssg /blog/[slug] (precedence), got: $sp_err"
echo "OK (gate 1b): /blog/[slug] has generateStaticParams yet classifies Dynamic (request-read precedence, matching next build's ƒ) — skipped + reason-recorded, staticparams op refuses it"

# --- Gate 2: structural — the dumb server is genuinely dumb ----------------------
serve="$repo/scripts/rsc/next-static-serve.mjs"
# Look at code, not comments: strip //-comments and /* */ blocks before grepping so the
# module's own explanation of the property does not false-trigger the property check.
serve_code="$(sed -E 's://.*$::' "$serve" | sed -E '/\/\*/,/\*\//d')"
echo "$serve_code" | grep -qE 'spawn\(' && fail "next-static-serve.mjs calls spawn() — it must not render per request"
echo "$serve_code" | grep -q 'child_process' && fail "next-static-serve.mjs imports child_process — it must be pure fs"
echo "$serve_code" | grep -qE "(from|import\(|require\()[^;]*(rsc-render|server/server)" && fail "next-static-serve.mjs imports an RSC bundle — it must import neither"
echo "OK (gate 2): next-static-serve.mjs imports neither RSC bundle and spawns no child (structurally dumb)"

# --- Gate 3: dumb serve (byte-equal, raw flight, dynamic 501/404) ---------------
serverlog="$(mktemp)"
node "$serve" "$static" 0 > "$serverlog" 2>&1 &
server_pid=$!
cleanup() {
  kill "$server_pid" 2>/dev/null || true
  agent-browser close 2>/dev/null || true
}
trap cleanup EXIT
for _ in $(seq 1 50); do
  port="$(grep -o 'localhost:[0-9]*' "$serverlog" | head -1 | cut -d: -f2 || true)"
  [ -n "${port:-}" ] && break
  sleep 0.2
done
[ -n "${port:-}" ] || { cat "$serverlog"; fail "static server did not start"; }
base="http://localhost:$port"
echo "static server on $base"

curl -s "$base/" > /tmp/ssg_curl_index.html
cmp -s /tmp/ssg_curl_index.html "$static/index.html" || fail "curl / is not byte-equal to static/index.html"
curl -s "$base/about" > /tmp/ssg_curl_about.html
cmp -s /tmp/ssg_curl_about.html "$static/about.html" || fail "curl /about is not byte-equal to static/about.html"
curl -s "$base/products/a" > /tmp/ssg_curl_prod.html
cmp -s /tmp/ssg_curl_prod.html "$static/products/a.html" || fail "curl /products/a is not byte-equal to static/products/a.html"
sed 's/<!--[^>]*-->//g' /tmp/ssg_curl_prod.html | grep -q "product: a" || fail "curl /products/a missing 'product: a'"
ctype="$(curl -s -o /dev/null -D - "$base/about?__rsc=1" | tr -d '\r' | grep -i '^content-type:' || true)"
echo "$ctype" | grep -qi 'text/x-component' || fail "/about?__rsc=1 content-type is not text/x-component (got: $ctype)"
curl -s "$base/about?__rsc=1" | head -c 20 | grep -qi "<!DOCTYPE" && fail "/about?__rsc=1 returned HTML, not raw flight"
dcode="$(curl -s -o /dev/null -w '%{http_code}' "$base/blog/anything")"
[ "$dcode" = "501" ] || [ "$dcode" = "404" ] || fail "/blog/anything did not 501/404 on the static server (got $dcode) — it must NEVER render"
curl -s "$base/blog/anything" | grep -qi "<!DOCTYPE" && fail "/blog/anything returned an HTML document (a static export must not render a dynamic route)"
rcode="$(curl -s -o /dev/null -w '%{http_code}' "$base/blog/x?__rsc=1")"
[ "$rcode" = "404" ] || fail "/blog/x?__rsc=1 did not 404 (got $rcode) — no prerendered flight for a dynamic route"
acode="$(curl -s -o /dev/null -w '%{http_code} %{content_type}' "$base/client.js")"
echo "$acode" | grep -qE '^200 text/javascript' || fail "/client.js not served (got: $acode)"
echo "OK (gate 3): dumb serve byte-equal HTML; ?__rsc=1 raw flight; dynamic 501/404 (never rendered); assets served"

# --- Gate 4: real browser — hydrate (zero console errors) + interact + soft-nav --
agent-browser open "$base/" >/dev/null 2>&1
agent-browser wait "#inc" >/dev/null 2>&1 || true
agent-browser console --clear >/dev/null 2>&1 || true
agent-browser eval 'location.reload()' >/dev/null 2>&1 || true
for _ in $(seq 1 40); do t="$(agent-browser get text "#counter" 2>/dev/null || true)"; echo "$t" | grep -q "count: 5" && break; sleep 0.2; done

hydrated="$(agent-browser eval 'Object.keys(document).some(k=>k.startsWith("__reactContainer$")) || (!!document.querySelector("#inc") && Object.keys(document.querySelector("#inc")).some(k=>k.startsWith("__reactFiber$")))' 2>/dev/null || true)"
echo "$hydrated" | grep -qi true || fail "the prerendered document did not hydrate (no __reactContainer/__reactFiber)"

initial="$(agent-browser get text "#counter" 2>/dev/null || true)"
echo "$initial" | grep -q "count: 5" || fail "prerendered initial island state is not 'count: 5' (got: $initial)"
agent-browser click "#inc" >/dev/null 2>&1
for _ in $(seq 1 20); do c="$(agent-browser get text "#counter" 2>/dev/null || true)"; echo "$c" | grep -q "count: 6" && break; sleep 0.2; done
c="$(agent-browser get text "#counter" 2>/dev/null || true)"
echo "$c" | grep -q "count: 6" || fail "clicking #inc did not increment (hydration from a PRERENDERED file failed; got: $c)"

console="$(agent-browser console 2>/dev/null || true)"
if echo "$console" | grep -qiE 'error|warning|hydrat|did not match|text content does not match'; then
  echo "$console"; fail "console had errors/warnings after hydrating a prerendered page (must be zero)"
fi
echo "OK (gate 4a): a PRERENDERED page hydrated with ZERO console errors and its island is interactive (count 5 -> 6)"

# Soft-nav: the next/link click fetches the PRERENDERED /about?__rsc=1 and diff-renders.
agent-browser eval 'window.__softnav = "kept"' >/dev/null 2>&1
agent-browser click "#about-link" >/dev/null 2>&1
for _ in $(seq 1 40); do t="$(agent-browser get text "#about" 2>/dev/null || true)"; echo "$t" | grep -q "About page" && break; sleep 0.2; done
t="$(agent-browser get text "#about" 2>/dev/null || true)"
echo "$t" | grep -q "About page" || fail "soft-nav: #about-link did not render the prerendered About content (got: $t)"
probe="$(agent-browser eval '[String(window.__softnav), location.pathname, !!document.querySelector("#app-shell")].join("|")' 2>/dev/null || true)"
echo "$probe" | grep -qF 'kept|/about|true' || fail "soft-nav from a static export: expected sentinel kept + /about + #app-shell preserved (probe: $probe)"
agent-browser eval 'history.back()' >/dev/null 2>&1
for _ in $(seq 1 40); do t="$(agent-browser get text "#heading" 2>/dev/null || true)"; echo "$t" | grep -q "from-server" && break; sleep 0.2; done
back="$(agent-browser eval '[String(window.__softnav), location.pathname].join("|")' 2>/dev/null || true)"
echo "$back" | grep -qF 'kept|/' || fail "soft-nav back: history.back() did not restore / (probe: $back)"
echo "OK (gate 4b): next/link soft-navigated to the PRERENDERED /about (?__rsc=1 diff-render, #app-shell preserved); history.back restored /"

# --- Gate 4c: a PER-generateStaticParams-ENTRY page (/products/a) is served statically + hydrates
# The SSG proof the task pins: generateStaticParams enumerated {id:a},{id:b} at build → one static
# .html per entry (products/a.html, products/b.html). Open products/a on the DUMB static server
# (no render, no child) and confirm the browser HYDRATES it (React attaches to the document; a
# hydration mismatch would surface a console error) with the enumerated param rendered.
agent-browser open "$base/products/a" >/dev/null 2>&1
for _ in $(seq 1 40); do t="$(agent-browser get text "#product" 2>/dev/null || true)"; echo "$t" | grep -q "product: a" && break; sleep 0.2; done
agent-browser console --clear >/dev/null 2>&1 || true
agent-browser eval 'location.reload()' >/dev/null 2>&1 || true
for _ in $(seq 1 40); do t="$(agent-browser get text "#product" 2>/dev/null || true)"; echo "$t" | grep -q "product: a" && break; sleep 0.2; done
pa="$(agent-browser get text "#product" 2>/dev/null || true)"
echo "$pa" | grep -q "product: a" || fail "prerendered per-gsp page /products/a did not render the enumerated param (expected 'product: a', got: $pa)"
phydr="$(agent-browser eval 'Object.keys(document).some(k=>k.startsWith("__reactContainer$")) || Object.keys(document.body).some(k=>k.startsWith("__reactContainer$"))' 2>/dev/null || true)"
echo "$phydr" | grep -qi true || fail "the prerendered per-gsp page /products/a did not hydrate (no __reactContainer\$ on document)"
pconsole="$(agent-browser console 2>/dev/null || true)"
if echo "$pconsole" | grep -qiE 'error|warning|hydrat|did not match|text content does not match'; then
  echo "$pconsole"; fail "console had errors/warnings hydrating the prerendered /products/a (must be zero)"
fi
# The sibling generateStaticParams entry is a DISTINCT static file with its own param.
curl -s "$base/products/b" | sed 's/<!--[^>]*-->//g' | grep -q "product: b" || fail "sibling gsp entry /products/b not served statically with 'product: b'"
echo "OK (gate 4c): a PER-generateStaticParams-ENTRY page (/products/a) served by the dumb static server HYDRATED with the enumerated param + ZERO console errors; sibling /products/b is a distinct static file"

echo "PASS: FULL SSG — build-time static prerender of every static+SSG route to on-disk .html+.rsc, served by a DUMB static file server (zero per-request render), hydrating + soft-navigating in a real browser; dynamic routes honestly skipped"
