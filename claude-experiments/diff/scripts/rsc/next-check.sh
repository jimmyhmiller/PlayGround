#!/usr/bin/env bash
# RSC Slice F gate — a REAL Next.js app-router app built and served entirely by
# diffpack via the native next app-router adapter, verified in a REAL browser
# (agent-browser).
#
# The fixture integration/next-app-router is a genuine `create-next-app --app`
# project (its authentic default is preserved under authentic-create-next-app/ for
# the gap report). Its app/ is a real app-router app that `next build` accepts
# unchanged: a Server Component root layout (app/layout.tsx) wrapping an async
# Server Component page (app/page.tsx) that renders a `next/link`, embeds a
# `"use client"` island (app/Counter.tsx), and passes a `"use server"` action
# (app/actions.ts) into it.
#
# diffpack's next app-router adapter (src/next_adapter.rs) detects the app-router
# conventions, scaffolds the three RSC entries + minimal `next/*` shims under
# .diffpack-next/, and the proven RSC spine (Slices A–E) builds all THREE graphs
# natively:
#   • client       (Target::Client)      -> browser bundle + RSC seam + Manifest #1
#   • react-server (Target::ReactServer) -> flight render/action bundle
#   • ssr          (Target::Server)      -> SSR-of-flight of the FULL app-router
#                                           document (the RootLayout owns <html>)
# The emitted Node orchestrator (scripts/rsc/next-server.mjs) wires them into an
# HTTP app. This gate boots it and asserts, in a real browser:
#   1. pre-hydration SSR of the app-router document tree — the layout wrapper
#      (#app-shell), the Server Component heading (Server:from-server), the
#      next/link (<a href="/about">), and the island's initial state (count: 5);
#   2. the client bundle carries NO server-only action code — the boundary holds;
#   3. clicking the island's button increments its useState count — hydration + seam;
#   4. clicking the server button round-trips the "use server" action over
#      /_action/ (increment(6) -> 7).
# Native build (Rust); Node + Chrome are only the oracle. Exit 0 = gate PASS.
set -euo pipefail

repo="$(cd "$(dirname "$0")/../.." && pwd)"
fixture="${1:-$repo/integration/next-app-router}"
diffpack="$repo/target/release/diffpack"
output="$fixture/.diffpack-output"

fail() { echo "FAIL: $*" >&2; exit 1; }

echo "== building diffpack (release) =="
cargo build --release --manifest-path "$repo/Cargo.toml"

if [ ! -d "$fixture/node_modules/react-server-dom-webpack" ]; then
  echo "== installing pinned RSC deps in $fixture =="
  (cd "$fixture" && npm install --no-audit --no-fund react-server-dom-webpack@19.2.4)
fi

echo "== native build: client graph =="
"$diffpack" build-app "$fixture" client --no-minify

echo "== native build: react-server render/action graph =="
"$diffpack" build-app "$fixture" react-server --no-minify
# Snapshot the react-server output aside before the ssr build overwrites server/.
rm -rf "$output/rsc-render"
cp -r "$output/server" "$output/rsc-render"

echo "== native build: ssr-of-flight graph =="
"$diffpack" build-app "$fixture" ssr --no-minify

# --- Gate 2 (static): the client bundle carries no server-only action code -------
if grep -rq "actions.ts" "$output/public/"*.js; then
  fail "client bundle references the \"use server\" module actions.ts"
fi
if grep -rq "n + 1" "$output/public/"*.js; then
  fail "client bundle contains the server action body (n + 1)"
fi
echo "OK (gate 2): no server action code in the client bundle"

# --- Boot the app server ---------------------------------------------------------
serverlog="$(mktemp)"
node "$repo/scripts/rsc/next-server.mjs" "$output" 0 > "$serverlog" 2>&1 &
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
[ -n "${port:-}" ] || { cat "$serverlog"; fail "app server did not start"; }
base="http://localhost:$port"
echo "app server on $base"

# --- Gate 1 (pre-hydration SSR document via curl) --------------------------------
html="$(curl -s "$base/")"
echo "$html" | grep -q "<!DOCTYPE html>" || { echo "$html"; fail "SSR did not produce a full document (no doctype; the RootLayout must own <html>)"; }
echo "$html" | grep -q 'id="app-shell"' || { echo "$html"; fail "SSR HTML missing the root layout wrapper (#app-shell) — app-router layout not composed"; }
echo "$html" | grep -q "Server:.*from-server" || { echo "$html"; fail "SSR HTML missing the Server Component text"; }
echo "$html" | grep -q 'href="/about"' || { echo "$html"; fail "SSR HTML missing the next/link (<a href=\"/about\">)"; }
echo "$html" | grep -q "count: .*5" || { echo "$html"; fail "SSR HTML missing the client island initial state (count: 5)"; }
echo "OK (gate 1): SSR renders the app-router document (layout + page + next/link + island state)"

# --- Gate 1b: next/font macro + Metadata API (React-hoisted into <head>) ----------
# The layout uses next/font/google (Geist) and exports metadata, exactly like the
# stock create-next-app template. diffpack rewrites the font macro, injects the
# font CSS, and renders the metadata; React 19 hoists both into <head>.
echo "$html" | grep -q "fonts.googleapis.com/css2?family=Geist" || { echo "$html"; fail "next/font: the Geist @import was not injected/hoisted into <head>"; }
echo "$html" | grep -q "__df_fontvar_geist" || { echo "$html"; fail "next/font: the CSS-variable class (--font-geist) is missing (the macro rewrite/CSS is broken)"; }
echo "$html" | grep -q 'class="__df_fontvar_geist"' || { echo "$html"; fail "next/font: \${geist.variable} did not resolve onto <html>"; }
echo "$html" | grep -q "<title[^>]*>diffpack next app-router</title>" || { echo "$html"; fail "Metadata API: <title> from export const metadata was not rendered"; }
echo "OK (gate 1b): next/font macro rewritten + font CSS hoisted, Metadata API <title> rendered"

# --- Gate 1c: globals.css + CSS Modules injected into <head>, scoping agrees -------
# The layout imports globals.css and the page imports a CSS Module; diffpack compiles
# both in the react-server graph (authoritative for Server-Component class scoping),
# preserves that stylesheet to public/rsc.css, and links it (React-hoisted). The
# module class on the element MUST match a class in the served stylesheet.
echo "$html" | grep -qE '<link[^>]*href="/rsc.css"' || { echo "$html"; fail "CSS: the app stylesheet <link href=/rsc.css> was not injected/hoisted into <head>"; }
module_class="$(echo "$html" | grep -oE 'class="_page_[a-z0-9]+"' | head -1 | grep -oE '_page_[a-z0-9]+')"
[ -n "$module_class" ] || { echo "$html"; fail "CSS Modules: the scoped class from styles.page was not applied to <main>"; }
css="$(curl -s "$base/rsc.css")"
echo "$css" | grep -q "._${module_class#_}" || echo "$css" | grep -q ".$module_class" || { echo "$css"; fail "CSS Modules: served /rsc.css has no rule for the applied class .$module_class (scoping disagrees)"; }
echo "$css" | grep -q "background: rgb(11, 22, 33)" || { echo "$css"; fail "CSS: globals.css (body background) missing from the served stylesheet"; }
echo "OK (gate 1c): globals.css + CSS-Module stylesheet linked into <head>; the applied class .$module_class matches the served CSS"

# --- Gate 1d: multi-route app-router (a SECOND route, under the same root layout) --
# app/about/page.tsx is a distinct route; the adapter discovers it, the react-server
# render matches the request pathname to it, and it renders as a FULL document under
# the root layout (nested-layout composition + per-request matching).
about="$(curl -s "$base/about")"
echo "$about" | grep -q "<!DOCTYPE html>" || { echo "$about"; fail "/about did not render a full document (multi-route matching broken)"; }
echo "$about" | grep -q 'id="app-shell"' || { echo "$about"; fail "/about is not wrapped by the root layout (nested-layout composition broken)"; }
echo "$about" | grep -q "About page (app-router route)" || { echo "$about"; fail "/about did not render the About page's Server Component content"; }
if echo "$about" | grep -q "from-server"; then echo "$about"; fail "/about wrongly rendered the index content (route matching fell back to /)"; fi
echo "OK (gate 1d): a second app-router route (/about) matches per-request and renders under the root layout"

# --- Gate 1e: dynamic segment ([slug]) — params extracted per request --------------
# app/blog/[slug]/page.tsx matches /blog/<slug> with params.slug captured from the URL
# (a Dynamic segment, not a literal). Two distinct slugs prove per-request extraction,
# not a hardcode. (React inserts <!-- --> between adjacent text nodes, so strip
# comments before matching "post: <slug>".)
blog_hello="$(curl -s "$base/blog/hello" | sed 's/<!--[^>]*-->//g')"
echo "$blog_hello" | grep -q "<!DOCTYPE html>" || { echo "$blog_hello"; fail "/blog/hello did not render a full document (dynamic segment matching broken)"; }
echo "$blog_hello" | grep -q 'id="app-shell"' || { echo "$blog_hello"; fail "/blog/hello not wrapped by the root layout"; }
echo "$blog_hello" | grep -q "post: hello" || { echo "$blog_hello"; fail "/blog/hello missing the extracted param (expected post: hello)"; }
blog_world="$(curl -s "$base/blog/world" | sed 's/<!--[^>]*-->//g')"
echo "$blog_world" | grep -q "post: world" || { echo "$blog_world"; fail "/blog/world missing the extracted param (expected post: world; per-request extraction broken)"; }
echo "OK (gate 1e): dynamic [slug] segment matched per-request (post: hello / post: world)"

# --- Gate 1f: a real 404 for an unknown path (no index fall-through) ----------------
code="$(curl -s -o /dev/null -w '%{http_code}' "$base/no/such/path")"
[ "$code" = "404" ] || fail "unknown path did not 404 (got HTTP $code) — matchRoute miss must render the not-found tree with a 404 status"
nf="$(curl -s "$base/no/such/path")"
echo "$nf" | grep -q "404 — page not found" || { echo "$nf"; fail "404 body did not render app/not-found.tsx"; }
if echo "$nf" | grep -q "from-server"; then echo "$nf"; fail "404 wrongly fell through to the index route (found from-server)"; fi
echo "OK (gate 1f): unknown path returns a real HTTP 404 rendering app/not-found.tsx (no index fall-through)"

# --- Gate 1g (http + structural): error boundary contains a throwing Server Component
# app/error-demo/page.tsx throws in a Server Component; the adapter wraps it in the
# generated client ErrorBoundary (paired with a segment Suspense), so the flight
# render completes (child exit 0) and SSR returns HTTP 200 (not a 500). The client
# error.tsx fallback recovers AFTER HYDRATION (asserted in the browser gate below) —
# NOTE: under NODE_ENV=production React sanitizes the Server-Component error message
# and defers the boundary recovery to the client, so the SSR HTML carries the empty
# Suspense placeholder, not the "boom-from-server" text (the dev server, a later
# slice, surfaces the real message). Here we assert the render did not crash.
ecode="$(curl -s -o /dev/null -w '%{http_code}' "$base/error-demo")"
[ "$ecode" = "200" ] || { tail -20 "$serverlog"; fail "/error-demo did not return 200 (the throw was not contained by the error boundary; got HTTP $ecode)"; }
edoc="$(curl -s "$base/error-demo")"
echo "$edoc" | grep -q 'id="app-shell"' || { echo "$edoc"; fail "/error-demo lost the root layout (#app-shell) — the render crashed instead of recovering"; }
rsc_entry="$fixture/.diffpack-next/rsc-entry.tsx"
grep -q "ERROR_BOUNDARY" "$rsc_entry" || fail "structural: generated rsc-entry has no ERROR_BOUNDARY composition"
grep -q "Suspense" "$rsc_entry" || fail "structural: generated rsc-entry has no Suspense composition"
echo "OK (gate 1g http+structural): throwing Server Component contained by the client ErrorBoundary (HTTP 200, #app-shell intact, boundary+Suspense composed)"

# --- Gate 1h: loading.tsx composes a Suspense, non-breaking -------------------------
# app/blog/[slug]/loading.tsx makes the adapter wrap the blog page in <Suspense>. The
# SSR uses onAllReady (waits for all Suspense), so the fallback is never in the final
# static HTML (true fallback-in-HTML is streaming, a later slice) — gate it
# structurally + non-breaking (the page still renders its content).
grep -q "loading: M" "$rsc_entry" || fail "structural: no loading boundary interned for the blog route"
echo "$blog_hello" | grep -q "post: hello" || fail "the loading Suspense broke the blog page render"
echo "OK (gate 1h): loading.tsx composed a Suspense around the blog page (structural + non-breaking)"

# --- Gate 1i: next/image fidelity (Slice J / gap 4.2) -------------------------------
# app/page.tsx renders <Image src="/hero.png" width=1200 height=300 sizes=... priority>
# (a raster with build-emitted responsive variants) and <Image src="/next.svg" ...> (an
# SVG, unoptimized). Assert the RASTER img carries a real srcset of >=2 build-emitted
# variant files, sizes/decoding/fetchpriority; the largest variant is a real 200
# image/png file; a priority preload <link> is hoisted; and the SVG renders raw with NO
# srcset. React 19 emits these attribute names camelCase (srcSet/fetchPriority) and the
# browser normalizes them case-insensitively — exactly as Next itself renders under
# React 19 — so the attribute-name greps here are case-insensitive.
hero_img="$(echo "$html" | grep -oiE '<img[^>]*id="hero"[^>]*>' | head -1)"
[ -n "$hero_img" ] || { echo "$html"; fail "next/image: the raster hero <img id=hero> was not rendered"; }
cand="$(echo "$hero_img" | grep -oiE '/_diffpack-image/[^ ]+\.png [0-9]+w' | wc -l | tr -d ' ')"
[ "${cand:-0}" -ge 2 ] || { echo "$hero_img"; fail "next/image: hero srcset has <2 build-emitted variant candidates (got: ${cand:-0})"; }
echo "$hero_img" | grep -qiE 'srcset="[^"]*/_diffpack-image/' || { echo "$hero_img"; fail "next/image: hero <img> has no srcset pointing at /_diffpack-image/ variants"; }
echo "$hero_img" | grep -qF 'sizes="(max-width: 600px) 100vw, 600px"' || { echo "$hero_img"; fail "next/image: hero <img> lost the sizes passthrough"; }
echo "$hero_img" | grep -qiE 'decoding="async"' || { echo "$hero_img"; fail "next/image: hero <img> missing decoding=async"; }
echo "$hero_img" | grep -qiE 'fetchpriority="high"' || { echo "$hero_img"; fail "next/image: priority image missing fetchpriority=high"; }
# The largest variant URL is a REAL emitted static file (200 image/png).
largest="$(echo "$hero_img" | grep -oiE 'src="/_diffpack-image/[^"]+\.png"' | head -1 | sed -E 's/^src="//i; s/"$//')"
[ -n "$largest" ] || { echo "$hero_img"; fail "next/image: hero <img> src is not a /_diffpack-image variant"; }
vhdr="$(curl -s -o /dev/null -w '%{http_code} %{content_type}' "$base$largest")"
echo "$vhdr" | grep -qE '^200 image/png' || fail "next/image: the largest variant $largest is not a real 200 image/png (got: $vhdr)"
# The priority preload <link rel=preload as=image> is hoisted into <head>.
echo "$html" | grep -qiE '<link[^>]*rel="preload"[^>]*as="image"' || { echo "$html"; fail "next/image: no priority preload <link rel=preload as=image> was hoisted"; }
echo "$html" | grep -oiE '<link[^>]*rel="preload"[^>]*as="image"[^>]*>' | grep -qiE 'imagesrcset="[^"]*/_diffpack-image/|href="/_diffpack-image/' || { echo "$html"; fail "next/image: the preload link does not reference the hero variants (imagesrcset/href)"; }
# The SVG image is unoptimized: raw src, NO srcset.
logo_img="$(echo "$html" | grep -oiE '<img[^>]*id="logo"[^>]*>' | head -1)"
[ -n "$logo_img" ] || { echo "$html"; fail "next/image: the SVG logo <img id=logo> was not rendered"; }
echo "$logo_img" | grep -qF 'src="/next.svg"' || { echo "$logo_img"; fail "next/image: SVG logo lost its raw src=/next.svg"; }
if echo "$logo_img" | grep -qiE 'srcset='; then echo "$logo_img"; fail "next/image: SVG logo must NOT have a srcset (unoptimized, byte-faithful to Next)"; fi
echo "$logo_img" | grep -qiE 'decoding="async"' || { echo "$logo_img"; fail "next/image: SVG logo missing decoding=async"; }
echo "OK (gate 1i): next/image — raster srcset (${cand} variants) + priority preload hoisted + real 200 image/png variants; SVG raw src, no srcset"

# --- Gate 5a: raw per-route flight over ?__rsc=1 (soft-nav transport) --------------
# The client Router fetches the target route's RAW flight via ?__rsc=1 to diff-render
# it without a full document load. Assert that endpoint returns raw flight rows (NOT
# an HTML document), with the RSC content-type, and that the SAME path WITHOUT the
# query still returns the full HTML document (no gate-1d regression).
flight="$(curl -s "$base/about?__rsc=1")"
[ -n "$flight" ] || fail "?__rsc=1 returned an empty body (no flight)"
if echo "$flight" | grep -qi "<!DOCTYPE html>"; then echo "$flight"; fail "?__rsc=1 returned an HTML document, not raw flight"; fi
# Header dump from a real GET (the orchestrator serves GET, not HEAD).
ctype="$(curl -s -o /dev/null -D - "$base/about?__rsc=1" | tr -d '\r' | grep -i '^content-type:' || true)"
echo "$ctype" | grep -qi 'text/x-component' || fail "?__rsc=1 content-type is not text/x-component (got: $ctype)"
about_full="$(curl -s "$base/about")"
echo "$about_full" | grep -q "<!DOCTYPE html>" || { echo "$about_full"; fail "/about WITHOUT ?__rsc=1 no longer returns the full HTML document (soft-nav endpoint shadowed the page)"; }
echo "OK (gate 5a): ?__rsc=1 serves the raw route flight (text/x-component); plain /about still full HTML"

# --- Gate 6: server-side redirect() -> a REAL HTTP 307 -----------------------------
# app/go/page.tsx is a Server Component that calls redirect('/about'). On the server
# that throws Next's NEXT_REDIRECT digest; the react-server render captures it via
# onError and reports it on the fd-3 control channel; the orchestrator issues a real
# 307 to /about (it never SSRs the errored tree). Following the redirect lands on the
# /about document.
redir="$(curl -s -o /dev/null -w '%{http_code} %header{location}' "$base/go")"
echo "$redir" | grep -q "^307 /about" || fail "redirect(): /go did not return 307 -> /about (got: $redir)"
go_followed="$(curl -sL "$base/go" | sed 's/<!--[^>]*-->//g')"
echo "$go_followed" | grep -q "About page (app-router route)" || { echo "$go_followed"; fail "redirect(): following /go did not land on the /about document"; }
echo "OK (gate 6): server-side redirect('/about') issued a real HTTP 307 to /about (followed to the About document)"

# --- Gate 7: cookies() reads the real request cookie -------------------------------
# app/blog/[slug]/page.tsx (an async Server Component) does `const c = await cookies();
# c.get('theme')?.value`. The per-request AsyncLocalStorage the render establishes must
# carry the request's Cookie header into that call. With --cookie 'theme=dark' the HTML
# contains `theme: dark`; without the cookie it is `theme: none` (proving it reads the
# ACTUAL request, not a constant).
cookie_html="$(curl -s --cookie 'theme=dark' "$base/blog/hello" | sed 's/<!--[^>]*-->//g')"
echo "$cookie_html" | grep -q "theme: dark" || { echo "$cookie_html"; fail "cookies(): request cookie 'theme=dark' did not reach await cookies() (expected 'theme: dark')"; }
nocookie_html="$(curl -s "$base/blog/hello" | sed 's/<!--[^>]*-->//g')"
echo "$nocookie_html" | grep -q "theme: none" || { echo "$nocookie_html"; fail "cookies(): with no cookie the value should be 'theme: none' (a real read, not a constant)"; }
echo "OK (gate 7): await cookies() read the real request cookie (theme=dark -> 'theme: dark'; none -> 'theme: none')"

# --- Gate 8 (SSR): useParams() resolves the matched segment on the SERVER -----------
# The `"use client"` island app/blog/[slug]/SlugBadge.tsx renders `slug: {useParams().slug}`.
# useParams() reads the app-router PathParamsContext, which the SSR entry feeds from the
# matched request params — so the SSR HTML already carries `slug: hello` (the browser
# half + zero-console-error hydration is asserted in the browser gate below).
echo "$nocookie_html" | grep -q "slug: hello" || { echo "$nocookie_html"; fail "useParams() SSR: the client island did not render 'slug: hello' (PathParamsContext not fed on SSR)"; }
slug_world="$(curl -s "$base/blog/world" | sed 's/<!--[^>]*-->//g')"
echo "$slug_world" | grep -q "slug: world" || { echo "$slug_world"; fail "useParams() SSR: /blog/world did not render 'slug: world' (per-request context broken)"; }
echo "OK (gate 8 SSR): useParams() resolved the matched segment on the server (slug: hello / slug: world)"

# --- Real browser: hydration + interactivity + action round-trip -----------------
agent-browser open "$base/" >/dev/null 2>&1
agent-browser wait "#server-inc" >/dev/null 2>&1 || true

read_count() { agent-browser get text "#counter" 2>/dev/null; }
read_result() { agent-browser get text "#server-result" 2>/dev/null; }

initial="$(read_count)"
echo "$initial" | grep -q "count: 5" || fail "browser initial count is not 5 (got: $initial)"

# Gate 3: hydration made the island interactive — local useState increments.
agent-browser click "#inc" >/dev/null 2>&1
for _ in $(seq 1 20); do c="$(read_count)"; echo "$c" | grep -q "count: 6" && break; sleep 0.2; done
c="$(read_count)"
echo "$c" | grep -q "count: 6" || fail "clicking #inc did not increment the count (hydration failed; got: $c)"
echo "OK (gate 3): the client island hydrated and is interactive (count 5 -> 6 on click)"

# Gate 4: the server action round-trips — increment(6) -> 7 over /_action/.
agent-browser click "#server-inc" >/dev/null 2>&1
for _ in $(seq 1 40); do r="$(read_result)"; echo "$r" | grep -q "server: 7" && break; sleep 0.25; done
r="$(read_result)"
echo "$r" | grep -q "server: 7" || { tail -20 "$serverlog"; fail "server action did not round-trip (expected server: 7, got: $r)"; }
echo "OK (gate 4): the \"use server\" action round-tripped over /_action/ (increment(6) -> 7)"

# The local state must survive the action (proves the action did not remount).
c="$(read_count)"
echo "$c" | grep -q "count: 6" || fail "local state was lost across the action call (got: $c)"

# --- Gate 5b: client-side SOFT navigation (no full document reload) ---------------
# Clicking the next/link to /about must fetch that route's flight (?__rsc=1) and
# diff-render it into the LIVE tree — no full document load. A page-scoped window
# sentinel survives iff there was no reload; the root layout (#app-shell) surviving
# proves a diff-render (not a fresh document). Then history.back() soft-navigates
# home the same way.
agent-browser open "$base/" >/dev/null 2>&1
agent-browser wait "#about-link" >/dev/null 2>&1 || fail "soft-nav: #about-link not present on / after hydration"
agent-browser eval 'window.__softnav = "kept"' >/dev/null 2>&1
agent-browser click "#about-link" >/dev/null 2>&1
for _ in $(seq 1 40); do t="$(agent-browser get text "#about" 2>/dev/null || true)"; echo "$t" | grep -q "About page" && break; sleep 0.2; done
t="$(agent-browser get text "#about" 2>/dev/null || true)"
echo "$t" | grep -q "About page" || fail "soft-nav: clicking #about-link did not render the About content (got: $t)"
# A `|`-joined probe (a lost sentinel becomes empty on reload); grep -F avoids the
# JSON-string double-escaping agent-browser applies to string eval results.
probe="$(agent-browser eval '[String(window.__softnav), location.pathname, !!document.querySelector("#app-shell")].join("|")' 2>/dev/null || true)"
echo "$probe" | grep -qF 'kept|/about|true' || fail "soft-nav: expected sentinel kept + path /about + #app-shell preserved (a FULL RELOAD or missing diff-render otherwise) — probe: $probe"
echo "OK (gate 5b-forward): #about-link soft-navigated to /about (sentinel + #app-shell survived, history updated), no reload"

# Backward: history.back() must soft-navigate home (fetch / flight, diff-render).
agent-browser eval 'history.back()' >/dev/null 2>&1
for _ in $(seq 1 40); do t="$(agent-browser get text "#heading" 2>/dev/null || true)"; echo "$t" | grep -q "Server:.*from-server" && break; sleep 0.2; done
t="$(agent-browser get text "#heading" 2>/dev/null || true)"
echo "$t" | grep -q "Server:.*from-server" || fail "soft-nav back: history.back() did not restore the index heading (got: $t)"
probe="$(agent-browser eval '[String(window.__softnav), location.pathname].join("|")' 2>/dev/null || true)"
echo "$probe" | grep -qF 'kept|/' || fail "soft-nav back: expected sentinel kept + path / after history.back() (a full reload otherwise) — probe: $probe"
echo "OK (gate 5b-back): history.back() soft-navigated home (sentinel survived, index Server Component re-rendered)"

# --- Gate 1g (browser): the client error.tsx fallback recovers after hydration ------
# The throwing Server Component's error is contained by the client ErrorBoundary; on
# the client, after hydration, the boundary catches and renders error.tsx's fallback
# (#error-demo with an "error caught:" message + a #reset button — proving the boundary
# actually rendered the user's fallback, not that the page merely didn't crash). The
# thrown message is prod-sanitized (see the http gate note), so we assert the fallback
# rendered + is interactive, not the literal "boom-from-server".
agent-browser open "$base/error-demo" >/dev/null 2>&1
for _ in $(seq 1 40); do t="$(agent-browser get text "#error-demo" 2>/dev/null || true)"; echo "$t" | grep -q "error caught:" && break; sleep 0.2; done
t="$(agent-browser get text "#error-demo" 2>/dev/null || true)"
echo "$t" | grep -q "error caught:" || fail "the error boundary did not recover on the client (expected 'error caught:' in #error-demo; got: $t)"
agent-browser eval '!!document.querySelector("#reset")' 2>/dev/null | grep -q true || fail "error.tsx fallback is missing the #reset button (the boundary did not render the fallback)"
echo "OK (gate 1g browser): the client error.tsx fallback recovered after hydration (#error-demo + #reset)"

# --- Gate 8 (browser): useParams() resolves on the client with ZERO hydration errors -
# Open /blog/hello in a real browser. The SlugBadge island's useParams().slug must read
# `hello` after hydration (from the PathParamsContext fed by the injected globals — the
# SAME value the SSR entry rendered), and hydration must be clean: no hydration-mismatch
# warning/error in the console (proving the client hooks read a React context fed
# identically on both sides, not a window global that diverges).
agent-browser open "$base/blog/hello" >/dev/null 2>&1
agent-browser console --clear >/dev/null 2>&1 || true
agent-browser eval 'location.reload()' >/dev/null 2>&1 || true
for _ in $(seq 1 40); do t="$(agent-browser get text "#slug-badge" 2>/dev/null || true)"; echo "$t" | grep -q "slug: hello" && break; sleep 0.2; done
t="$(agent-browser get text "#slug-badge" 2>/dev/null || true)"
echo "$t" | grep -q "slug: hello" || fail "useParams() client: #slug-badge did not read 'slug: hello' after hydration (got: $t)"
uparam="$(agent-browser eval 'document.querySelector("#slug-badge").textContent' 2>/dev/null || true)"
echo "$uparam" | grep -qF "slug: hello" || fail "useParams() client: expected 'slug: hello' from the hydrated island (got: $uparam)"
console="$(agent-browser console 2>/dev/null || true)"
if echo "$console" | grep -qiE 'hydrat|did not match|text content does not match'; then
  echo "$console"; fail "useParams() client: hydration mismatch in the console (the client hooks must read the SAME context values as SSR)"
fi
echo "OK (gate 8 browser): useParams() resolved 'slug: hello' on the client after clean hydration (no mismatch)"

echo "PASS: a REAL Next.js app-router app — app-router layout+page composed, flight render + SSR-of-full-document + browser hydration + server-action round-trip, all built and served by diffpack via the native next app-router adapter"
