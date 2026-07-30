#!/usr/bin/env bash
# Slice K gate — `diffpack dev` for the Next.js app-router app.
#
# `diffpack dev integration/next-app-router` boots the Next dev topology (the SAME
# three RSC graphs the production build uses — client / react-server / ssr — kept
# alive per-environment, served by the embedded next orchestrator, with the diffpack
# reverse proxy in front injecting the Fast Refresh + WebSocket HMR preamble). This
# gate proves BOTH dev edit classes in a REAL browser (agent-browser):
#
#   1. Boot + hydrate: `/` server-renders the app-router document, the diffpack
#      WebSocket HMR + Fast Refresh preamble is injected, and the `"use client"`
#      island hydrates (clicking #inc moves its useState count 5 -> 6).
#   2. Island edit = STATE-PRESERVING Fast Refresh: editing app/Counter.tsx's label
#      (`count: ` -> `tally: `) pushes a WS `update` (no reload), and the SAME live
#      node now reads `tally: 6` — the hook state (6) survived, proving Fast Refresh
#      through the flight-resolved client reference (NOT a remount/reload).
#   3. Server-component edit = correct reload: editing app/page.tsx's `from-server`
#      string makes the new text appear in the browser AND in a fresh `curl /` — the
#      orchestrator spawns a fresh react-server child per GET, so the reload shows the
#      newly server-rendered content.
#
# Native build (Rust); Node + Chrome are only the oracle. The fixture files are always
# restored (a trap). Exit 0 = gate PASS.
# Strict mode, the ERR net (no abort is ever silent) and fail() — see _gate-prelude.sh.
source "$(dirname "$0")/_gate-prelude.sh"

repo="$(cd "$(dirname "$0")/../.." && pwd)"
fixture="$repo/integration/next-app-router"
diffpack="$repo/target/release/diffpack"
port="${DIFFPACK_NEXT_DEV_PORT:-8968}"
base="http://127.0.0.1:$port"

counter="$fixture/app/Counter.tsx"
page="$fixture/app/page.tsx"
stamp="$(date +%s)"

echo "== building diffpack (release) =="
cargo build --release --manifest-path "$repo/Cargo.toml"

# RSC deps + the React Fast Refresh runtime (@vitejs/plugin-react ships the runtime;
# react-refresh is its core). node_modules is gitignored, so install on demand.
if [ ! -d "$fixture/node_modules/react-server-dom-webpack" ]; then
  echo "== installing pinned RSC deps in $fixture =="
  (cd "$fixture" && npm install --no-audit --no-fund react-server-dom-webpack@19.2.4)
fi
if [ ! -f "$fixture/node_modules/@vitejs/plugin-react/dist/refresh-runtime.js" ] \
   && [ ! -f "$fixture/node_modules/react-refresh/cjs/react-refresh-runtime.development.js" ]; then
  echo "== installing the React Fast Refresh runtime in $fixture =="
  (cd "$fixture" && npm install --no-audit --no-fund --save-dev @vitejs/plugin-react react-refresh)
fi

# Snapshot the fixture files we edit; ALWAYS restore them + reap the dev server.
cp "$counter" "/tmp/next-dev-counter.$stamp.bak"
cp "$page" "/tmp/next-dev-page.$stamp.bak"
devpid=""
cleanup() {
  [ -n "$devpid" ] && kill "$devpid" 2>/dev/null || true
  agent-browser close 2>/dev/null || true
  cp "/tmp/next-dev-counter.$stamp.bak" "$counter" 2>/dev/null || true
  cp "/tmp/next-dev-page.$stamp.bak" "$page" 2>/dev/null || true
}
trap cleanup EXIT

grep -q "count: " "$counter" || fail "app/Counter.tsx no longer contains 'count: ' — refusing to edit"
grep -q "from-server" "$page" || fail "app/page.tsx no longer contains 'from-server' — refusing to edit"

echo "== booting: diffpack dev $fixture $port =="
rm -rf "$fixture/.diffpack-output"
devlog="$(mktemp)"
"$diffpack" dev "$fixture" "$port" > "$devlog" 2>&1 &
devpid=$!

for _ in $(seq 1 120); do
  curl -s -o /dev/null "$base/" && break
  # Bail early if the dev server died.
  kill -0 "$devpid" 2>/dev/null || { cat "$devlog"; fail "dev server exited during startup"; }
  sleep 1
done
curl -s -o /dev/null "$base/" || { cat "$devlog"; fail "dev server did not come up on $base"; }
echo "dev server up on $base"

# --- Gate D1: SSR document + HMR preamble injected -------------------------------
html="$(curl -s "$base/")"
echo "$html" | grep -q "<!DOCTYPE html>" || { echo "$html" | head -5; fail "dev / did not serve the app-router document"; }
echo "$html" | grep -q 'id="app-shell"' || fail "dev / missing the root layout (#app-shell)"
echo "$html" | grep -q "/__diffpack_hmr/ws" || fail "dev / missing the WebSocket HMR client (Fast Refresh preamble not injected)"
echo "$html" | grep -q '\$RefreshRuntime\$' || fail "dev / missing the React Fast Refresh runtime preamble"
echo "OK (gate D1): dev server SSRs the app-router document with the WS HMR + Fast Refresh preamble injected"

# --- Gate D1b: the stylesheet the document links is actually SERVED --------------
# REGRESSION. The react-server graph compiles the app's CSS and preserves it to
# `public/rsc.css`; the client graph is emitted afterwards into the SAME `public/`
# and its prune deletes everything the CLIENT graph did not write — which took the
# sheet with it. Nothing downstream noticed: the <link> is guarded on the artifact
# beside the render bundle (untouched by that prune), so the document went on linking
# /rsc.css while GET /rsc.css returned the 404 HTML shell. Served with
# `X-Content-Type-Options: nosniff`, the browser rejected the HTML outright and the
# page rendered fully unstyled — on cal.com and on this app, from a cold dev boot.
#
# The production gate (next-check.sh, gate 1c) could never catch it: `build-app`
# builds the client and react-server graphs in SEPARATE processes, so the preserve
# lands after the client's prune and the ordering hazard does not exist there. This
# asserts STATUS and CONTENT-TYPE as well as bytes, because the failure served a
# perfectly readable 200-shaped HTML body on the CSS URL.
echo "$html" | grep -qE '<link[^>]*href="/rsc.css"' || { echo "$html"; fail "dev: the app stylesheet <link href=/rsc.css> was not linked into the document"; }
css_head="$(curl -s -o /tmp/diffpack-dev-rsc.css -w '%{http_code} %{content_type}' "$base/rsc.css")"
case "$css_head" in
  200\ text/css*) ;;
  *) echo "$css_head"; head -5 /tmp/diffpack-dev-rsc.css; fail "dev: the document links /rsc.css but GET /rsc.css returned '$css_head' (expected '200 text/css'); the page renders unstyled";;
esac
grep -q "background: rgb(11, 22, 33)" /tmp/diffpack-dev-rsc.css || { head -5 /tmp/diffpack-dev-rsc.css; fail "dev: served /rsc.css does not carry globals.css (body background)"; }
module_class="$(echo "$html" | grep -oE -m1 '_page_[a-z0-9]+' || true)"
[ -n "$module_class" ] || { echo "$html"; fail "dev: the scoped CSS-Module class was not applied to <main>"; }
grep -q "._${module_class#_}" /tmp/diffpack-dev-rsc.css || grep -q ".$module_class" /tmp/diffpack-dev-rsc.css || { head -5 /tmp/diffpack-dev-rsc.css; fail "dev: served /rsc.css has no rule for the applied class .$module_class (scoping disagrees)"; }
rm -f /tmp/diffpack-dev-rsc.css
echo "OK (gate D1b): the linked /rsc.css is served as text/css and carries globals.css + the CSS-Module rule for .$module_class"

# --- Gate D1c: useId survives the SSR -> hydration seam ---------------------------
# Runs BEFORE D2 deliberately: it reloads the page to capture a clean console, which
# resets the island's useState, and D3 asserts Fast Refresh PRESERVES the 6 that D2
# clicks in. Ordered after D2 it put the counter back to 5 and failed D3 with
# "expected 'tally: 6' ... got 'tally: 5'" — a real gate reporting a bug that was not
# there.
#
# REGRESSION. React derives `useId` from the tree-id FORK a multi-child parent pushes,
# not from the rendered markup. The SSR entry wrapped the flight root as a SINGLE child
# of the hooks providers, while the client's Router wrapped it in a Fragment holding TWO
# children — the tree and the intercept-modal slot, which is a `null` SLOT (not an absent
# child) whenever no modal is open, i.e. always at hydration. One extra fork on the client
# shifted every useId beneath it, so the two sides generated different ids from identical
# markup. On cal.com that surfaced as base-ui rendering `base-ui-_R_2lmdbpi_` on the server
# and `base-ui-_R_amplf69_` on the client: React reports it as a mismatch it "won't patch
# up", and every <label for> is left pointing at an id no input carries.
#
# The assertion is on REACT'S OWN VERDICT, not on the DOM, and that distinction is the
# whole gate. React reports this mismatch and then leaves the server's attribute in place
# ("This won't be patched up"), so the hydrated DOM still reads back the SERVER's id and
# an SSR-vs-DOM attribute comparison passes while the bug is live — measured, not assumed:
# that comparison was tried here first and passed against a deliberately reverted build.
# Only the console distinguishes the two.
#
# `data-uid` is still asserted present, because the console is only meaningful if the
# probe actually rendered — a useId that never ran cannot mismatch, and a silently
# dropped probe would turn this gate green forever.
ssr_uid="$(curl -s "$base/" | grep -oE -m1 'data-uid="[^"]*"' | sed 's/data-uid="//;s/"//')"
[ -n "$ssr_uid" ] || { curl -s "$base/" | head -20; fail "dev: the island did not server-render its useId probe (data-uid); the seam probe is gone and this gate can no longer detect anything"; }
agent-browser console --clear >/dev/null 2>&1
agent-browser open "$base/" >/dev/null 2>&1
agent-browser wait "#island" >/dev/null 2>&1 || fail "dev: #island never appeared, so hydration never ran"
for _ in $(seq 1 20); do [ -n "$(agent-browser eval "document.getElementById('inc') ? '1' : ''" 2>/dev/null | tr -d '\"')" ] && break; sleep 0.3; done
sleep 1
mismatch="$(agent-browser console 2>/dev/null | grep -cE "didn't match|hydrated but" || true)"
[ "$mismatch" = "0" ] || { agent-browser console 2>/dev/null | grep -A20 -E "hydrated but" | head -30; fail "dev: React reported a hydration mismatch — the SSR and client entries disagree on the tree shape above the flight root, so every useId under the tree differs across the seam"; }
echo "OK (gate D1c): useId ($ssr_uid) crossed the SSR -> hydration seam with no React hydration mismatch"

# --- Gate D2: island hydrated + interactive (5 -> 6) -----------------------------
agent-browser open "$base/" >/dev/null 2>&1
agent-browser wait "#inc" >/dev/null 2>&1 || fail "dev: #inc island button not present"
init="$(agent-browser get text '#counter' 2>/dev/null || true)"
echo "$init" | grep -q "count: 5" || fail "dev: island initial state is not 'count: 5' (got: $init)"
agent-browser click "#inc" >/dev/null 2>&1
for _ in $(seq 1 30); do c="$(agent-browser get text '#counter' 2>/dev/null || true)"; echo "$c" | grep -q "count: 6" && break; sleep 0.3; done
c="$(agent-browser get text '#counter' 2>/dev/null || true)"
echo "$c" | grep -q "count: 6" || fail "dev: clicking #inc did not increment (hydration failed; got: $c)"
echo "OK (gate D2): the client island hydrated and is interactive (count 5 -> 6)"

# --- Gate D3: island edit = state-preserving Fast Refresh (no reload) -------------
# Tag the live document; a lost tag proves a full reload happened.
agent-browser eval "window.__nextdev='kept-$stamp'" >/dev/null 2>&1
before_len="$(wc -c < "$devlog")"
# Edit the island's rendered label. State (count=6) must survive; the label must swap.
sed -i.tmp 's/count: /tally: /' "$counter" && rm -f "$counter.tmp"
echo "edited app/Counter.tsx (count: -> tally:); waiting for the Fast Refresh update..."
for _ in $(seq 1 60); do t="$(agent-browser get text '#counter' 2>/dev/null || true)"; echo "$t" | grep -q "tally:" && break; sleep 0.3; done
t="$(agent-browser get text '#counter' 2>/dev/null || true)"
echo "$t" | grep -q "tally: 6" || { tail -20 "$devlog"; fail "island Fast Refresh failed: expected 'tally: 6' (label swapped, state preserved), got: $t"; }
kept="$(agent-browser eval 'String(window.__nextdev)' 2>/dev/null || true)"
echo "$kept" | grep -qF "kept-$stamp" || fail "island edit caused a FULL RELOAD (page sentinel lost) — must be a state-preserving hot update, not a reload"
# The dev loop must have pushed a client HMR update, not a reload.
tail -c "+$((before_len + 1))" "$devlog" | grep -q "hmr update" || { tail -20 "$devlog"; fail "the dev loop did not push a client HMR update for the island edit"; }
echo "OK (gate D3): island edit hot-swapped the label on the SAME live node with state preserved (tally: 6, no reload)"

# --- Gate D4: server-component edit = correct reload (new server-rendered text) ---
marker="from-server-dev-$stamp"
sed -i.tmp "s/from-server/$marker/" "$page" && rm -f "$page.tmp"
echo "edited app/page.tsx (from-server -> $marker); waiting for the reload..."
for _ in $(seq 1 60); do t="$(agent-browser get text '#heading' 2>/dev/null || true)"; echo "$t" | grep -q "$marker" && break; sleep 0.3; done
t="$(agent-browser get text '#heading' 2>/dev/null || true)"
echo "$t" | grep -q "$marker" || { tail -20 "$devlog"; fail "server-component edit did not reach the browser (expected '$marker' in #heading, got: $t)"; }
# A fresh request must also carry the new text (a fresh react-server child rendered it).
fresh="$(curl -s "$base/" | sed 's/<!--[^>]*-->//g')"
echo "$fresh" | grep -q "Server:$marker" || { echo "$fresh" | grep -o 'Server:[A-Za-z0-9-]*' | head; fail "server-component edit not server-rendered on a fresh request (expected Server:$marker)"; }
echo "OK (gate D4): server-component edit re-rendered server-side (new text in the browser AND a fresh curl)"

echo "PASS: diffpack dev for the Next app-router app — SSR + HMR preamble, state-preserving island Fast Refresh, and correct server-component reload, all built natively by diffpack"
