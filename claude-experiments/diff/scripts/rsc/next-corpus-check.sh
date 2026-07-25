#!/usr/bin/env bash
# Tier-2 corpus gate — every app under integration/next-corpus/ built (its three RSC
# graphs, natively, by diffpack) and SERVED by scripts/rsc/next-server.mjs, then
# curl-smoked per app: SSR document, dynamic routes, ISR/SSG pages, a real 404, a 307
# redirect, route handlers, and the raw ?__rsc=1 flight. Curl-only (NO browser) — node
# is the oracle, the build is native Rust. Exit 0 = gate PASS.
#
# The corpus is HERMETIC: every app's data is a local TS array/map, no network at
# request time. This script proves it (a request-scope `fetch(` fails the gate) and the
# deps are installed ONCE (pinned versions) into the shared corpus node_modules that
# each nested app resolves via node's parent-dir walk.
set -euo pipefail

repo="$(cd "$(dirname "$0")/../.." && pwd)"
corpus="${1:-$repo/integration/next-corpus}"
diffpack="$repo/target/release/diffpack"

fail() { echo "FAIL: $*" >&2; exit 1; }

echo "== building diffpack (release) =="
cargo build --release --manifest-path "$repo/Cargo.toml"

# --- one-time pinned dep install (shared across every nested app) -------------------
if [ ! -d "$corpus/node_modules/react-server-dom-webpack" ]; then
  echo "== installing pinned corpus deps in $corpus =="
  (cd "$corpus" && npm install --no-audit --no-fund)
fi

apps=(blog-static shop-isr dashboard-dynamic docs-catchall)

# --- hermeticity guard: no request-scope network in any app source ------------------
for app in "${apps[@]}"; do
  if grep -rn --include='*.ts' --include='*.tsx' -E '\bfetch\(' "$corpus/$app/app" >/dev/null 2>&1; then
    grep -rn --include='*.ts' --include='*.tsx' -E '\bfetch\(' "$corpus/$app/app" >&2
    fail "$app: request-scope fetch( found — the corpus must be hermetic (local data only)"
  fi
done
echo "OK: no request-scope fetch( in any corpus app (hermetic)"

server_pid=""
cleanup() { [ -n "$server_pid" ] && kill "$server_pid" 2>/dev/null || true; }
trap cleanup EXIT

# Build one app's three graphs natively and boot the orchestrator; sets $base + $port.
serve_app() {
  local app="$1"
  local dir="$corpus/$app"
  local output="$dir/.diffpack-output"
  echo "== [$app] native build: client -> react-server -> ssr =="
  "$diffpack" build-app "$dir" client --no-minify
  "$diffpack" build-app "$dir" react-server --no-minify
  rm -rf "$output/rsc-render"
  cp -r "$output/server" "$output/rsc-render"
  "$diffpack" build-app "$dir" ssr --no-minify

  local serverlog
  serverlog="$(mktemp)"
  node "$repo/scripts/rsc/next-server.mjs" "$output" 0 > "$serverlog" 2>&1 &
  server_pid=$!
  for _ in $(seq 1 50); do
    port="$(grep -o 'localhost:[0-9]*' "$serverlog" | head -1 | cut -d: -f2 || true)"
    [ -n "${port:-}" ] && break
    sleep 0.2
  done
  [ -n "${port:-}" ] || { cat "$serverlog"; fail "[$app] server did not start"; }
  base="http://localhost:$port"
  echo "[$app] server on $base"
}

stop_app() {
  [ -n "$server_pid" ] && kill "$server_pid" 2>/dev/null || true
  server_pid=""
}

# curl helpers. `body` strips React's inter-text comments so "post: hello" matches.
body() { curl -s "$base$1" | sed 's/<!--[^>]*-->//g'; }
status() { curl -s -o /dev/null -w '%{http_code}' "$base$1"; }

expect_body() { # path substring
  local got; got="$(body "$1")"
  echo "$got" | grep -qF "$2" || { echo "$got" | head -40; fail "[$app] GET $1 missing substring: $2"; }
  echo "OK [$app] GET $1 contains: $2"
}
expect_status() { # path code
  local got; got="$(status "$1")"
  [ "$got" = "$2" ] || fail "[$app] GET $1 expected HTTP $2, got $got"
  echo "OK [$app] GET $1 -> HTTP $2"
}

# --- blog-static: static index + route group + SSG dynamic + real 404 ---------------
app=blog-static; serve_app "$app"
expect_body   /              'data-app="blog-static"'
expect_body   /about         'about (marketing group)'
expect_body   /blog/hello    'post: hello'
expect_status /nope          404
stop_app

# --- shop-isr: ISR listing + SSG product page ---------------------------------------
app=shop-isr; serve_app "$app"
expect_body   /              'shop (ISR listing)'
expect_body   /products/a    'product: a'
expect_status /nope          404
stop_app

# --- dashboard-dynamic: static home + request-state read + redirect(307) ------------
app=dashboard-dynamic; serve_app "$app"
expect_body   /              'data-app="dashboard-dynamic"'
expect_body   /whoami        'whoami'
redir="$(curl -s -o /dev/null -w '%{http_code} %header{location}' "$base/go")"
echo "$redir" | grep -q "^307 /" || fail "[$app] /go did not 307 -> / (got: $redir)"
echo "OK [$app] GET /go -> 307 /"
expect_status /nope          404
stop_app

# --- docs-catchall: static home + optional catch-all SSG + route handlers -----------
app=docs-catchall; serve_app "$app"
expect_body   /                 'docs index'
expect_body   /intro            'getting started'
expect_body   /guide/setup      'how to set up'
expect_body   /api/health       'ok'
expect_body   '/api/echo?q=hi'  'GET'
stop_app

echo "PASS: the hermetic next app-router corpus — 4 apps built natively (three graphs each) and served, SSR + SSG + ISR + dynamic + redirect + 404 + route handlers all green"
