#!/usr/bin/env bash
# RSC Slice E gate — the minimal-but-real RSC app, built and served entirely by
# diffpack, verified in a REAL browser via agent-browser.
#
# diffpack natively builds all THREE RSC graphs of the fixture:
#   • client       (Target::Client)      -> public/ browser bundle + the RSC seam +
#                                           Manifest #1 (client-references manifest)
#   • react-server (Target::ReactServer) -> the flight render/action bundle (its
#                                           "use client" island becomes client refs;
#                                           its "use server" action registers)
#   • ssr          (Target::Server)      -> the SSR-of-flight bundle (the island is
#                                           real code; a server-references manifest
#                                           records its own ids for the ssrModuleMapping)
# The emitted Node orchestrator (scripts/rsc/rsc-server.mjs) wires them into an HTTP
# app. This gate boots it and asserts, in a real browser:
#   1. the pre-hydration SSR HTML carries the Server Component text AND the client
#      island's initial state (count: 5) — flight render + SSR-of-flight;
#   2. the client bundle carries NO server-only action code — the boundary holds;
#   3. clicking the island's button increments its useState count — hydration + seam;
#   4. clicking the server button round-trips a "use server" action over /_action/
#      and renders its result — encodeReply -> dispatch -> flight -> createFromFetch.
# Native build (Rust); Node + Chrome are only the oracle. Exit 0 = gate PASS.
set -euo pipefail

repo="$(cd "$(dirname "$0")/../.." && pwd)"
fixture="${1:-$repo/integration/rsc-reference}"
diffpack="$repo/target/release/diffpack"
output="$fixture/.diffpack-output"

fail() { echo "FAIL: $*" >&2; exit 1; }

echo "== building diffpack (release) =="
cargo build --release --manifest-path "$repo/Cargo.toml"

if [ ! -d "$fixture/node_modules/react-server-dom-webpack" ]; then
  echo "== installing pinned RSC deps in $fixture =="
  (cd "$fixture" && npm install --no-audit --no-fund)
fi

echo "== native build: client graph =="
"$diffpack" build-app "$fixture" client --no-minify

echo "== native build: react-server render/action graph =="
"$diffpack" build-app "$fixture" react-server --no-minify
# The react-server and ssr graphs both emit server/server.mjs; snapshot the
# react-server output (and its own references manifest) aside before the ssr build
# overwrites server/. This is orchestration of diffpack's outputs, not a build step.
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
node "$repo/scripts/rsc/rsc-server.mjs" "$output" 0 > "$serverlog" 2>&1 &
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

# --- Gate 1 (pre-hydration SSR HTML via curl) ------------------------------------
html="$(curl -s "$base/")"
echo "$html" | grep -q "Server:from-server" || { echo "$html"; fail "SSR HTML missing the Server Component text"; }
echo "$html" | grep -q "count: 5" || { echo "$html"; fail "SSR HTML missing the client island initial state (count: 5)"; }
echo "$html" | grep -q 'id="root"' || fail "SSR HTML missing the hydration #root container"
echo "OK (gate 1): SSR HTML carries the Server Component text and the island's initial state"

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

echo "PASS: minimal real RSC app — flight render + SSR-of-flight + browser hydration + server-action round-trip, all built and served by diffpack"
