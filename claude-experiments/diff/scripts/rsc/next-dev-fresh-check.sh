#!/usr/bin/env bash
# `diffpack dev` FRESHNESS gate — an edit must reach the SERVER-RENDERED HTML.
#
# This is the check that was missing. `next-dev-hmr-check.sh` asserts the HMR push
# happened, and it passed for months while a fresh `curl` of the same route returned
# the OLD text forever: the orchestrator keyed its SSR module cache on the mtime of
# `server/server.mjs` alone, an island edit only re-emits the chunk that HOSTS the
# changed module, and even a forced entry re-import could not help because the entry
# reaches its split chunks through query-less `import("./server.chunk-N.mjs")` URLs
# that Node answers from its ESM cache. Asserting "the push happened" cannot see any
# of that. Asserting "a freshly fetched DOCUMENT contains the new string" sees all of
# it, for both edit classes:
#
#   * a "use client" island edit  — rendered into the HTML by the SSR-of-flight graph
#   * a Server Component edit     — rendered into the flight by the react-server graph
#
# Each edit is also required to land WITHOUT the server logging an error, so a
# "freshness" that comes from crashing and respawning a worker (the behaviour this
# replaced: `Module is not loaded: <id>`, then a 4-second cold restart) fails here.
#
# Native build (Rust); Node is the oracle only. Exit 0 = PASS.
# Strict mode, the ERR net (no abort is ever silent) and fail() — see _gate-prelude.sh.
source "$(dirname "$0")/_gate-prelude.sh"

repo="$(cd "$(dirname "$0")/../.." && pwd)"
fixture="${1:-$repo/integration/next-app-router}"
page="$fixture/app/page.tsx"
counter="$fixture/app/Counter.tsx"
stamp="$(date +%s)"
log="/tmp/next-dev-fresh.$stamp.log"
port=""

echo "== building diffpack (release) =="
cargo build --release --manifest-path "$repo/Cargo.toml"

if [ ! -d "$fixture/node_modules/react-server-dom-webpack" ]; then
  echo "== installing pinned RSC deps in $fixture =="
  (cd "$fixture" && npm install --no-audit --no-fund react-server-dom-webpack@19.2.4)
fi

cp "$page" "/tmp/next-dev-fresh-page.$stamp.bak"
cp "$counter" "/tmp/next-dev-fresh-counter.$stamp.bak"
cleanup() {
  if [ -n "${dev_pid:-}" ]; then
    # The dev server's full descendant tree: the node orchestrator and its
    # react-server worker outlive a bare kill of the parent. `wait` reaps the shell's
    # own job so its SIGKILL is not reported as a spurious failure line.
    pkill -P "$dev_pid" 2>/dev/null || true
    kill -9 "$dev_pid" 2>/dev/null || true
    wait "$dev_pid" 2>/dev/null || true
  fi
  pkill -f "next-server.mjs $fixture" 2>/dev/null || true
  cp "/tmp/next-dev-fresh-page.$stamp.bak" "$page"
  cp "/tmp/next-dev-fresh-counter.$stamp.bak" "$counter"
}
trap cleanup EXIT

grep -q "from-server" "$page" || fail "app/page.tsx no longer contains 'from-server' — refusing to run"
grep -q "count: " "$counter" || fail "app/Counter.tsx no longer contains 'count: ' — refusing to run"

# Refuse to run against a port someone else holds: a leaked dev server answering
# these requests would make this gate pass while measuring nothing.
port=$((20000 + RANDOM % 20000))
if lsof -tnP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
  fail "port $port is already in use — refusing to start (a leaked server would fake this gate)"
fi

echo "== starting diffpack dev on :$port =="
"$repo/target/release/diffpack" dev "$fixture" "$port" >"$log" 2>&1 &
dev_pid=$!
for _ in $(seq 1 120); do
  curl -fsS -o /dev/null -m 3 "http://127.0.0.1:$port/" 2>/dev/null && break
  sleep 1
done
curl -fsS -o /dev/null -m 5 "http://127.0.0.1:$port/" || fail "dev server never served / (see $log)"

node "$repo/scripts/rsc/dev-fresh-probe.mjs" "$port" "$fixture" "$log" || fail "freshness probe failed (see $log)"

echo "PASS: an island edit AND a server-component edit each reached a freshly fetched document, with no server error"
