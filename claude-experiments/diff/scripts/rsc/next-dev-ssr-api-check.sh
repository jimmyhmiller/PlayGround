#!/usr/bin/env bash
# `diffpack dev` SSR-API gate — rendering every route must not MISUSE react-dom.
#
# The dev orchestrator renders documents through the BUFFERED path
# (`renderFlightToDocument`, which pipes from `onAllReady`); production streams
# (`renderFlightToStream`, which pipes from `onShellReady`). Different callbacks, with
# different re-entry conditions, so next-check.sh's identical assertion on the
# production server log does not cover this one. It was a dev render of /error-demo
# that logged
#
#   next-ssr onError: React currently only supports piping to one writable stream.
#
# the same message cal.com logged once per request.
#
# What this gate is and is NOT. It is a broad sweep: every route, one read of the log,
# so ANY react-dom misuse across the whole app fails the build. It is NOT the
# regression lock for the double-`pipe` bug — whether a given route trips that one
# depends on a microtask race (whether the boundary's fallback is still pending when
# the content finishes), and a gate that catches a bug only sometimes is not a gate.
# The deterministic pair is scripts/rsc/tests/ssr-pipe-once.test.mjs (reproduces the
# upstream double call, shows the guard absorbing it) plus
# `both_renderers_pipe_at_most_once_however_often_react_says_ready` in next_adapter.rs
# (the shipped entries carry that guard).
#
# No browser, no edits: boot dev, GET every route the fixture has, read the log.
# Native build (Rust); Node is the oracle only. Exit 0 = PASS.
# Strict mode, the ERR net (no abort is ever silent) and fail() — see _gate-prelude.sh.
source "$(dirname "$0")/_gate-prelude.sh"
source "$(dirname "$0")/_react-dom-misuse.sh"

repo="$(cd "$(dirname "$0")/../.." && pwd)"
fixture="${1:-$repo/integration/next-app-router}"
stamp="$(date +%s)"
log="/tmp/next-dev-ssr-api.$stamp.log"

echo "== building diffpack (release) =="
cargo build --release --manifest-path "$repo/Cargo.toml"

cleanup() {
  if [ -n "${dev_pid:-}" ]; then
    # The orchestrator and its react-server worker outlive a bare kill of the parent.
    pkill -P "$dev_pid" 2>/dev/null || true
    kill -9 "$dev_pid" 2>/dev/null || true
    wait "$dev_pid" 2>/dev/null || true
  fi
  pkill -f "next-server.mjs $fixture" 2>/dev/null || true
}
trap cleanup EXIT

# Refuse a port someone else holds: a leaked dev server answering these requests would
# make this gate pass while rendering nothing of ours.
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

# Every document route the fixture has. /error-demo is the one that reproduces the
# double `onAllReady` (a Suspense boundary that finishes last while still holding
# abortable fallback tasks), and /blog/hello and /slow carry the other boundary shapes
# — but the point of sweeping the whole app is that the trigger is a RENDER shape, so a
# hand-picked route list would go stale the moment the fixture grows a new one.
routes="/ /about /blog/hello /dashboard /slow /gallery /products /nav-demo /error-demo /isr /use-cache /conventions-demo /meta-demo /image-demo"
rendered=0
for route in $routes; do
  code="$(curl -s -o /dev/null -m 60 -w '%{http_code}' "http://127.0.0.1:$port$route" || true)"
  [ "$code" = "200" ] || { tail -40 "$log"; fail "GET $route -> HTTP $code (expected 200; this gate asserts on what rendering LOGS, so a route that did not render proves nothing)"; }
  rendered=$((rendered + 1))
done
[ "$rendered" -ge 10 ] || fail "only $rendered routes rendered — refusing to certify a sweep this thin"
echo "OK: $rendered routes rendered through the dev orchestrator's buffered SSR path"

assert_no_react_dom_misuse "$log" "$rendered dev routes"

echo "PASS: every route rendered by \`diffpack dev\` with no react-dom API misuse in the server log"
