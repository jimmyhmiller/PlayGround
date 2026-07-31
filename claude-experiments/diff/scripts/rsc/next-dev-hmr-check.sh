#!/usr/bin/env bash
# Slice DEV-BENCH liveness gate — `diffpack dev` vs `next dev --turbopack` HMR bench.
#
# This is a NON-REGRESSION liveness row, NOT a latency threshold: dev-server
# edit-to-update latency is machine-dependent, so this gate does NOT assert
# "diffpack is Nms". It asserts the benchmark HARNESS still runs end-to-end on
# BOTH dev servers, in a REAL browser, and produces a well-formed results file —
# and, load-bearingly, that the two edited fixture files are byte-identical to
# their originals afterward (a leaked nonce edit would silently corrupt the
# next-check / next-dev / next-authentic gates that edit the same files).
#
# It runs the small config (--samples 3 --starts 2). The full measured numbers in
# docs/COMPETITIVE_BENCHMARKS.md come from the default run (node scripts/bench-dev-hmr.mjs).
# Native build (Rust); Node + Chrome (agent-browser) are the oracle only.
# Exit 0 = PASS.
# Strict mode, the ERR net (no abort is ever silent) and fail() — see _gate-prelude.sh.
source "$(dirname "$0")/_gate-prelude.sh"

repo="$(cd "$(dirname "$0")/../.." && pwd)"
fixture="$repo/integration/next-app-router"
counter="$fixture/app/Counter.tsx"
page="$fixture/app/page.tsx"
# Write to a SEPARATE file so this small (3/2) liveness run never clobbers the
# canonical bench/results/dev-hmr-results.json that docs/COMPETITIVE_BENCHMARKS.md
# quotes (produced by the default `node scripts/bench-dev-hmr.mjs`).
results="$repo/bench/results/dev-hmr-liveness.json"
stamp="$(date +%s)"

echo "== building diffpack (release) =="
cargo build --release --manifest-path "$repo/Cargo.toml"

# Same RSC + Fast Refresh runtime deps the next-dev gate needs (node_modules is
# gitignored → install on demand).
if [ ! -d "$fixture/node_modules/react-server-dom-webpack" ]; then
  echo "== installing pinned RSC deps in $fixture =="
  (cd "$fixture" && npm install --no-audit --no-fund react-server-dom-webpack@19.2.4)
fi
if [ ! -f "$fixture/node_modules/@vitejs/plugin-react/dist/refresh-runtime.js" ] \
   && [ ! -f "$fixture/node_modules/react-refresh/cjs/react-refresh-runtime.development.js" ]; then
  echo "== installing the React Fast Refresh runtime in $fixture =="
  (cd "$fixture" && npm install --no-audit --no-fund --save-dev @vitejs/plugin-react react-refresh)
fi
[ -x "$fixture/node_modules/.bin/next" ] || fail "fixture next binary missing (npm install in $fixture)"

# Snapshot the two fixture files the bench edits; ALWAYS restore + close browser.
cp "$counter" "/tmp/next-hmr-counter.$stamp.bak"
cp "$page" "/tmp/next-hmr-page.$stamp.bak"
cleanup() {
  agent-browser close 2>/dev/null || true
  pkill -f "diffpack dev $fixture" 2>/dev/null || true
  # Restore ONLY if the bench left them dirty (the bench restores itself; this is a backstop).
  cmp -s "$counter" "/tmp/next-hmr-counter.$stamp.bak" || cp "/tmp/next-hmr-counter.$stamp.bak" "$counter"
  cmp -s "$page"    "/tmp/next-hmr-page.$stamp.bak"    || cp "/tmp/next-hmr-page.$stamp.bak"    "$page"
}
trap cleanup EXIT

grep -q "count: " "$counter" || fail "app/Counter.tsx no longer contains 'count: ' — refusing to run"
grep -q "from-server" "$page" || fail "app/page.tsx no longer contains 'from-server' — refusing to run"

echo "== running bench-dev-hmr.mjs (liveness config: --samples 3 --starts 2) =="
rm -f "$results"
node "$repo/scripts/bench-dev-hmr.mjs" --samples 3 --starts 2 --out "$results"

# --- Assert the fixture files came back byte-identical (the load-bearing check) --
cmp -s "$counter" "/tmp/next-hmr-counter.$stamp.bak" || fail "app/Counter.tsx NOT restored to original after the bench (a leaked nonce edit would corrupt other gates)"
cmp -s "$page"    "/tmp/next-hmr-page.$stamp.bak"    || fail "app/page.tsx NOT restored to original after the bench"
echo "OK: both edited fixture files restored byte-identical"

# --- Assert the results JSON is well-formed for BOTH servers ---------------------
[ -f "$results" ] || fail "bench did not write $results"
node - "$results" <<'NODE'
import { readFileSync } from "node:fs";
const r = JSON.parse(readFileSync(process.argv[2], "utf8"));
const fail = (m) => { console.error("FAIL: " + m); process.exit(1); };
for (const key of ["diffpack", "next"]) {
  const s = r.servers?.[key];
  if (!s) fail(`results missing server '${key}'`);
  for (const p of ["ready", "firstByte"]) {
    if (typeof s.startup?.[p]?.median !== "number") fail(`${key}.startup.${p}.median missing`);
  }
  const classes = ["client-text (Fast Refresh)", "server-text (RSC refresh)"];
  for (const c of classes) {
    const h = s.hmr?.[c];
    if (!h) fail(`${key}.hmr['${c}'] missing`);
    if (typeof h.warm?.median !== "number") fail(`${key}.hmr['${c}'].warm.median missing`);
    if (typeof h.warmup?.delta !== "number") fail(`${key}.hmr['${c}'].warmup.delta missing (cold-first)`);
    if (!(h.warm.median > 0)) fail(`${key}.hmr['${c}'].warm.median not positive`);
  }
}
// Semantics ground truth: BOTH servers now keep a server-component edit hot — an
// in-place RSC flight refresh, no full document reload.
if (r.servers.diffpack.hmr["server-text (RSC refresh)"].semantics !== "state-preserving hot update")
  fail("diffpack server-text should be a state-preserving (RSC-refresh) hot update");
if (r.servers.next.hmr["server-text (RSC refresh)"].semantics !== "state-preserving hot update")
  fail("next server-text should be a state-preserving (RSC-refresh) hot update");
console.log("OK: results JSON well-formed for diffpack + next (startup + 2 hmr classes each)");
NODE

echo "PASS: dev HMR benchmark ran end-to-end on both dev servers, produced a well-formed results file, and left the fixture pristine"
