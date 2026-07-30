#!/bin/zsh
# cal.com's own Playwright suite as the oracle, against a diffpack PRODUCTION build or
# against `diffpack dev`.
#
#   scripts/calcom-e2e.sh prod <log-dir>
#   scripts/calcom-e2e.sh dev  <log-dir>
#   scripts/calcom-e2e.sh prod <log-dir> --grep "login"
#
# Protocol, from docs/STATUS_2026-07-28.md §10-11 — every line of it cost a wasted run
# to learn:
#
#   * PLAYWRIGHT_HEADLESS=1, always. cal.com's config is `headless = !!process.env.CI ||
#     !!process.env.PLAYWRIGHT_HEADLESS`, so a bare run opens a real Chrome window per
#     worker. NEVER use CI=1 for it: `reuseExistingServer: !process.env.CI` would then be
#     false and Playwright would start cal.com's OWN `yarn start` instead of measuring
#     the server this script started. It also gates `maxFailures: 10`, which is what
#     aborts a red run early.
#   * Run from the cal.com REPO ROOT with --config: the project definitions and their
#     testDir live there. From apps/web Playwright reports no projects at all.
#   * The pristine template DB is restored before every run. The suite mutates data.
#   * The :3000 owner is checked first. A leaked listener answering from a previous run
#     once invalidated an entire measurement.
#   * The server gets the app's own .env. `next start` loads it; our orchestrator is
#     plain Node, and without it next-auth has no secret and Prisma connects to a
#     database named after the unix user (both were seen, as 500s on /apps).
set -u

MODE=${1:?usage: calcom-e2e.sh <prod|dev> <log-dir> [extra playwright args...]}
OUT=${2:?usage: calcom-e2e.sh <prod|dev> <log-dir> [extra playwright args...]}
shift 2 || true

APP=${CALCOM_APP:-/tmp/dpe2e/calcom}
WEB=$APP/apps/web
REPO=${DIFFPACK_REPO:-$(cd "$(dirname "$0")/.." && pwd)}
DIFFPACK=$REPO/target/release/diffpack
export PATH=/opt/homebrew/opt/postgresql@17/bin:$PATH
export FORCE_COLOR=0
export PLAYWRIGHT_HEADLESS=1
mkdir -p "$OUT"

if [[ "$MODE" != "prod" && "$MODE" != "dev" ]]; then
  echo "ABORT: mode must be prod or dev, got $MODE"
  exit 2
fi
[[ -x "$DIFFPACK" ]] || { echo "ABORT: no $DIFFPACK — cargo build --release"; exit 2 }

owner=$(lsof -tnP -iTCP:3000 -sTCP:LISTEN 2>/dev/null | tr '\n' ' ')
if [[ -n "$owner" ]]; then
  echo "ABORT: port 3000 is held by pid(s) $owner — refusing to run against another server"
  exit 2
fi

echo "== restoring the pristine template database =="
psql -d postgres -qc 'DROP DATABASE IF EXISTS calcom_diffpack_e2e' >/dev/null 2>&1
psql -d postgres -qc 'CREATE DATABASE calcom_diffpack_e2e TEMPLATE calcom_e2e_pristine' || exit 3
echo "   users: $(psql -d calcom_diffpack_e2e -tAc 'select count(*) from users' | tr -d ' ')"

# The app's environment, for the server only. Exported here so both modes get it.
set -a
source "$APP/.env"
set +a

if [[ "$MODE" == "prod" ]]; then
  echo "== diffpack production build =="
  rm -rf "$WEB/.diffpack-output"
  ( cd "$WEB" && "$DIFFPACK" build-app . production ) > "$OUT/build.log" 2>&1 \
    || { echo "BUILD FAILED:"; tail -25 "$OUT/build.log"; exit 4 }
  echo "   client.js $(du -h "$WEB/.diffpack-output/public/client.js" | cut -f1), \
$(ls "$WEB/.diffpack-output/public"/client.*.js 2>/dev/null | wc -l | tr -d ' ') client chunks"
  echo "== serving the build on :3000 =="
  ( cd "$WEB" && NODE_ENV=production node "$WEB/.diffpack-output/next-server.mjs" \
      "$WEB/.diffpack-output" 3000 ) > "$OUT/server.log" 2>&1 &
else
  echo "== diffpack dev on :3000 =="
  rm -rf "$WEB/.diffpack-output"
  ( cd "$REPO" && "$DIFFPACK" dev "$WEB" 3000 ) > "$OUT/server.log" 2>&1 &
fi
server_pid=$!

code=""
for i in {1..240}; do
  code=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:3000/auth/login 2>/dev/null)
  [[ "$code" == "200" ]] && break
  sleep 1
done
if [[ "$code" != "200" ]]; then
  echo "SERVER NEVER ANSWERED 200 (last $code):"; tail -25 "$OUT/server.log"
  kill $server_pid 2>/dev/null; exit 5
fi
echo "   up (pid $server_pid), /auth/login -> 200"

# The five files the recorded 69-test selection came from: login, event types, booking
# pages, the app-router 404, and theme.
echo "== suite (headless) =="
( cd "$APP" && "$APP/node_modules/.bin/playwright" test \
    --config playwright.config.ts \
    --project=@calcom/web \
    apps/web/playwright/login.e2e.ts \
    apps/web/playwright/event-types.e2e.ts \
    apps/web/playwright/booking-pages.e2e.ts \
    apps/web/playwright/app-router-not-found.e2e.ts \
    apps/web/playwright/change-theme.e2e.ts \
    --reporter=list "$@" ) > "$OUT/suite.log" 2>&1
suite_status=$?

# Kill the whole descendant tree: both server modes outlive a bare kill of the parent
# (dev keeps its react-server worker, the orchestrator keeps its render children).
for pid in $(pgrep -P $server_pid) $server_pid; do kill $pid 2>/dev/null; done
sleep 2
for pid in $(pgrep -P $server_pid) $server_pid; do kill -9 $pid 2>/dev/null; done

echo "== result (${MODE}) =="
grep -E "^\s+[0-9]+ (passed|failed|skipped|flaky)|passed \(|failed \(" "$OUT/suite.log" | tail -6
echo "exit $suite_status; full log $OUT/suite.log"
