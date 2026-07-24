#!/usr/bin/env bash
#
# Reusable HMR generality harness (test-only; NEVER part of the build).
#
# Runs `diffpack dev` against ONE real app and proves, in a real browser driven by
# `agent-browser`, that a component edit produces a STATE-PRESERVING React Fast
# Refresh update — the daily-workflow payoff of diffpack's low diff times, verified
# on apps beyond the pinned fixtures.
#
# It is deliberately dependency-light: the ONLY tools it needs are the built
# `diffpack` binary, `agent-browser` (a global CLI — no per-app puppeteer), and the
# app's own `npm install`. That is what makes it point-at-any-app portable and
# CI-friendly.
#
# Usage:
#   integration/hmr-harness/hmr-check.sh <app.conf> [--keep]
#
# An app.conf is a small KEY=VALUE file (see apps/*.conf). Fields:
#   NAME       human label
#   LOCAL_DIR  app dir relative to the repo root (use an already-checked-out app);
#              OR set REPO (+ optional COMMIT) to shallow-clone one into a work dir
#   REPO       git URL to clone when LOCAL_DIR is unset
#   COMMIT     commit to pin the clone to (reproducible; strongly recommended)
#   INSTALL    dependency install command (default: npm install --no-audit --no-fund)
#   EDIT_FILE  component file to edit, relative to the app dir
#   FIND       exact substring in EDIT_FILE to replace (must be present)
#   REPLACE    replacement base text (the harness appends a unique stamp)
#   MOUNT_TEXT text expected on the initial render (default: FIND)
#   STATE      preserve | reload  (default: preserve — expect Fast Refresh)
#   IGNORE_ERR extra regex of console errors to ignore (joined with the defaults)
#
# Exit: 0 = pass, 1 = assertion failed, 2 = usage, 3 = missing tool/setup.
set -euo pipefail

# --- locate the repo + binary -------------------------------------------------
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
DIFFPACK="${DIFFPACK:-$REPO_ROOT/target/release/diffpack}"

die() { echo "ERROR: $*" >&2; exit "${2:-1}"; }

[ $# -ge 1 ] || die "usage: hmr-check.sh <app.conf> [--keep]" 2
CONF="$1"; shift
KEEP=0
[ "${1:-}" = "--keep" ] && KEEP=1
[ -f "$CONF" ] || die "config not found: $CONF" 2

# --- config -------------------------------------------------------------------
NAME=""; LOCAL_DIR=""; REPO=""; COMMIT=""; EDIT_FILE=""; FIND=""; REPLACE=""
MOUNT_TEXT=""; STATE="preserve"; IGNORE_ERR=""
INSTALL="npm install --no-audit --no-fund"
# shellcheck disable=SC1090
source "$CONF"
: "${NAME:?conf must set NAME}"
: "${EDIT_FILE:?conf must set EDIT_FILE}"
: "${FIND:?conf must set FIND}"
: "${REPLACE:?conf must set REPLACE}"
MOUNT_TEXT="${MOUNT_TEXT:-$FIND}"

# --- tools: diffpack + agent-browser -----------------------------------------
if [ ! -x "$DIFFPACK" ]; then
  echo "diffpack binary missing; building release..."
  ( cd "$REPO_ROOT" && cargo build --release ) || die "cargo build failed" 3
fi

if command -v agent-browser >/dev/null 2>&1; then
  AB="agent-browser"
elif command -v npx >/dev/null 2>&1 && npx --no-install agent-browser --version >/dev/null 2>&1; then
  AB="npx agent-browser"
else
  die "agent-browser not found. Install once with:
      npm install -g agent-browser && agent-browser install   # add --with-deps on Linux CI
   (or make 'npx agent-browser' resolvable)" 3
fi

# Ensure the browser binary is present (idempotent; a no-op once installed).
if ! $AB open about:blank >/dev/null 2>&1; then
  echo "Installing agent-browser browser binaries..."
  if [ "$(uname)" = "Linux" ]; then $AB install --with-deps; else $AB install; fi
fi
$AB close --all >/dev/null 2>&1 || true

SESSION="hmr-$NAME-$$"
WORK=""
DEV_PID=""
APP_DIR=""
PASS=0

cleanup() {
  set +e
  [ -n "$BACKUP" ] && [ -f "$BACKUP" ] && mv -f "$BACKUP" "$APP_DIR/$EDIT_FILE" 2>/dev/null
  [ -n "$DEV_PID" ] && kill "$DEV_PID" 2>/dev/null
  $AB --session "$SESSION" close >/dev/null 2>&1
  if [ -n "$WORK" ] && [ "$KEEP" = "0" ]; then rm -rf "$WORK"; fi
}
BACKUP=""
trap cleanup EXIT

# --- resolve the app dir (local or cloned) -----------------------------------
if [ -n "$LOCAL_DIR" ]; then
  APP_DIR="$REPO_ROOT/$LOCAL_DIR"
  [ -d "$APP_DIR" ] || die "LOCAL_DIR not found: $APP_DIR"
else
  [ -n "$REPO" ] || die "conf must set LOCAL_DIR or REPO"
  WORK="$(mktemp -d "${TMPDIR:-/tmp}/hmr-$NAME.XXXXXX")"
  APP_DIR="$WORK/app"
  echo "[$NAME] cloning $REPO${COMMIT:+ @ ${COMMIT:0:10}}..."
  if [ -n "$COMMIT" ]; then
    git init -q "$APP_DIR"
    ( cd "$APP_DIR" && git remote add origin "$REPO" \
        && git fetch -q --depth 1 origin "$COMMIT" && git checkout -q FETCH_HEAD )
  else
    git clone -q --depth 1 "$REPO" "$APP_DIR"
  fi
fi

# --- deps ---------------------------------------------------------------------
if [ ! -d "$APP_DIR/node_modules" ]; then
  echo "[$NAME] installing deps ($INSTALL)..."
  ( cd "$APP_DIR" && eval "$INSTALL" ) || die "dependency install failed"
fi
[ -f "$APP_DIR/$EDIT_FILE" ] || die "EDIT_FILE not found: $APP_DIR/$EDIT_FILE"
grep -qF "$FIND" "$APP_DIR/$EDIT_FILE" || die "FIND text not present in $EDIT_FILE: '$FIND'"

# --- free port + boot diffpack dev -------------------------------------------
free_port() {
  local p
  for p in $(seq 9200 9400); do
    if ! (exec 3<>"/dev/tcp/127.0.0.1/$p") 2>/dev/null; then echo "$p"; return 0; fi
    exec 3>&- 2>/dev/null || true
  done
  return 1
}
PORT="$(free_port)" || die "no free port"
BASE="http://127.0.0.1:$PORT"
DEVLOG="$(mktemp "${TMPDIR:-/tmp}/hmr-$NAME-dev.XXXXXX")"
echo "[$NAME] diffpack dev on :$PORT (app: $APP_DIR)"
( cd "$APP_DIR" && exec "$DIFFPACK" dev . "$PORT" ) >"$DEVLOG" 2>&1 &
DEV_PID=$!
disown "$DEV_PID" 2>/dev/null || true  # keep bash from printing "Terminated" on cleanup

for _ in $(seq 1 200); do
  curl -sf "$BASE/" >/dev/null 2>&1 && break
  kill -0 "$DEV_PID" 2>/dev/null || { echo "--- dev log ---"; tail -20 "$DEVLOG"; die "dev server exited during startup"; }
  sleep 0.3
done
curl -sf "$BASE/" >/dev/null 2>&1 || { tail -20 "$DEVLOG"; die "dev server did not come up"; }

# --- drive the browser --------------------------------------------------------
echo "[$NAME] loading $BASE/ ..."
$AB --session "$SESSION" open "$BASE/" >/dev/null
$AB --session "$SESSION" wait --text "$MOUNT_TEXT" >/dev/null \
  || die "app did not render the expected mount text: '$MOUNT_TEXT'"

# Install a state probe + error capture on the live page.
$AB --session "$SESSION" eval --stdin >/dev/null <<'JS'
window.__hmr_probe = 'alive';
window.__hmrErr = [];
var _e = console.error;
console.error = function () { try { window.__hmrErr.push(Array.prototype.join.call(arguments, ' ')); } catch (_) {} _e.apply(console, arguments); };
window.addEventListener('error', function (ev) { window.__hmrErr.push(String((ev && (ev.message || ev.error)) || ev)); });
'installed';
JS

# Edit the component: replace FIND with a unique new string.
STAMP="$(date +%s)-$$"
NEW="${REPLACE} ${STAMP}"
BACKUP="$(mktemp "${TMPDIR:-/tmp}/hmr-$NAME-edit.XXXXXX")"
cp "$APP_DIR/$EDIT_FILE" "$BACKUP"
F="$FIND" R="$NEW" perl -0777 -pi -e 'BEGIN{$f=$ENV{F};$r=$ENV{R};} s/\Q$f\E/$r/g' "$APP_DIR/$EDIT_FILE"

# The Fast Refresh update should swap in the new text.
if ! $AB --session "$SESSION" wait --text "$NEW" >/dev/null 2>&1; then
  echo "[$NAME] FAIL: edited text never appeared (no hot update)"
  echo "--- dev log (tail) ---"; tail -8 "$DEVLOG"
  exit 1
fi

# Read the outcome: did the page-scoped probe survive (Fast Refresh, no reload)?
RESULT="$($AB --session "$SESSION" eval --stdin <<'JS'
(function () {
  var errs = (window.__hmrErr || []).slice(0, 5);
  return 'PROBE=' + (window.__hmr_probe || 'GONE') + '|ERRS=' + JSON.stringify(errs);
})();
JS
)"
# The CLI prints the string quoted; strip the outer quotes for grepping.
RESULT="${RESULT%\"}"; RESULT="${RESULT#\"}"

PROBE="GONE"
case "$RESULT" in *PROBE=alive*) PROBE="alive";; esac

# Filter console errors: drop known-ignorable (missing static assets) + conf extra.
IGNORE_RE='Failed to load resource|favicon|icons\.svg'
[ -n "$IGNORE_ERR" ] && IGNORE_RE="$IGNORE_RE|$IGNORE_ERR"
ERR_JSON="${RESULT#*ERRS=}"
REAL_ERRS="$(printf '%s' "$ERR_JSON" | tr ',' '\n' | grep -vE "$IGNORE_RE" | grep -E '[A-Za-z]' || true)"

# --- assert -------------------------------------------------------------------
echo "[$NAME] rebuild: $(grep -Eo 'rebuilt .*' "$DEVLOG" | tail -1)"
echo "[$NAME] probe after edit: $PROBE (expected: $STATE)"

OK=1
if [ "$STATE" = "preserve" ]; then
  [ "$PROBE" = "alive" ] || { echo "[$NAME] FAIL: page reloaded — Fast Refresh did NOT preserve state"; OK=0; }
else
  [ "$PROBE" = "GONE" ] || { echo "[$NAME] note: expected a reload but state was preserved"; }
fi
if [ -n "$REAL_ERRS" ]; then
  echo "[$NAME] FAIL: uncaught JS errors during the hot update:"; printf '  %s\n' "$REAL_ERRS"; OK=0
fi

if [ "$OK" = "1" ]; then
  echo "PASS [$NAME]: state-preserving Fast Refresh on a real edit"
  PASS=1
  exit 0
else
  echo "FAIL [$NAME]"
  exit 1
fi
