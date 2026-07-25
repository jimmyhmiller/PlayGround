#!/usr/bin/env bash
#
# One-command verification gate. Run this before every change and from CI.
#
#   ./check.sh              # Rust gates (hard) + dev/HMR browser gates (if tools present)
#   ./check.sh --fast       # Rust gates only (build + lib tests + clippy)
#   ./check.sh --full       # also the broader parity/conformance wall
#   ./check.sh --strict      # a skipped browser gate (missing node/chrome/deps) FAILS
#
# Tiers:
#   1. Rust      — cargo build, cargo test --lib, cargo clippy -D warnings. ALWAYS a
#                  hard gate; the rest depends on the built binary.
#   2. Dev/HMR   — the SPA + TanStack dev-server oracles and the offline HMR harness.
#                  Skipped (not failed) when node / Chrome / fixture deps are absent,
#                  unless --strict.
#   3. Broader   — conformance + five-app behavioral parity + reference acceptance.
#                  Only with --full.
#
# Exit 0 only if every gate that RAN passed (and, with --strict, none were skipped).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

FAST=0; FULL=0; STRICT=0
for a in "$@"; do case "$a" in
  --fast) FAST=1 ;; --full) FULL=1 ;; --strict) STRICT=1 ;;
  *) echo "unknown flag: $a" >&2; exit 2 ;;
esac; done

pass=0; fail=0; skip=0; failed=""; skipped=""
step() {  # step "name" cmd...
  local name="$1"; shift
  printf '\n\033[1m=== %s ===\033[0m\n' "$name"
  if "$@"; then echo "PASS: $name"; pass=$((pass+1)); return 0
  else echo "FAIL: $name"; fail=$((fail+1)); failed="$failed\n  - $name"; return 1; fi
}
skip_step() {  # skip_step "name" "reason"
  if [ "$STRICT" = "1" ]; then
    printf '\n\033[1m=== %s ===\033[0m\nFAIL (--strict): %s\n' "$1" "$2"
    fail=$((fail+1)); failed="$failed\n  - $1 (unavailable: $2)"
  else
    printf '\n\033[1m=== %s ===\033[0m\nSKIP: %s\n' "$1" "$2"
    skip=$((skip+1)); skipped="$skipped\n  - $1 ($2)"
  fi
}

# --- Tier 1: Rust (hard gate) -------------------------------------------------
step "cargo build --release" cargo build --release || { echo "build failed — aborting"; exit 1; }
step "cargo test --release --lib" cargo test --release --lib || true
step "cargo clippy -D warnings" cargo clippy --release --all-targets -- -D warnings || true

if [ "$FAST" = "1" ]; then
  echo; echo "=== gate summary (fast): $pass passed, $fail failed ==="
  [ "$fail" = "0" ]; exit $?
fi

# --- prerequisites for browser gates -----------------------------------------
have_node=0; command -v node >/dev/null 2>&1 && have_node=1
CHROME="${CHROME:-}"
for c in "$CHROME" "$HOME/.cache/ms-playwright/chromium-1194/chrome-linux/chrome" \
         /usr/bin/google-chrome /usr/bin/chromium \
         "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"; do
  [ -n "$c" ] && [ -x "$c" ] && { CHROME="$c"; export CHROME; break; }
done
have_chrome=0; [ -n "$CHROME" ] && have_chrome=1

deps_ready() { [ -d "$1/node_modules" ]; }

# --- Tier 2: dev / HMR browser gates -----------------------------------------
dev_gate() {  # dev_gate "name" <dir> <node-script>
  local name="$1" dir="$2" script="$3"
  if [ "$have_node" = "0" ]; then skip_step "$name" "node not found"; return; fi
  if [ "$have_chrome" = "0" ]; then skip_step "$name" "Chrome not found"; return; fi
  if ! deps_ready "$dir"; then skip_step "$name" "$dir/node_modules missing (run npm ci there)"; return; fi
  step "$name" bash -c "cd '$dir' && node '$script'"
}
dev_gate "SPA dev-server HMR oracle" integration/vite-react-reference dev-check.mjs
dev_gate "TanStack dev-server HMR oracle" integration/tanstack-start-reference dev-check.mjs

if [ "$have_node" = "1" ] && [ "$have_chrome" = "1" ]; then
  step "HMR generality harness (offline)" integration/hmr-harness/run-all.sh --offline
else
  skip_step "HMR generality harness (offline)" "node/Chrome not found"
fi

# --- Tier 2b: RSC / Next spine (needs node + agent-browser + fixture deps) ----
have_ab=0; command -v agent-browser >/dev/null 2>&1 && have_ab=1
rsc_gate() {  # rsc_gate "name" <script> [needs-dir]
  local name="$1" script="$2" needs="${3:-}"
  if [ "$have_node" = "0" ]; then skip_step "$name" "node not found"; return; fi
  if [ "$have_ab" = "0" ]; then skip_step "$name" "agent-browser not found"; return; fi
  if [ -n "$needs" ] && ! deps_ready "$needs"; then skip_step "$name" "$needs/node_modules missing (npm install there)"; return; fi
  step "$name" bash "$script"
}
rsc_gate "RSC seam (client-references + __webpack_* seam)" scripts/rsc/seam-check.sh
rsc_gate "RSC server actions round-trip" scripts/rsc/action-check.sh integration/rsc-action
rsc_gate "RSC flight render + SSR" scripts/rsc/flight-check.sh integration/rsc-reference
rsc_gate "RSC minimal app (SSR + hydrate + action)" scripts/rsc/rsc-check.sh integration/rsc-reference
rsc_gate "Next app-router (real create-next-app, RSC end-to-end)" scripts/rsc/next-check.sh integration/next-app-router
rsc_gate "Next dev server (Fast Refresh island + server-component reload)" scripts/rsc/next-dev-check.sh integration/next-app-router
rsc_gate "Next UNMODIFIED create-next-app default (build + render + hydrate)" scripts/rsc/next-authentic-check.sh integration/next-app-router
rsc_gate "Next SSG (prerender + dumb static serve + hydrate + soft-nav)" scripts/rsc/next-ssg-check.sh integration/next-app-router
rsc_gate "Next corpus (multi-app SSR + classification smoke)" scripts/rsc/next-corpus-check.sh integration/next-corpus
rsc_gate "Dev HMR bench (diffpack dev vs next --turbopack, liveness/non-regression)" scripts/rsc/next-dev-hmr-check.sh integration/next-app-router

# --- Tier 3: broader wall (--full) -------------------------------------------
if [ "$FULL" = "1" ]; then
  if [ "$have_node" = "1" ]; then
    deps_ready conformance \
      && step "conformance suite" bash -c "cd conformance && node run.mjs" \
      || skip_step "conformance suite" "conformance/node_modules missing"
    if [ "$have_chrome" = "1" ]; then
      deps_ready integration/app-parity \
        && step "five-app behavioral parity" bash -c "cd integration/app-parity && node run.mjs" \
        || skip_step "five-app behavioral parity" "integration/app-parity/node_modules missing"
      deps_ready integration/vite-react-reference \
        && step "SPA build acceptance + browser" bash -c "cd integration/vite-react-reference && node acceptance.mjs diffpack --strict && node browser-check.mjs diffpack" \
        || skip_step "SPA build acceptance + browser" "deps missing"
    else
      skip_step "five-app behavioral parity" "Chrome not found"
    fi
  else
    skip_step "broader wall (--full)" "node not found"
  fi
fi

# --- summary ------------------------------------------------------------------
echo
echo "=== gate summary: $pass passed, $fail failed, $skip skipped ==="
[ -n "$failed" ] && printf 'failed:%b\n' "$failed"
[ -n "$skipped" ] && printf 'skipped:%b\n' "$skipped"
[ "$fail" = "0" ]
