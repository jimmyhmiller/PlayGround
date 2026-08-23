#!/usr/bin/env bash
# Shared config + helpers for the ox-runner task system.
# Sourced by ox, ox-daemon and ox-run. Not executable on its own.

OX_HOME="${OX_HOME:-$HOME/ox-runner}"

OX_INBOX="$OX_HOME/inbox"
OX_PROCESSING="$OX_HOME/processing"
OX_DONE="$OX_HOME/done"
OX_FAILED="$OX_HOME/failed"
OX_RUNS="$OX_HOME/runs"
OX_STATE="$OX_HOME/state"
# Real project checkouts live in the canonical projects root, matching the
# layout on the laptop. Runs never work in them directly; they get worktrees.
OX_PROJECTS="${OX_PROJECTS:-$HOME/Documents/Code/projects}"
OX_WORK="$OX_HOME/work"
OX_LOGS="$OX_HOME/logs"
OX_BIN="$OX_HOME/bin"
OX_CONFIG="$OX_HOME/config"
OX_REPOS="$OX_CONFIG/repos.tsv"
OX_RUNSLOG="$OX_HOME/runs.jsonl"
OX_ENV="$OX_HOME/env"

# Defaults, overridable per-task via frontmatter or globally via $OX_HOME/config/settings
OX_MODEL_DEFAULT="stealth/ox-alpha"
OX_PROFILE_DEFAULT="ox"
OX_EFFORT_DEFAULT="high"
OX_TIMEOUT_DEFAULT=3600
OX_MAX_ATTEMPTS_DEFAULT=3
OX_CONCURRENCY_DEFAULT=2
OX_SANDBOX_DEFAULT="danger-full-access"
OX_POLL_INTERVAL_DEFAULT=2
OX_SETTLE_SECONDS_DEFAULT=2
OX_BRANCH_PREFIX_DEFAULT="ox"
OX_COMMIT_NAME_DEFAULT="Ox Alpha Runner"
OX_COMMIT_EMAIL_DEFAULT="jimmyhmiller@gmail.com"

[ -f "$OX_CONFIG/settings" ] && . "$OX_CONFIG/settings"

OX_TIMEOUT="${OX_TIMEOUT:-$OX_TIMEOUT_DEFAULT}"
OX_MAX_ATTEMPTS="${OX_MAX_ATTEMPTS:-$OX_MAX_ATTEMPTS_DEFAULT}"
OX_CONCURRENCY="${OX_CONCURRENCY:-$OX_CONCURRENCY_DEFAULT}"
OX_SANDBOX="${OX_SANDBOX:-$OX_SANDBOX_DEFAULT}"
OX_POLL_INTERVAL="${OX_POLL_INTERVAL:-$OX_POLL_INTERVAL_DEFAULT}"
OX_SETTLE_SECONDS="${OX_SETTLE_SECONDS:-$OX_SETTLE_SECONDS_DEFAULT}"
OX_BRANCH_PREFIX="${OX_BRANCH_PREFIX:-$OX_BRANCH_PREFIX_DEFAULT}"
OX_COMMIT_NAME="${OX_COMMIT_NAME:-$OX_COMMIT_NAME_DEFAULT}"
OX_COMMIT_EMAIL="${OX_COMMIT_EMAIL:-$OX_COMMIT_EMAIL_DEFAULT}"
OX_MODEL="${OX_MODEL:-$OX_MODEL_DEFAULT}"
OX_PROFILE="${OX_PROFILE:-$OX_PROFILE_DEFAULT}"
OX_EFFORT="${OX_EFFORT:-$OX_EFFORT_DEFAULT}"

ox_load_env() {
  if [ -f "$OX_ENV" ]; then
    set -a
    # shellcheck disable=SC1090
    . "$OX_ENV"
    set +a
  fi
}

ox_ensure_dirs() {
  mkdir -p "$OX_INBOX" "$OX_PROCESSING" "$OX_DONE" "$OX_FAILED" \
           "$OX_RUNS" "$OX_STATE" "$OX_PROJECTS" "$OX_WORK" "$OX_LOGS" \
           "$OX_CONFIG"
  touch "$OX_REPOS" "$OX_RUNSLOG"
}

ox_now()     { date -u +%Y-%m-%dT%H:%M:%SZ; }
ox_epoch()   { date -u +%s; }
ox_stamp()   { date -u +%Y%m%d-%H%M%S; }

ox_log() {
  printf '%s [%s] %s\n' "$(ox_now)" "${OX_LOG_TAG:-ox}" "$*" >&2
}

# Turn arbitrary text into a safe branch/id slug.
ox_slug() {
  printf '%s' "$1" \
    | tr '[:upper:]' '[:lower:]' \
    | sed -e 's/[^a-z0-9]\+/-/g' -e 's/^-\+//' -e 's/-\+$//' \
    | cut -c1-48
}

# JSON-escape a string using python (always available, no jq dependency for input).
ox_json_str() {
  python3 -c 'import json,sys; sys.stdout.write(json.dumps(sys.stdin.read()))'
}

# Look up a registered repo. Prints "url<TAB>base" or returns 1.
ox_repo_lookup() {
  local name="$1"
  awk -v n="$name" -F'\t' '$1==n {print $2 "\t" $3; found=1; exit} END{exit !found}' "$OX_REPOS"
}

ox_repo_names() { awk -F'\t' 'NF{print $1}' "$OX_REPOS"; }

# Path to the project checkout for a repo name.
ox_project_path() { printf '%s/%s' "$OX_PROJECTS" "$1"; }

ox_die() { ox_log "FATAL: $*"; exit 1; }
