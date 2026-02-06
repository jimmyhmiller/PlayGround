#!/usr/bin/env bash
set -euo pipefail

SLEEP_SECS="${SLEEP_SECS:-1}"
AUTO="${AUTO:-0}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_PATH="${ROOT_DIR}/target/debug/scope"

IS_TTY=0
if [[ -t 1 ]] && command -v tput >/dev/null 2>&1; then
  IS_TTY=1
fi

TMP_HOME="$(mktemp -d)"
export HOME="${TMP_HOME}"
export USER="demo"

PROJECT="demo"

ID=""
UPDATED=0
COMMENTED=0
CLOSED=0

LAST_CMD=""
LAST_OUTPUT=""

pause_auto() {
  if [[ "${AUTO}" == "1" ]]; then
    sleep "${SLEEP_SECS}"
  fi
}

enter_screen() {
  if [[ "${IS_TTY}" == "1" ]]; then
    tput smcup || true
    tput civis || true
    clear || true
  fi
}

exit_screen() {
  if [[ "${IS_TTY}" == "1" ]]; then
    tput cnorm || true
    tput rmcup || true
  fi
}

trap exit_screen EXIT

ensure_id() {
  if [[ -z "${ID}" ]]; then
    step_2
  fi
}

run_cmd() {
  local cmd=("$@")
  LAST_CMD="$*"
  local out
  out="$("${cmd[@]}" 2>&1 || true)"
  if [[ -n "${LAST_OUTPUT}" ]]; then
    LAST_OUTPUT+=$'\n'
  fi
  LAST_OUTPUT+="$ ${LAST_CMD}"$'\n'
  LAST_OUTPUT+="${out}"
}

reset_output() {
  LAST_CMD=""
  LAST_OUTPUT=""
}

line() {
  local cols
  if [[ "${IS_TTY}" == "1" ]]; then
    cols="$(tput cols 2>/dev/null || echo 80)"
  else
    cols=80
  fi
  printf '%*s\n' "${cols}" '' | tr ' ' '='
}

render() {
  local title="$1"
  local step="$2"
  local total="$3"
  local desc="$4"

  if [[ "${IS_TTY}" == "1" ]]; then
    clear || true
  fi
  line
  printf "Scope CLI Demo | project: %s | issue: %s\n" "${PROJECT}" "${ID:-<not created>}"
  printf "HOME: %s | root: %s\n" "${HOME}" "${HOME}/.scope/projects/${PROJECT}"
  line
  printf "Step %s/%s: %s\n\n" "${step}" "${total}" "${title}"
  printf "%s\n\n" "${desc}"

  line
  printf "Controls: n=next  p=prev  r=repeat  a=auto  j=jump  q=quit\n"
  line
  printf "Output:\n"
  if [[ -n "${LAST_OUTPUT}" ]]; then
    printf "%s\n" "${LAST_OUTPUT}"
  else
    printf "(no output yet)\n"
  fi
}

step_1_desc() {
  cat <<'EOF'
Initialize a project in the isolated demo HOME. This creates:
- issues/, events/, index/, and project.toml
EOF
}

step_1() {
  reset_output
  run_cmd "${BIN_PATH}" issues init --project "${PROJECT}"
}

step_2_desc() {
  cat <<'EOF'
Create a new issue. IDs are human-readable (SC:<adjective>-<adjective>-<animal>).
EOF
}

step_2() {
  reset_output
  if [[ -z "${ID}" ]]; then
    ID="$("${BIN_PATH}" issues create --project "${PROJECT}" --title "Add blocked filter" --priority p1 --assignee jimmy --label cli)"
  fi
  LAST_OUTPUT="Created issue: ${ID}"
}

step_3_desc() {
  cat <<'EOF'
List issues from the local index (fast path).
EOF
}

step_3() {
  reset_output
  run_cmd "${BIN_PATH}" issues list --project "${PROJECT}"
}

step_4_desc() {
  cat <<'EOF'
Show the issue markdown with YAML frontmatter + body.
EOF
}

step_4() {
  ensure_id
  reset_output
  run_cmd "${BIN_PATH}" issues show --project "${PROJECT}" "${ID}"
}

step_5_desc() {
  cat <<'EOF'
Update status and labels (writes to events + updates snapshot + index).
EOF
}

step_5() {
  ensure_id
  reset_output
  if [[ "${UPDATED}" == "0" ]]; then
    run_cmd "${BIN_PATH}" issues update --project "${PROJECT}" "${ID}" --status in_progress --add-label planning
    UPDATED=1
  else
    LAST_OUTPUT="(already updated)"
  fi
}

step_6_desc() {
  cat <<'EOF'
Add a comment (stored as an event, rendered into the snapshot).
EOF
}

step_6() {
  ensure_id
  reset_output
  if [[ "${COMMENTED}" == "0" ]]; then
    run_cmd "${BIN_PATH}" issues comments add --project "${PROJECT}" "${ID}" --body "Need to confirm expected output shape."
    COMMENTED=1
  else
    LAST_OUTPUT="(comment already added)"
  fi
}

step_7_desc() {
  cat <<'EOF'
List comments directly from the event log.
EOF
}

step_7() {
  ensure_id
  reset_output
  run_cmd "${BIN_PATH}" issues comments list --project "${PROJECT}" "${ID}"
}

step_8_desc() {
  cat <<'EOF'
List issues in JSON for automation and agents.
EOF
}

step_8() {
  ensure_id
  reset_output
  run_cmd "${BIN_PATH}" issues list --project "${PROJECT}" --json
}

step_9_desc() {
  cat <<'EOF'
Inspect on-disk files: snapshot, index, and event log.
EOF
}

step_9() {
  ensure_id
  reset_output
  ISSUE_PATH="${HOME}/.scope/projects/${PROJECT}/issues/${ID}.md"
  INDEX_PATH="${HOME}/.scope/projects/${PROJECT}/index/issues.json"
  EVENTS_PATH="${HOME}/.scope/projects/${PROJECT}/events/${ID}.jsonl"

  run_cmd ls -la "${HOME}/.scope/projects/${PROJECT}"
  run_cmd sed -n '1,120p' "${ISSUE_PATH}"
  run_cmd sed -n '1,120p' "${INDEX_PATH}"
  run_cmd sed -n '1,120p' "${EVENTS_PATH}"
}

step_10_desc() {
  cat <<'EOF'
Close the issue (status=done). This appends a close event.
EOF
}

step_10() {
  ensure_id
  reset_output
  if [[ "${CLOSED}" == "0" ]]; then
    run_cmd "${BIN_PATH}" issues close --project "${PROJECT}" "${ID}"
    CLOSED=1
  else
    LAST_OUTPUT="(already closed)"
  fi
}

step_11_desc() {
  cat <<'EOF'
Show final state and wrap the demo.
EOF
}

step_11() {
  ensure_id
  reset_output
  run_cmd "${BIN_PATH}" issues show --project "${PROJECT}" "${ID}"
  LAST_OUTPUT+=$'\n\nDemo complete.'
}

STEPS=(step_1 step_2 step_3 step_4 step_5 step_6 step_7 step_8 step_9 step_10 step_11)
TITLES=(
  "Initialize project"
  "Create issue"
  "List issues"
  "Show issue"
  "Update status + labels"
  "Add comment"
  "List comments"
  "List issues (JSON)"
  "Inspect files"
  "Close issue"
  "Final state"
)

DESCS=(
  step_1_desc
  step_2_desc
  step_3_desc
  step_4_desc
  step_5_desc
  step_6_desc
  step_7_desc
  step_8_desc
  step_9_desc
  step_10_desc
  step_11_desc
)

TOTAL_STEPS="${#STEPS[@]}"

say_build() {
  reset_output
  LAST_OUTPUT="Building scope..."
  run_cmd cargo build -q
  if [[ ! -x "${BIN_PATH}" ]]; then
    echo "Binary not found at ${BIN_PATH}"
    exit 1
  fi
}

enter_screen
say_build
reset_output
step_1

current=1
while true; do
  render "${TITLES[$((current-1))]}" "${current}" "${TOTAL_STEPS}" "$("${DESCS[$((current-1))]}")"

  if [[ "${AUTO}" == "1" || "${IS_TTY}" == "0" ]]; then
    if [[ "${current}" -ge "${TOTAL_STEPS}" ]]; then
      break
    fi
    current=$((current + 1))
    "${STEPS[$((current-1))]}"
    pause_auto
    continue
  fi

  printf "\n[n]ext [p]rev [r]epeat [a]uto [j]ump [q]uit > "
  IFS= read -r -n 1 choice
  printf "\n"

  case "${choice}" in
    n|N)
      if [[ "${current}" -lt "${TOTAL_STEPS}" ]]; then
        current=$((current + 1))
        "${STEPS[$((current-1))]}"
      fi
      ;;
    p|P)
      if [[ "${current}" -gt 1 ]]; then
        current=$((current - 1))
        "${STEPS[$((current-1))]}"
      fi
      ;;
    r|R)
      "${STEPS[$((current-1))]}"
      ;;
    a|A)
      AUTO=1
      ;;
    j|J)
      printf "Step (1-%s): " "${TOTAL_STEPS}"
      read -r stepnum
      if [[ "${stepnum}" =~ ^[0-9]+$ ]] && [[ "${stepnum}" -ge 1 ]] && [[ "${stepnum}" -le "${TOTAL_STEPS}" ]]; then
        current="${stepnum}"
        "${STEPS[$((current-1))]}"
      fi
      ;;
    q|Q)
      break
      ;;
    *)
      ;;
  esac
done
