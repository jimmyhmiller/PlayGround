#!/usr/bin/env bash
# DeepSeek chat-completion adapter for flowline agent turns.
# Stands in for `codex exec`: composes one user message from the prompt file
# plus inlined context sections (the API cannot inspect the repository the way
# an agentic CLI can), calls the API once, and writes the assistant text to
# --output. Exits nonzero on any failure so the caller's marker-file chain
# records a failed turn.
set -euo pipefail

PROMPT_FILE=""
OUTPUT_FILE=""
CONTEXT_FILES=()
REPO_MAP=0
EXTRACT_FLOW=0
WRAP=0
INLINE_NAMED_FILES=0
SPLIT_FLOW=""

while [ $# -gt 0 ]; do
  case "$1" in
    --prompt) PROMPT_FILE="$2"; shift 2 ;;
    --output) OUTPUT_FILE="$2"; shift 2 ;;
    --context) CONTEXT_FILES+=("$2"); shift 2 ;;
    --repo-map) REPO_MAP=1; shift ;;
    --extract-flow) EXTRACT_FLOW=1; shift ;;
    --wrap) WRAP=1; shift ;;
    --inline-named-files) INLINE_NAMED_FILES=1; shift ;;
    --split-flow) SPLIT_FLOW="$2"; shift 2 ;;
    *) echo "deepseek-agent: unknown argument: $1" >&2; exit 2 ;;
  esac
done

[ -n "$PROMPT_FILE" ] && [ -f "$PROMPT_FILE" ] || { echo "deepseek-agent: --prompt file missing or unreadable: '$PROMPT_FILE'" >&2; exit 2; }
[ -n "$OUTPUT_FILE" ] || { echo "deepseek-agent: --output is required" >&2; exit 2; }

KEY="${DEEPSEEK_KEY:-${DEEPSEEK_API_KEY:-}}"
[ -n "$KEY" ] || { echo "deepseek-agent: DEEPSEEK_KEY / DEEPSEEK_API_KEY is not set" >&2; exit 2; }
BASE="${DEEPSEEK_BASE:-https://api.deepseek.com}"
BASE="${BASE%/}"
MODEL="${DEEPSEEK_MODEL:-deepseek-v4-pro}"
# Reasoning models spend from the same budget on hidden reasoning before any
# visible content, so this must stay generous or replies come back empty.
MAX_TOKENS="${DEEPSEEK_MAX_TOKENS:-32768}"
TIMEOUT="${DEEPSEEK_TIMEOUT:-300}"

command -v jq >/dev/null || { echo "deepseek-agent: jq is required" >&2; exit 2; }

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
MESSAGE="$WORK/message.md"

cat "$PROMPT_FILE" > "$MESSAGE"

for contextFile in ${CONTEXT_FILES[@]+"${CONTEXT_FILES[@]}"}; do
  if [ -s "$contextFile" ]; then
    { printf '\n\n=== %s ===\n' "$contextFile"; cat "$contextFile"; } >> "$MESSAGE"
  fi
done

# Inline the contents of tracked repository files the message names, so
# review/analysis agents work from real code instead of confabulating.
if [ "$INLINE_NAMED_FILES" = 1 ] && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git ls-files > "$WORK/tracked.txt"
  PER_FILE_LIMIT=120000
  TOTAL_LIMIT=300000
  inlinedBytes=0
  inlinedCount=0
  for candidate in $(grep -oE '[A-Za-z0-9_][A-Za-z0-9_./-]*\.[A-Za-z0-9]+' "$MESSAGE" | sed 's|^\./||' | sort -u); do
    [ "$inlinedCount" -ge 8 ] && break
    remaining=$((TOTAL_LIMIT - inlinedBytes))
    [ "$remaining" -le 0 ] && break
    if grep -qxF "$candidate" "$WORK/tracked.txt" && [ -f "$candidate" ]; then
      take=$PER_FILE_LIMIT
      [ "$remaining" -lt "$take" ] && take=$remaining
      size=$(wc -c < "$candidate")
      label="$candidate (contents)"
      [ "$size" -gt "$take" ] && label="$candidate (contents, truncated at $take of $size bytes)"
      { printf '\n\n=== %s ===\n' "$label"; head -c "$take" "$candidate"; } >> "$MESSAGE"
      inlinedBytes=$((inlinedBytes + take))
      inlinedCount=$((inlinedCount + 1))
    fi
  done
fi

if [ "$REPO_MAP" = 1 ]; then
  { printf '\n\n=== repository file map ===\n'; git ls-files 2>/dev/null | head -n 400; } >> "$MESSAGE" || true
  if [ -s README.md ]; then
    { printf '\n\n=== README.md (beginning) ===\n'; head -c 4000 README.md; } >> "$MESSAGE"
  fi
fi

if [ "$WRAP" = 1 ]; then
  printf '\n\nKeep every line of your reply under 60 characters so it fits the application panel.\n' >> "$MESSAGE"
fi

jq -n --rawfile prompt "$MESSAGE" --arg model "$MODEL" --argjson maxTokens "$MAX_TOKENS" \
  '{model: $model, messages: [{role: "user", content: $prompt}], max_tokens: $maxTokens, stream: false}' \
  > "$WORK/payload.json"

HTTP_STATUS=$(curl -sS --max-time "$TIMEOUT" -o "$WORK/response.json" -w '%{http_code}' \
  "$BASE/chat/completions" \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d @"$WORK/payload.json") || { echo "deepseek-agent: curl failed reaching $BASE" >&2; exit 1; }

if [ "$HTTP_STATUS" != 200 ]; then
  echo "deepseek-agent: HTTP $HTTP_STATUS from $BASE/chat/completions" >&2
  cat "$WORK/response.json" >&2 || true
  exit 1
fi

jq -r '.choices[0].message.content // empty' "$WORK/response.json" > "$WORK/content.md"
if [ -z "$(tr -d '[:space:]' < "$WORK/content.md")" ]; then
  echo "deepseek-agent: response contained no message content (finish_reason: $(jq -r '.choices[0].finish_reason' "$WORK/response.json")) — if finish_reason is 'length', raise DEEPSEEK_MAX_TOKENS" >&2
  jq 'del(.choices[0].message.reasoning_content)' "$WORK/response.json" >&2 || cat "$WORK/response.json" >&2
  exit 1
fi

# --split-flow: the reply carries a workflow draft in its (single) fenced
# block; route the block to the draft file and only the prose to --output.
if [ -n "$SPLIT_FLOW" ]; then
  awk 'insideFence && /^```/ {exit} insideFence {print} /^```/ && !insideFence {insideFence=1}' "$WORK/content.md" > "$WORK/draft.flow"
  if [ -s "$WORK/draft.flow" ] && head -n 1 "$WORK/draft.flow" | grep -q '^workflow '; then
    cp "$WORK/draft.flow" "$SPLIT_FLOW"
    awk 'BEGIN {skip = 0; done = 0}
         /^```/ { if (!done) { if (skip) {skip = 0; done = 1} else {skip = 1}; next } }
         skip {next}
         {print}' "$WORK/content.md" > "$WORK/prose.md"
    mv "$WORK/prose.md" "$WORK/content.md"
    echo "deepseek-agent: updated $SPLIT_FLOW"
  fi
fi

cp "$WORK/content.md" "$OUTPUT_FILE"

if [ "$EXTRACT_FLOW" = 1 ]; then
  flowPath=$(grep -m 1 -E '^ *file: *workflows/[A-Za-z0-9._-]+\.flow *$' "$WORK/content.md" | sed -E 's/^ *file: *//; s/ *$//' || true)
  if [ -z "$flowPath" ]; then
    echo "deepseek-agent: reply has no 'file: workflows/<name>.flow' line; cannot write the workflow" >&2
    exit 1
  fi
  awk 'insideFence && /^```/ {exit} insideFence {print} /^```/ && !insideFence {insideFence=1}' "$WORK/content.md" > "$WORK/program.flow"
  if ! [ -s "$WORK/program.flow" ]; then
    echo "deepseek-agent: reply has no fenced code block containing the workflow" >&2
    exit 1
  fi
  if ! head -n 1 "$WORK/program.flow" | grep -q '^workflow '; then
    echo "deepseek-agent: extracted block does not start with a 'workflow' line:" >&2
    head -n 3 "$WORK/program.flow" >&2
    exit 1
  fi
  mkdir -p workflows
  cp "$WORK/program.flow" "$flowPath"
  echo "deepseek-agent: wrote $flowPath"
fi
