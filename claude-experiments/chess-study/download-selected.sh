#!/bin/bash

# Download specific Chessly courses

set -e

DATA_DIR="$(dirname "$0")/data"
mkdir -p "$DATA_DIR/courses" "$DATA_DIR/chapters" "$DATA_DIR/studies" "$DATA_DIR/variations"

CHESSLY_COOKIE="${CHESSLY_COOKIE:-__Secure-cst=4UDQHee3C785P21tPsG0-Un-r8N7GWb06mMmws8ozYLj}"

fetch() {
  curl -s "$1" -H "User-Agent: Mozilla/5.0" -H "Referer: https://chessly.com/" -H "Origin: https://chessly.com" -H "Cookie: $CHESSLY_COOKIE"
}

# Selected course IDs
COURSES=(
  "775a4eea-5a50-47da-b7dd-5790ef829fbe:Vienna"
  "d4f3504b-8bbd-4435-ae30-a0b8372c9286:Caro-Kann Defense"
  "840e407d-54cc-45e3-9f3e-bff84a26d66d:Gotham Gambit"
)

echo "=== Downloading Selected Courses ==="

for ENTRY in "${COURSES[@]}"; do
  COURSE_ID="${ENTRY%%:*}"
  COURSE_NAME="${ENTRY#*:}"

  echo ""
  echo ">>> $COURSE_NAME"

  # Get chapters
  fetch "https://cag.chessly.com/beta/openings/courses/$COURSE_ID/chapters" > "$DATA_DIR/courses/$COURSE_ID.json"

  CHAPTERS=$(jq -r '.[] | "\(.id)|\(.name)"' "$DATA_DIR/courses/$COURSE_ID.json")

  while IFS='|' read -r CHAPTER_ID CHAPTER_NAME; do
    [ -z "$CHAPTER_ID" ] && continue
    echo "    Chapter: $CHAPTER_NAME"

    # Get studies
    fetch "https://cag.chessly.com/beta/openings/courses/chapters/$CHAPTER_ID/studies" > "$DATA_DIR/chapters/$CHAPTER_ID.json"

    STUDIES=$(jq -r '.[].id' "$DATA_DIR/chapters/$CHAPTER_ID.json")

    for STUDY_ID in $STUDIES; do
      [ -z "$STUDY_ID" ] && continue

      # Get variation IDs
      VAR_IDS=$(fetch "https://cag.chessly.com/beta/openings/courses/studies/$STUDY_ID/variations")
      echo "$VAR_IDS" > "$DATA_DIR/studies/$STUDY_ID.json"

      # Download each variation
      mkdir -p "$DATA_DIR/variations/$STUDY_ID"
      for VAR_ID in $(echo "$VAR_IDS" | jq -r '.[]' 2>/dev/null); do
        fetch "https://cag.chessly.com/beta/openings/courses/studies/variations/$VAR_ID" > "$DATA_DIR/variations/$STUDY_ID/$VAR_ID.json"
        sleep 0.1
      done
    done
    sleep 0.2
  done <<< "$CHAPTERS"
done

echo ""
echo "=== Done ==="
