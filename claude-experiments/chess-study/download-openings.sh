#!/bin/bash

# Chessly Opening Data Downloader
# Downloads: courses -> chapters -> studies -> variations

set -e

DATA_DIR="$(dirname "$0")/data"
mkdir -p "$DATA_DIR"

# Cookie for authentication
CHESSLY_COOKIE="${CHESSLY_COOKIE:-__Secure-cst=4UDQHee3C785P21tPsG0-Un-r8N7GWb06mMmws8ozYLj}"

fetch() {
  curl -s "$1" \
    -H "User-Agent: Mozilla/5.0" \
    -H "Referer: https://chessly.com/" \
    -H "Origin: https://chessly.com" \
    -H "Cookie: $CHESSLY_COOKIE"
}

echo "=== Chessly Opening Data Downloader ==="
echo ""

# Step 1: Download courses
echo "[1/4] Downloading courses list..."
fetch "https://cag.chessly.com/beta/openings/courses" > "$DATA_DIR/courses.json"
COURSE_COUNT=$(jq 'length' "$DATA_DIR/courses.json")
echo "       Found $COURSE_COUNT courses"

mkdir -p "$DATA_DIR/courses"
mkdir -p "$DATA_DIR/chapters"
mkdir -p "$DATA_DIR/studies"
mkdir -p "$DATA_DIR/variations"

# Step 2: Download chapters for each course
echo ""
echo "[2/4] Downloading chapters for each course..."
COURSE_IDS=$(jq -r '.[].id' "$DATA_DIR/courses.json")

for COURSE_ID in $COURSE_IDS; do
  COURSE_NAME=$(jq -r ".[] | select(.id==\"$COURSE_ID\") | .name" "$DATA_DIR/courses.json")
  echo "       Course: $COURSE_NAME"
  fetch "https://cag.chessly.com/beta/openings/courses/$COURSE_ID/chapters" > "$DATA_DIR/courses/$COURSE_ID.json"
  sleep 0.3
done

# Step 3: Download studies for each chapter
echo ""
echo "[3/4] Downloading studies for each chapter..."
for COURSE_FILE in "$DATA_DIR/courses"/*.json; do
  CHAPTER_IDS=$(jq -r '.[].id' "$COURSE_FILE" 2>/dev/null || echo "")
  for CHAPTER_ID in $CHAPTER_IDS; do
    CHAPTER_NAME=$(jq -r ".[] | select(.id==\"$CHAPTER_ID\") | .name" "$COURSE_FILE")
    echo "       Chapter: $CHAPTER_NAME"
    fetch "https://cag.chessly.com/beta/openings/courses/chapters/$CHAPTER_ID/studies" > "$DATA_DIR/chapters/$CHAPTER_ID.json"
    sleep 0.3
  done
done

# Step 4: Download variations for each study
echo ""
echo "[4/4] Downloading variations for each study..."
TOTAL_VARIATIONS=0
for CHAPTER_FILE in "$DATA_DIR/chapters"/*.json; do
  STUDY_IDS=$(jq -r '.[].id' "$CHAPTER_FILE" 2>/dev/null || echo "")
  for STUDY_ID in $STUDY_IDS; do
    STUDY_NAME=$(jq -r ".[] | select(.id==\"$STUDY_ID\") | .name" "$CHAPTER_FILE")
    echo "       Study: $STUDY_NAME"

    # Get variation IDs for this study
    VARIATION_IDS=$(fetch "https://cag.chessly.com/beta/openings/courses/studies/$STUDY_ID/variations")
    echo "$VARIATION_IDS" > "$DATA_DIR/studies/$STUDY_ID.json"

    # Download each variation
    mkdir -p "$DATA_DIR/variations/$STUDY_ID"
    for VAR_ID in $(echo "$VARIATION_IDS" | jq -r '.[]' 2>/dev/null || echo ""); do
      fetch "https://cag.chessly.com/beta/openings/courses/studies/variations/$VAR_ID" > "$DATA_DIR/variations/$STUDY_ID/$VAR_ID.json"
      TOTAL_VARIATIONS=$((TOTAL_VARIATIONS + 1))
      sleep 0.2
    done
    sleep 0.3
  done
done

echo ""
echo "=== Download Complete ==="
echo "Downloaded $TOTAL_VARIATIONS variations"
echo "Data saved in: $DATA_DIR"
echo ""
echo "Structure:"
echo "  data/courses.json     - All courses"
echo "  data/courses/*.json   - Chapters per course"
echo "  data/chapters/*.json  - Studies per chapter"
echo "  data/studies/*.json   - Variation IDs per study"
echo "  data/variations/*/*.json - Individual variation moves"
