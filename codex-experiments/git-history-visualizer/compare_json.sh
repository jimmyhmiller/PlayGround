#!/bin/bash

ORACLE_DIR="/var/folders/b3/821wsm8x1vgg7xqhxhqgtftc0000gn/T/git-hist-benchmark-331k9vrv/oracle"
RUST_DIR="/var/folders/b3/821wsm8x1vgg7xqhxhqgtftc0000gn/T/git-hist-benchmark-331k9vrv/rust"
OUTPUT_FILE="/Users/jimmyhmiller/Documents/Code/PlayGround/codex-experiments/git-history-visualizer/diff_output.txt"

echo "Formatting and comparing JSON files..." > "$OUTPUT_FILE"
echo "=======================================" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Get list of JSON files
JSON_FILES=($(cd "$ORACLE_DIR" && ls *.json))

for file in "${JSON_FILES[@]}"; do
    echo "Comparing $file..." >> "$OUTPUT_FILE"

    # Sort and format both files, handling arrays properly
    jq --sort-keys 'walk(if type == "array" then sort else . end)' "$ORACLE_DIR/$file" > /tmp/oracle_sorted.json
    jq --sort-keys 'walk(if type == "array" then sort else . end)' "$RUST_DIR/$file" > /tmp/rust_sorted.json

    # Compare the sorted versions
    if diff -u /tmp/oracle_sorted.json /tmp/rust_sorted.json >> "$OUTPUT_FILE" 2>&1; then
        echo "  ✓ $file: IDENTICAL" >> "$OUTPUT_FILE"
    else
        echo "  ✗ $file: DIFFERENT" >> "$OUTPUT_FILE"
    fi
    echo "" >> "$OUTPUT_FILE"
done

# Cleanup
rm -f /tmp/oracle_sorted.json /tmp/rust_sorted.json

echo "Done! Results written to $OUTPUT_FILE"
cat "$OUTPUT_FILE"
