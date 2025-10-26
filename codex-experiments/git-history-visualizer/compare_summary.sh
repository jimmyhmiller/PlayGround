#!/bin/bash

ORACLE_DIR="/var/folders/b3/821wsm8x1vgg7xqhxhqgtftc0000gn/T/git-hist-benchmark-331k9vrv/oracle"
RUST_DIR="/var/folders/b3/821wsm8x1vgg7xqhxhqgtftc0000gn/T/git-hist-benchmark-331k9vrv/rust"
OUTPUT_FILE="/Users/jimmyhmiller/Documents/Code/PlayGround/codex-experiments/git-history-visualizer/diff_summary.txt"

echo "JSON Comparison Summary" > "$OUTPUT_FILE"
echo "======================" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Get list of JSON files
JSON_FILES=($(cd "$ORACLE_DIR" && ls *.json))

identical_count=0
different_count=0

for file in "${JSON_FILES[@]}"; do
    # Sort and format both files, handling arrays properly
    jq --sort-keys 'walk(if type == "array" then sort else . end)' "$ORACLE_DIR/$file" > /tmp/oracle_sorted.json 2>&1
    jq --sort-keys 'walk(if type == "array" then sort else . end)' "$RUST_DIR/$file" > /tmp/rust_sorted.json 2>&1

    # Compare the sorted versions
    if diff -q /tmp/oracle_sorted.json /tmp/rust_sorted.json > /dev/null 2>&1; then
        echo "✓ $file: IDENTICAL" >> "$OUTPUT_FILE"
        ((identical_count++))
    else
        echo "✗ $file: DIFFERENT" >> "$OUTPUT_FILE"
        ((different_count++))

        # Count differences
        diff_lines=$(diff /tmp/oracle_sorted.json /tmp/rust_sorted.json | grep -c '^[<>]' || true)
        echo "  → $diff_lines lines differ" >> "$OUTPUT_FILE"
    fi
done

# Cleanup
rm -f /tmp/oracle_sorted.json /tmp/rust_sorted.json

echo "" >> "$OUTPUT_FILE"
echo "Summary:" >> "$OUTPUT_FILE"
echo "--------" >> "$OUTPUT_FILE"
echo "Identical: $identical_count" >> "$OUTPUT_FILE"
echo "Different: $different_count" >> "$OUTPUT_FILE"
echo "Total: $((identical_count + different_count))" >> "$OUTPUT_FILE"

cat "$OUTPUT_FILE"
