#!/bin/bash

ORACLE_DIR="/var/folders/b3/821wsm8x1vgg7xqhxhqgtftc0000gn/T/git-hist-benchmark-331k9vrv/oracle"
RUST_DIR="/var/folders/b3/821wsm8x1vgg7xqhxhqgtftc0000gn/T/git-hist-benchmark-331k9vrv/rust"

echo "Analyzing differences in authors.json..."
echo "========================================"

# Sort and format both files
jq --sort-keys 'walk(if type == "array" then sort else . end)' "$ORACLE_DIR/authors.json" > /tmp/oracle_sorted.json
jq --sort-keys 'walk(if type == "array" then sort else . end)' "$RUST_DIR/authors.json" > /tmp/rust_sorted.json

# Show first 50 lines of differences
echo ""
echo "First 50 lines of diff:"
diff -u /tmp/oracle_sorted.json /tmp/rust_sorted.json | head -70

# Check if all numeric differences follow a pattern
echo ""
echo "Checking for systematic numeric offset..."
diff /tmp/oracle_sorted.json /tmp/rust_sorted.json | grep '^<' | head -20 | sed 's/[^0-9]/ /g' | tr -s ' ' '\n' | grep -v '^$' | head -10 > /tmp/oracle_nums.txt
diff /tmp/oracle_sorted.json /tmp/rust_sorted.json | grep '^>' | head -20 | sed 's/[^0-9]/ /g' | tr -s ' ' '\n' | grep -v '^$' | head -10 > /tmp/rust_nums.txt

echo "Sample Oracle values:"
head -10 /tmp/oracle_nums.txt
echo ""
echo "Sample Rust values:"
head -10 /tmp/rust_nums.txt

# Calculate differences if both are numeric
if [ -s /tmp/oracle_nums.txt ] && [ -s /tmp/rust_nums.txt ]; then
    echo ""
    echo "Differences (rust - oracle):"
    paste /tmp/oracle_nums.txt /tmp/rust_nums.txt | head -10 | awk '{print $2 - $1}'
fi

rm -f /tmp/oracle_sorted.json /tmp/rust_sorted.json /tmp/oracle_nums.txt /tmp/rust_nums.txt
