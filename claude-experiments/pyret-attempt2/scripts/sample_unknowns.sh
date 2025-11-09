#!/bin/bash
BASE_DIR="/Users/jimmyhmiller/Documents/Code/open-source/pyret-lang"

check_file() {
    local file="$1"
    local location="$2"
    
    # Extract line number
    line=$(echo "$location" | grep -oE "^[0-9]+" | head -1)
    
    if [ -z "$line" ]; then
        echo "  [No line number]"
        return
    fi
    
    # Show the line
    full_path="$BASE_DIR/$file"
    if [ -f "$full_path" ]; then
        content=$(sed -n "${line}p" "$full_path" 2>/dev/null | head -1)
        echo "  Line $line: $content"
    else
        echo "  [File not found]"
    fi
}

# Sample 10 unknown files
grep -A1 "^=== " bulk_test_results/failure_analysis.txt | \
  grep -B1 "Unexpected tokens after program end" | \
  grep "^===" | \
  tail -20 | \
  head -10 | \
  while IFS= read -r line; do
    file=$(echo "$line" | sed 's/^=== //' | sed 's/ ===$//')
    location=$(grep -A1 "^=== $file ===" bulk_test_results/failure_analysis.txt | tail -1 | \
               grep -oE 'location: "[^"]+' | sed 's/location: "//')
    echo "File: $file"
    check_file "$file" "$location"
    echo ""
  done
