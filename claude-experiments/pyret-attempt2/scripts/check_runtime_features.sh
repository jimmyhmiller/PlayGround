#!/bin/bash
BASE_DIR="/Users/jimmyhmiller/Documents/Code/open-source/pyret-lang"

echo "Checking 'Runtime feature' files..."
echo ""

grep "Runtime feature" bulk_test_results/failing_files.txt | head -10 | while read -r line; do
    file=$(echo "$line" | sed 's/ *#.*//')
    
    # Get error location
    location=$(grep -A1 "^=== $file ===" bulk_test_results/failure_analysis.txt | tail -1 | \
               grep -oE 'location: "[^"]+' | sed 's/location: "//' | head -1)
    
    line_num=$(echo "$location" | grep -oE "^[0-9]+" | head -1)
    
    full_path="$BASE_DIR/$file"
    
    echo "File: $file"
    echo "Error at line: $line_num"
    
    if [ -f "$full_path" ] && [ -n "$line_num" ]; then
        # Show context: 2 lines before and 2 after
        start=$((line_num - 2))
        end=$((line_num + 2))
        if [ $start -lt 1 ]; then start=1; fi
        
        sed -n "${start},${end}p" "$full_path" | nl -v $start -ba | sed "s/^  *${line_num}/>>>/"
    fi
    echo ""
done
