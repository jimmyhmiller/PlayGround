#!/bin/bash
BASE_DIR="/Users/jimmyhmiller/Documents/Code/open-source/pyret-lang"
OUTPUT_FILE="bulk_test_results/failure_analysis.txt"
> "$OUTPUT_FILE"

while IFS= read -r file; do
    if [ -z "$file" ]; then continue; fi
    
    echo "=== $file ===" >> "$OUTPUT_FILE"
    
    full_path="$BASE_DIR/$file"
    if [ ! -f "$full_path" ]; then
        echo "ERROR: File not found" >> "$OUTPUT_FILE"
        continue
    fi
    
    # Run parser and capture error
    error=$(cargo run --quiet --bin to_pyret_json "$full_path" 2>&1 | grep -v "warning:" | grep -E "(Error|Expected|Unexpected|failed)" | head -1)
    
    if [ -z "$error" ]; then
        echo "SUCCESS: File parses correctly" >> "$OUTPUT_FILE"
    else
        echo "$error" >> "$OUTPUT_FILE"
    fi
    echo "" >> "$OUTPUT_FILE"
done < bulk_test_results/failing_files.txt

echo "Analysis complete. Results in $OUTPUT_FILE"
