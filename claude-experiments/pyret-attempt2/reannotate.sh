#!/bin/bash
# Reannotate all files in bulk_test_results/failing_files.txt
# Run this after making parser changes to update the annotations

set -e

echo "🔄 Re-running bulk tests and updating annotations..."
echo ""

# Step 1: Re-run the parser on all files to get fresh error data
echo "Step 1: Testing all files with current parser..."
BASE_DIR="/Users/jimmyhmiller/Documents/Code/open-source/pyret-lang"
OUTPUT_FILE="bulk_test_results/failure_analysis.txt"
> "$OUTPUT_FILE"

file_count=0
while IFS= read -r line; do
    # Extract filename (before the #)
    file=$(echo "$line" | sed 's/ *#.*//')
    
    if [ -z "$file" ]; then continue; fi
    
    file_count=$((file_count + 1))
    if [ $((file_count % 50)) -eq 0 ]; then
        echo "  Tested $file_count files..."
    fi
    
    echo "=== $file ===" >> "$OUTPUT_FILE"
    
    full_path="$BASE_DIR/$file"
    if [ ! -f "$full_path" ]; then
        echo "ERROR: File not found" >> "$OUTPUT_FILE"
        continue
    fi
    
    # Run parser and capture error
    error=$(cargo run --quiet --bin to_pyret_json "$full_path" 2>&1 | \
            grep -v "warning:" | \
            grep -E "(Error|Expected|Unexpected|failed)" | \
            head -1)
    
    if [ -z "$error" ]; then
        echo "SUCCESS: File parses correctly" >> "$OUTPUT_FILE"
    else
        echo "$error" >> "$OUTPUT_FILE"
    fi
    echo "" >> "$OUTPUT_FILE"
done < bulk_test_results/failing_files.txt

echo "  Tested $file_count files total."
echo ""

# Step 2: Categorize parse errors
echo "Step 2: Categorizing parse errors..."
python3 final_accurate_categorize.py 2>/dev/null
echo ""

# Step 3: Check files that parse for AST differences
echo "Step 3: Checking AST differences for files that parse..."
python3 categorize_mismatches.py 2>/dev/null
echo ""

echo "✅ Re-annotation complete!"
echo ""
echo "Updated file: bulk_test_results/failing_files.txt"
echo ""
echo "To see summary, run:"
echo "  python3 print_summary.py"
