#!/bin/bash
BASE_DIR="/Users/jimmyhmiller/Documents/Code/open-source/pyret-lang"
TEMP_DIR="/tmp/pyret_verification"
mkdir -p "$TEMP_DIR"

echo "Verifying SUCCESS files against official parser..."
echo ""

count=0
verified=0
mismatch=0

# Create a temporary file for updates
temp_output=$(mktemp)

while IFS= read -r line; do
    # Check if this is a SUCCESS line
    if ! echo "$line" | grep -q "SUCCESS (verified)"; then
        echo "$line" >> "$temp_output"
        continue
    fi
    
    file=$(echo "$line" | sed 's/ *#.*//')
    full_path="$BASE_DIR/$file"
    
    if [ ! -f "$full_path" ]; then
        echo "$line" >> "$temp_output"
        continue
    fi
    
    count=$((count + 1))
    
    # Run the comparison script
    result=$(./compare_parsers.sh "$full_path" 2>&1)
    
    if echo "$result" | grep -q "✅ ASTs are identical"; then
        echo "✅ VERIFIED: $file"
        echo "$file  # ✅ MATCHES official parser" >> "$temp_output"
        verified=$((verified + 1))
    else
        echo "❌ MISMATCH: $file"
        # Extract what the difference is
        diff_info=$(echo "$result" | grep -A1 "First difference" | tail -1 | head -c 60)
        echo "$file  # ❌ PARSES but doesn't match official ($diff_info...)" >> "$temp_output"
        mismatch=$((mismatch + 1))
    fi
    
done < bulk_test_results/failing_files.txt

# Replace the original file
mv "$temp_output" bulk_test_results/failing_files.txt

echo ""
echo "================================="
echo "Summary:"
echo "  Verified matches: $verified"
echo "  Mismatches: $mismatch"
echo "  Total tested: $count"
echo "================================="
echo ""
echo "File updated: bulk_test_results/failing_files.txt"
