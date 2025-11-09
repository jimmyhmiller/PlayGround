#!/bin/bash
BASE_DIR="/Users/jimmyhmiller/Documents/Code/open-source/pyret-lang"

echo "Testing files marked as SUCCESS to verify they match official parser..."
echo ""

count=0
matches=0
mismatches=0

grep "SUCCESS - Parses correctly now!" bulk_test_results/failing_files.txt | while read -r line; do
    file=$(echo "$line" | sed 's/ *#.*//')
    full_path="$BASE_DIR/$file"
    
    if [ ! -f "$full_path" ]; then
        echo "❌ SKIP: $file (not found)"
        continue
    fi
    
    count=$((count + 1))
    
    # Run comparison
    result=$(./compare_parsers.sh "$full_path" 2>&1)
    
    if echo "$result" | grep -q "✅ ASTs are identical"; then
        echo "✅ MATCH: $file"
        matches=$((matches + 1))
    else
        echo "❌ MISMATCH: $file"
        # Show first difference
        echo "$result" | grep -A2 "First difference" | head -3
        mismatches=$((mismatches + 1))
    fi
    echo ""
done

echo "================================="
echo "Summary:"
echo "  Matches: $matches"
echo "  Mismatches: $mismatches"
echo "================================="
