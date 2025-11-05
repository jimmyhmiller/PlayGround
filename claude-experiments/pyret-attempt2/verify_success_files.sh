#!/bin/bash
BASE_DIR="/Users/jimmyhmiller/Documents/Code/open-source/pyret-lang"

echo "Verifying files marked as SUCCESS..."
echo ""

actually_success=0
actually_fail=0
crash=0

grep "SUCCESS - Parses correctly now!" bulk_test_results/failing_files.txt | head -20 | while read -r line; do
    file=$(echo "$line" | sed 's/ *#.*//')
    full_path="$BASE_DIR/$file"
    
    if [ ! -f "$full_path" ]; then
        continue
    fi
    
    # Try to parse with our parser
    output=$(cargo run --quiet --bin to_pyret_json "$full_path" 2>&1)
    exit_code=$?
    
    if [ $exit_code -ne 0 ]; then
        # Check if it's a crash or parse error
        if echo "$output" | grep -q "Abort trap\|signal:\|SIGABRT"; then
            echo "💥 CRASH: $file (stack overflow or panic)"
            crash=$((crash + 1))
        else
            echo "❌ PARSE ERROR: $file"
            echo "$output" | grep -i "error" | head -1
            actually_fail=$((actually_fail + 1))
        fi
    else
        echo "✅ PARSES: $file"
        actually_success=$((actually_success + 1))
    fi
done
