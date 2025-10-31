#!/bin/bash

# Script to parse all .arr files in the pyret-lang directory
PYRET_DIR="/Users/jimmyhmiller/Documents/Code/open-source/pyret-lang"
PARSER="./target/release/pyret-attempt"
OUTPUT_DIR="./parse_results"
LOG_FILE="$OUTPUT_DIR/parse_log.txt"
SUCCESS_FILE="$OUTPUT_DIR/successful.txt"
FAILED_FILE="$OUTPUT_DIR/failed.txt"
SUMMARY_FILE="$OUTPUT_DIR/summary.txt"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Clear previous results
> "$LOG_FILE"
> "$SUCCESS_FILE"
> "$FAILED_FILE"

echo "Starting to parse all .arr files in $PYRET_DIR"
echo "Results will be saved to $OUTPUT_DIR"
echo ""

total=0
success=0
failed=0

# Find all .arr files and parse them
while IFS= read -r file; do
    ((total++))
    echo "[$total] Parsing: $file" | tee -a "$LOG_FILE"

    if $PARSER "$file" >> "$LOG_FILE" 2>&1; then
        ((success++))
        echo "  ✓ Success" | tee -a "$LOG_FILE"
        echo "$file" >> "$SUCCESS_FILE"
    else
        ((failed++))
        echo "  ✗ Failed" | tee -a "$LOG_FILE"
        echo "$file" >> "$FAILED_FILE"
    fi

    # Show progress every 50 files
    if ((total % 50 == 0)); then
        echo ""
        echo "Progress: $total files processed ($success successful, $failed failed)"
        echo ""
    fi
done < <(find "$PYRET_DIR" -name "*.arr" -type f | sort)

# Generate summary
echo "" | tee -a "$LOG_FILE"
echo "================================" | tee -a "$LOG_FILE"
echo "PARSING COMPLETE" | tee -a "$LOG_FILE"
echo "================================" | tee -a "$LOG_FILE"
echo "Total files: $total" | tee -a "$LOG_FILE"
echo "Successful:  $success ($(awk "BEGIN {printf \"%.1f\", ($success/$total)*100}")%)" | tee -a "$LOG_FILE"
echo "Failed:      $failed ($(awk "BEGIN {printf \"%.1f\", ($failed/$total)*100}")%)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Save summary to file
cat > "$SUMMARY_FILE" <<EOF
Pyret Parser Results
====================
Date: $(date)
Total files: $total
Successful:  $success ($(awk "BEGIN {printf \"%.1f\", ($success/$total)*100}")%)
Failed:      $failed ($(awk "BEGIN {printf \"%.1f\", ($failed/$total)*100}")%)

Detailed logs: $LOG_FILE
Successful files: $SUCCESS_FILE
Failed files: $FAILED_FILE
EOF

echo "Summary saved to: $SUMMARY_FILE"
echo "Full log saved to: $LOG_FILE"
echo "Successful files list: $SUCCESS_FILE"
echo "Failed files list: $FAILED_FILE"
