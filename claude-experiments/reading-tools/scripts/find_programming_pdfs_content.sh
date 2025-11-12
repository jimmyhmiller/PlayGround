#!/bin/bash

# Find programming-related PDFs by extracting and analyzing actual content
# This catches PDFs with sparse metadata

DEST_DIR="/Users/jimmyhmiller/Documents/Code/readings"
SEARCH_DIRS=(
    "$HOME/Documents"
    "$HOME/Downloads"
    "$HOME/Desktop"
)

# Programming-related keywords (case-insensitive)
KEYWORDS=(
    "programming" "computer" "software" "algorithm" "code" "coding"
    "compiler" "language" "type" "typed" "functional" "object"
    "computation" "computing" "lambda" "syntax" "semantics"
    "parser" "interpreter" "runtime" "virtual machine"
    "debug" "test" "development" "developer" "engineering"
    "java" "python" "javascript" "haskell" "lisp" "scheme" "rust"
    "turing" "calculus" "abstract" "data structure"
    "network" "protocol" "api" "framework" "library"
    "variable" "function" "method" "class" "interface"
    "memory" "cpu" "processor" "execution" "compile"
    "recursion" "iteration" "loop" "conditional"
    "boolean" "integer" "string" "array" "list"
    "parallelism" "concurrent" "thread" "process"
    "neural network" "machine learning" "artificial intelligence"
)

# Build grep pattern
PATTERN=$(IFS='|'; echo "${KEYWORDS[*]}")

echo "Searching for programming-related PDFs using content extraction..."
echo "Extracting first 3 pages of each PDF for analysis..."
echo ""

count=0
found=0
already_exists=0

# Find all PDFs
find "${SEARCH_DIRS[@]}" -name "*.pdf" -type f 2>/dev/null | \
    grep -v "/open-source/" | \
    grep -v "/.pub-cache/" | \
    grep -v "/Library/" | \
    grep -v "/node_modules/" | \
    grep -v "/.git/" | \
while read -r pdf; do
    ((count++))

    # Show progress every 50 PDFs
    if [ $((count % 50)) -eq 0 ]; then
        echo "... processed $count PDFs so far (found: $found new, skipped: $already_exists existing)"
    fi

    filename=$(basename "$pdf")

    # Skip if already exists
    if [ -f "$DEST_DIR/$filename" ]; then
        ((already_exists++))
        continue
    fi

    # Extract first 3 pages of text (fast)
    # Use -l 3 to limit to first 3 pages, -q for quiet mode
    text=$(pdftotext -l 3 -q "$pdf" - 2>/dev/null)

    # Check if extraction succeeded and text contains keywords
    if [ -n "$text" ] && echo "$text" | grep -iE "$PATTERN" > /dev/null; then
        echo "✓ Found: $filename"
        echo "  Path: $pdf"
        cp "$pdf" "$DEST_DIR/" && echo "  → Copied" && ((found++))
        echo ""
    fi
done

echo "========================================"
echo "Scanned: $count PDFs"
echo "Found: $found NEW programming-related PDFs"
echo "Skipped: $already_exists (already existed)"
echo "Total in directory now: $(ls -1 "$DEST_DIR"/*.pdf 2>/dev/null | wc -l | tr -d ' ')"
echo "Copied to: $DEST_DIR"
