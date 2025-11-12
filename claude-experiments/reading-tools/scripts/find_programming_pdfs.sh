#!/bin/bash

# Find all PDFs and check if they're programming-related based on metadata
# Then copy them to the current directory

DEST_DIR="/Users/jimmyhmiller/Documents/Code/readings"
SEARCH_DIRS=(
    "$HOME/Documents"
    "$HOME/Downloads"
    "$HOME/Desktop"
)

# Programming-related keywords to search for (case-insensitive)
KEYWORDS=(
    "programming" "computer" "software" "algorithm" "code" "coding"
    "compiler" "language" "type" "typed" "functional" "object"
    "computation" "computing" "lambda" "syntax" "semantics"
    "parser" "interpreter" "runtime" "virtual machine" "vm"
    "debug" "test" "development" "developer" "engineering"
    "java" "python" "javascript" "haskell" "lisp" "scheme"
    "turing" "calculus" "abstract" "data structure" "graph"
    "network" "protocol" "api" "framework" "library"
)

# Build grep pattern from keywords
PATTERN=$(IFS='|'; echo "${KEYWORDS[*]}")

echo "Searching for programming-related PDFs..."
echo "This may take a few minutes..."
echo ""

count=0
found=0

# Find all PDFs (excluding common cache/library directories)
find "${SEARCH_DIRS[@]}" -name "*.pdf" -type f 2>/dev/null | \
    grep -v "/open-source/" | \
    grep -v "/.pub-cache/" | \
    grep -v "/Library/" | \
    grep -v "/node_modules/" | \
    grep -v "/.git/" | \
while read -r pdf; do
    ((count++))

    # Extract metadata
    metadata=$(mdls -name kMDItemTitle -name kMDItemKeywords -name kMDItemSubject -name kMDItemAuthors -name kMDItemTextContent "$pdf" 2>/dev/null)

    # Check if metadata contains programming keywords (case-insensitive)
    if echo "$metadata" | grep -iE "$PATTERN" > /dev/null; then
        filename=$(basename "$pdf")
        echo "✓ Found: $filename"
        echo "  Path: $pdf"

        # Copy to destination (handle filename conflicts)
        if [ -f "$DEST_DIR/$filename" ]; then
            echo "  → Already exists, skipping"
        else
            cp "$pdf" "$DEST_DIR/" && echo "  → Copied"
            ((found++))
        fi
        echo ""
    fi
done

echo "----------------------------------------"
echo "Scanned: $count PDFs"
echo "Found: $found programming-related PDFs"
echo "Copied to: $DEST_DIR"
