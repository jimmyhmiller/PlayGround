#!/bin/bash

# Find programming-related PDFs using local LLM with JSON tool calling
# Outputs to a separate directory to avoid mixing with existing PDFs

DEST_DIR="${1:-/Users/jimmyhmiller/Documents/Code/readings/llm_classified}"
SEARCH_DIRS=(
    "$HOME/Documents"
    "$HOME/Downloads"
    "$HOME/Desktop"
)

LLM_URL="http://localhost:8080/completion"

# JSON grammar to force structured output
JSON_GRAMMAR='root ::= "{" space "\"is_programming\"" space ":" space boolean space "}" space
boolean ::= "true" | "false"
space ::= " "*'

echo "Using local LLM server at $LLM_URL"
echo "Output directory: $DEST_DIR"
echo ""

# Create output directory
mkdir -p "$DEST_DIR"

# Test if server is available
if ! curl -s http://localhost:8080/health >/dev/null 2>&1; then
    echo "ERROR: LLM server at localhost:8080 is not responding"
    exit 1
fi

echo "✓ LLM server is online"
echo ""

count=0
found=0
already_exists=0
skipped=0

# Find all PDFs
find "${SEARCH_DIRS[@]}" -name "*.pdf" -type f 2>/dev/null | \
    grep -v "/open-source/" | \
    grep -v "/.pub-cache/" | \
    grep -v "/Library/" | \
    grep -v "/node_modules/" | \
    grep -v "/.git/" | \
while read -r pdf; do
    ((count++))

    filename=$(basename "$pdf")

    # Skip if already exists
    if [ -f "$DEST_DIR/$filename" ]; then
        ((already_exists++))
        if [ $((count % 50)) -eq 0 ]; then
            echo "... processed $count PDFs (found: $found new, skipped: $skipped, exists: $already_exists)"
        fi
        continue
    fi

    # Extract first 2 pages of text
    text=$(pdftotext -l 2 -q "$pdf" - 2>/dev/null | head -c 1500)

    # Skip if extraction failed or text is too short
    if [ -z "$text" ] || [ ${#text} -lt 100 ]; then
        ((skipped++))
        if [ $((count % 50)) -eq 0 ]; then
            echo "... processed $count PDFs (found: $found new, skipped: $skipped, exists: $already_exists)"
        fi
        continue
    fi

    # Build prompt - very simple and direct
    prompt="Read this document excerpt:

$text

Question: Is this a computer science or programming research paper/article/book? (NOT a resume, invoice, flyer, manual, or personal document)

Answer in JSON: {\"is_programming\": true/false}

JSON:"

    # Call local LLM with JSON grammar
    response=$(curl -s "$LLM_URL" \
        -H "Content-Type: application/json" \
        -d "$(jq -n --arg prompt "$prompt" --arg grammar "$JSON_GRAMMAR" '{
            prompt: $prompt,
            temperature: 0.3,
            max_tokens: 50,
            grammar: $grammar
        }')" 2>/dev/null)

    # Extract the JSON content
    content=$(echo "$response" | python3 -c "import sys, json; print(json.load(sys.stdin).get('content', ''))" 2>/dev/null)

    # Parse the is_programming field
    is_programming=$(echo "$content" | python3 -c "import sys, json; print(json.loads(sys.stdin.read()).get('is_programming', 'false'))" 2>/dev/null)

    if [ "$is_programming" = "True" ] || [ "$is_programming" = "true" ]; then
        echo "✓ Found: $filename"
        echo "  Path: $pdf"
        cp "$pdf" "$DEST_DIR/" && echo "  → Copied" && ((found++))
        echo ""
    else
        ((skipped++))
    fi

    # Progress update
    if [ $((count % 10)) -eq 0 ]; then
        echo "... processed $count PDFs (found: $found new, skipped: $skipped, exists: $already_exists)"
    fi
done

echo ""
echo "========================================"
echo "Scanned: $count PDFs"
echo "Found: $found NEW programming-related PDFs"
echo "Skipped: $skipped (not programming-related or no text)"
echo "Already existed: $already_exists"
echo "Total in output directory: $(ls -1 "$DEST_DIR"/*.pdf 2>/dev/null | wc -l | tr -d ' ')"
echo "Output directory: $DEST_DIR"
