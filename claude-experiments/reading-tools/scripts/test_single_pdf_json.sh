#!/bin/bash

# Test local LLM classification with JSON tool calling

LLM_URL="http://localhost:8080/completion"
PDF_PATH="$1"

# JSON grammar
JSON_GRAMMAR='root ::= "{" space "\"is_programming\"" space ":" space boolean space "}" space
boolean ::= "true" | "false"
space ::= " "*'

if [ -z "$PDF_PATH" ]; then
    echo "Usage: ./test_single_pdf_json.sh <path_to_pdf>"
    echo ""
    echo "Example: ./test_single_pdf_json.sh \"IASHA Teeshirts flyer.pdf\""
    exit 1
fi

if [ ! -f "$PDF_PATH" ]; then
    echo "Error: File not found: $PDF_PATH"
    exit 1
fi

filename=$(basename "$PDF_PATH")

echo "=========================================="
echo "Testing PDF: $filename"
echo "Using: Local LLM with JSON grammar"
echo "=========================================="
echo ""

# Test server
if ! curl -s http://localhost:8080/health >/dev/null 2>&1; then
    echo "ERROR: LLM server at localhost:8080 is not responding"
    exit 1
fi

echo "✓ LLM server is online"
echo ""

# Extract text
echo "Extracting first 2 pages..."
text=$(pdftotext -l 2 -q "$PDF_PATH" - 2>/dev/null | head -c 1500)

if [ -z "$text" ]; then
    echo "ERROR: Could not extract text from PDF"
    exit 1
fi

echo "Extracted text (first 500 chars):"
echo "----------------------------------------"
echo "$text" | head -c 500
echo ""
echo "... (truncated)"
echo ""
echo "=========================================="
echo "Asking LLM to classify..."
echo "=========================================="
echo ""

# Build prompt - very simple and direct
prompt="Read this document excerpt:

$text

Question: Is this a computer science or programming research paper/article/book? (NOT a resume, invoice, flyer, manual, or personal document)

Answer in JSON: {\"is_programming\": true/false}

JSON:"

# Call local LLM
echo "Calling LLM with JSON grammar constraint..."
response=$(curl -s "$LLM_URL" \
    -H "Content-Type: application/json" \
    -d "$(jq -n --arg prompt "$prompt" --arg grammar "$JSON_GRAMMAR" '{
        prompt: $prompt,
        temperature: 0.3,
        max_tokens: 50,
        grammar: $grammar
    }')" 2>/dev/null)

echo ""
echo "Raw response:"
echo "----------------------------------------"
echo "$response" | python3 -c "import sys, json; print(json.dumps(json.load(sys.stdin), indent=2))" 2>/dev/null
echo ""

# Extract the JSON content
content=$(echo "$response" | python3 -c "import sys, json; print(json.load(sys.stdin).get('content', ''))" 2>/dev/null)

echo "LLM JSON output:"
echo "----------------------------------------"
echo "$content"
echo ""

# Parse the is_programming field
is_programming=$(echo "$content" | python3 -c "import sys, json; print(json.loads(sys.stdin.read()).get('is_programming', 'false'))" 2>/dev/null)

echo "=========================================="
echo "Extracted value: is_programming = $is_programming"
echo "=========================================="

if [ "$is_programming" = "True" ] || [ "$is_programming" = "true" ]; then
    echo "✓ This PDF would be COPIED (classified as programming-related)"
elif [ "$is_programming" = "False" ] || [ "$is_programming" = "false" ]; then
    echo "✗ This PDF would be SKIPPED (not programming-related)"
else
    echo "? Unable to parse response"
fi
