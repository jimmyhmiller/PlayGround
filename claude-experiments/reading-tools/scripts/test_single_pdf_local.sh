#!/bin/bash

# Test local LLM classification on a single PDF with full output

LLM_URL="http://localhost:8080/completion"
PDF_PATH="$1"

if [ -z "$PDF_PATH" ]; then
    echo "Usage: ./test_single_pdf_local.sh <path_to_pdf>"
    echo ""
    echo "Example: ./test_single_pdf_local.sh \"IASHA Teeshirts flyer.pdf\""
    exit 1
fi

if [ ! -f "$PDF_PATH" ]; then
    echo "Error: File not found: $PDF_PATH"
    exit 1
fi

filename=$(basename "$PDF_PATH")

echo "=========================================="
echo "Testing PDF: $filename"
echo "Using: Local LLM at localhost:8080"
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

# Create prompt with few-shot examples
prompt="You are a PDF classifier. Determine if a document is a computer science research paper, technical article, or book.

EXAMPLES:

Text: \"CHAPTER 41 Is the whole universe a computer? The theory that the whole universe is a computer... Alan Turing introduced in 1936...\"
Answer: yes

Text: \"Abstract. We introduce return-oriented programming, a technique by which an attacker can induce arbitrary behavior in a program...\"
Answer: yes

Text: \"Resume: John Smith. Skills: Python, Java, C++. Experience: Software Engineer at Google...\"
Answer: no

Text: \"Invoice #12345. Amount due: \$500. Please pay by...\"
Answer: no

Text: \"T-SHIRT ORDER FORM. Sizes available: S, M, L, XL. Price: \$20...\"
Answer: no

Text: \"X-Plane Flight Manual. Chapter 1: Introduction to Aviation. This manual covers the operation of...\"
Answer: no

NOW CLASSIFY THIS:

Text excerpt:
\"\"\"
$text
\"\"\"

Answer ONLY with 'yes' or 'no':
"

# Call local LLM
echo "Calling LLM..."
full_response=$(curl -s "$LLM_URL" \
    -H "Content-Type: application/json" \
    -d "$(jq -n --arg prompt "$prompt" '{
        prompt: $prompt,
        temperature: 0.0,
        max_tokens: 20
    }')" 2>/dev/null | python3 -c "import sys, json; print(json.load(sys.stdin).get('content', ''))" 2>/dev/null)

echo "Full LLM response:"
echo "----------------------------------------"
echo "$full_response"
echo ""

# Extract yes/no
response=$(echo "$full_response" | tr '[:upper:]' '[:lower:]' | grep -oE '(yes|no)' | head -1)

echo "=========================================="
echo "Extracted answer: $response"
echo "=========================================="

if [ "$response" = "yes" ]; then
    echo "✓ This PDF would be COPIED (classified as programming-related)"
elif [ "$response" = "no" ]; then
    echo "✗ This PDF would be SKIPPED (not programming-related)"
else
    echo "? Unable to extract clear yes/no from response"
fi
