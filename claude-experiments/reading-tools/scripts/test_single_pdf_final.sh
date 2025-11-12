#!/bin/bash

# Test local LLM classification with tool calling

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROMPT_FILE="$SCRIPT_DIR/prompt.txt"

LLM_URL="http://localhost:8080/v1/chat/completions"
PDF_PATH="$1"

# Check if prompt file exists
if [ ! -f "$PROMPT_FILE" ]; then
    echo "ERROR: prompt.txt not found at $PROMPT_FILE"
    exit 1
fi

if [ -z "$PDF_PATH" ]; then
    echo "Usage: ./test_single_pdf_final.sh <path_to_pdf>"
    echo ""
    echo "Example: ./test_single_pdf_final.sh \"IASHA Teeshirts flyer.pdf\""
    exit 1
fi

if [ ! -f "$PDF_PATH" ]; then
    echo "Error: File not found: $PDF_PATH"
    exit 1
fi

filename=$(basename "$PDF_PATH")

echo "=========================================="
echo "Testing PDF: $filename"
echo "Using: OpenAI-compatible tool calling API"
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
echo "Extracting first 3 pages..."
text=$(pdftotext -l 3 -q "$PDF_PATH" - 2>/dev/null | head -c 2000)

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
echo "Loading prompt from prompt.txt..."
echo "=========================================="
echo ""

# Build full prompt from prompt.txt + document text
prompt_base=$(cat "$PROMPT_FILE")
full_prompt="${prompt_base}${text}"

echo "Prompt template (first 300 chars):"
echo "----------------------------------------"
echo "$prompt_base" | head -c 300
echo ""
echo "... (plus extracted document text)"
echo ""

# Escape for JSON
escaped_prompt=$(echo "$full_prompt" | python3 -c "import sys, json; print(json.dumps(sys.stdin.read()))")

echo "=========================================="
echo "Calling LLM with tool calling..."
echo "=========================================="
echo ""

# Call LLM
response=$(curl -s "$LLM_URL" \
    -H "Content-Type: application/json" \
    -d '{
      "messages": [{
        "role": "user",
        "content": '"$escaped_prompt"'
      }],
      "tools": [{
        "type": "function",
        "function": {
          "name": "classify_pdf",
          "description": "Classify if document is CS/programming related",
          "parameters": {
            "type": "object",
            "properties": {
              "is_programming": {
                "type": "boolean",
                "description": "True if CS/programming research paper/article/book. False for resumes, invoices, flyers, manuals, ads."
              }
            },
            "required": ["is_programming"]
          }
        }
      }],
      "tool_choice": {"type": "function", "function": {"name": "classify_pdf"}},
      "max_tokens": 500
    }' 2>/dev/null)

echo "Full response:"
echo "----------------------------------------"
echo "$response" | python3 -c "import sys, json; print(json.dumps(json.load(sys.stdin), indent=2))" 2>/dev/null
echo ""

# Extract tool call
echo "=========================================="
echo "Extracting tool call..."
echo "=========================================="
echo ""

result=$(echo "$response" | python3 -c "
import sys, json
try:
    r = json.load(sys.stdin)
    msg = r.get('choices', [{}])[0].get('message', {})

    # Show reasoning if available
    reasoning = msg.get('reasoning_content', '')
    if reasoning:
        print('Reasoning:', reasoning)
        print()

    # Get tool call
    tool_calls = msg.get('tool_calls', [])
    if tool_calls:
        args = json.loads(tool_calls[0]['function']['arguments'])
        is_prog = args.get('is_programming', False)
        print('Tool call arguments:', json.dumps(args, indent=2))
        print()
        print('is_programming =', is_prog)
    else:
        print('No tool call found')
except Exception as e:
    print('Error:', e)
" 2>/dev/null)

echo "$result"
echo ""

# Final verdict
is_programming=$(echo "$response" | python3 -c "
import sys, json
try:
    r = json.load(sys.stdin)
    tool_calls = r.get('choices', [{}])[0].get('message', {}).get('tool_calls', [])
    if tool_calls:
        args = json.loads(tool_calls[0]['function']['arguments'])
        print(str(args.get('is_programming', False)).lower())
    else:
        print('false')
except:
    print('false')
" 2>/dev/null)

echo "=========================================="
if [ "$is_programming" = "true" ]; then
    echo "✓ This PDF would be COPIED (classified as programming-related)"
else
    echo "✗ This PDF would be SKIPPED (not programming-related)"
fi
echo "=========================================="
