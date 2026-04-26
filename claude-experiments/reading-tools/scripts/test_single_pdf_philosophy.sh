#!/bin/bash

# Test local LLM philosophy classification with tool calling

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROMPT_FILE="$SCRIPT_DIR/prompt_philosophy.txt"

LLM_URL="http://localhost:8080/v1/chat/completions"
PDF_PATH="$1"

if [ ! -f "$PROMPT_FILE" ]; then
    echo "ERROR: prompt_philosophy.txt not found at $PROMPT_FILE"
    exit 1
fi

if [ -z "$PDF_PATH" ]; then
    echo "Usage: ./test_single_pdf_philosophy.sh <path_to_pdf>"
    exit 1
fi

if [ ! -f "$PDF_PATH" ]; then
    echo "Error: File not found: $PDF_PATH"
    exit 1
fi

filename=$(basename "$PDF_PATH")

echo "=========================================="
echo "Testing PDF: $filename"
echo "=========================================="
echo ""

if ! curl -s http://localhost:8080/health >/dev/null 2>&1; then
    echo "ERROR: LLM server at localhost:8080 is not responding"
    exit 1
fi

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

prompt_base=$(cat "$PROMPT_FILE")
full_prompt="${prompt_base}${text}"
escaped_prompt=$(echo "$full_prompt" | python3 -c "import sys, json; print(json.dumps(sys.stdin.read()))")

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
          "description": "Classify if document is a work of philosophy",
          "parameters": {
            "type": "object",
            "properties": {
              "is_philosophy": {
                "type": "boolean",
                "description": "True if a philosophy paper, article, monograph, book, lecture, essay, or primary source. False otherwise."
              }
            },
            "required": ["is_philosophy"]
          }
        }
      }],
      "tool_choice": {"type": "function", "function": {"name": "classify_pdf"}},
      "max_tokens": 500
    }' 2>/dev/null)

result=$(echo "$response" | python3 -c "
import sys, json
try:
    r = json.load(sys.stdin)
    msg = r.get('choices', [{}])[0].get('message', {})
    reasoning = msg.get('reasoning_content', '')
    if reasoning:
        print('Reasoning:', reasoning)
        print()
    tool_calls = msg.get('tool_calls', [])
    if tool_calls:
        args = json.loads(tool_calls[0]['function']['arguments'])
        print('Tool call arguments:', json.dumps(args, indent=2))
        print()
        print('is_philosophy =', args.get('is_philosophy', False))
    else:
        print('No tool call found')
except Exception as e:
    print('Error:', e)
" 2>/dev/null)

echo "$result"
echo ""

is_philosophy=$(echo "$response" | python3 -c "
import sys, json
try:
    r = json.load(sys.stdin)
    tool_calls = r.get('choices', [{}])[0].get('message', {}).get('tool_calls', [])
    if tool_calls:
        args = json.loads(tool_calls[0]['function']['arguments'])
        print(str(args.get('is_philosophy', False)).lower())
    else:
        print('false')
except:
    print('false')
" 2>/dev/null)

echo "=========================================="
if [ "$is_philosophy" = "true" ]; then
    echo "✓ This PDF would be COPIED (classified as philosophy)"
else
    echo "✗ This PDF would be SKIPPED (not philosophy)"
fi
echo "=========================================="
