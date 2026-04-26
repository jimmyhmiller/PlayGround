#!/bin/bash

# Find philosophy-related PDFs using local LLM with proper tool calling
# Outputs to a separate directory
# Respects .gitignore files automatically (uses fd instead of find)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROMPT_FILE="$SCRIPT_DIR/prompt_philosophy.txt"

DEST_DIR="${1:-/Users/jimmyhmiller/Documents/Code/readings/philosophy_classified}"
SEARCH_DIRS=(
    "$HOME/Documents"
    "$HOME/Downloads"
    "$HOME/Desktop"
)

LLM_URL="http://localhost:8080/v1/chat/completions"

# Check if prompt file exists
if [ ! -f "$PROMPT_FILE" ]; then
    echo "ERROR: prompt_philosophy.txt not found at $PROMPT_FILE"
    exit 1
fi

echo "Using local LLM server with tool calling"
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

# Find all PDFs (using fd which respects .gitignore)
fd --type f --extension pdf . "${SEARCH_DIRS[@]}" 2>/dev/null | \
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

    # Extract first 3 pages of text
    text=$(pdftotext -l 3 -q "$pdf" - 2>/dev/null | head -c 2000)

    # Skip if extraction failed or text is too short
    if [ -z "$text" ] || [ ${#text} -lt 100 ]; then
        ((skipped++))
        if [ $((count % 50)) -eq 0 ]; then
            echo "... processed $count PDFs (found: $found new, skipped: $skipped, exists: $already_exists)"
        fi
        continue
    fi

    # Build full prompt from prompt_philosophy.txt + document text
    prompt_base=$(cat "$PROMPT_FILE")
    full_prompt="${prompt_base}${text}"

    # Escape for JSON
    escaped_prompt=$(echo "$full_prompt" | python3 -c "import sys, json; print(json.dumps(sys.stdin.read()))")

    # Call LLM with tool calling
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
                    "description": "True if a philosophy research paper, journal article, monograph, book, lecture, essay, or primary source. False for resumes, invoices, flyers, manuals, ads, programming/CS material, empirical-science papers without philosophical content, or other non-philosophy documents."
                  }
                },
                "required": ["is_philosophy"]
              }
            }
          }],
          "tool_choice": {"type": "function", "function": {"name": "classify_pdf"}},
          "max_tokens": 500
        }' 2>/dev/null)

    # Extract tool call arguments
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

    if [ "$is_philosophy" = "true" ]; then
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
echo "Found: $found NEW philosophy-related PDFs"
echo "Skipped: $skipped (not philosophy-related or no text)"
echo "Already existed: $already_exists"
echo "Total in output directory: $(ls -1 "$DEST_DIR"/*.pdf 2>/dev/null | wc -l | tr -d ' ')"
echo "Output directory: $DEST_DIR"
