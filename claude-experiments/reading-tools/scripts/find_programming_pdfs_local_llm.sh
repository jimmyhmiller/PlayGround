#!/bin/bash

# Find programming-related PDFs using local LLM server at localhost:8080
# Much more accurate than keyword matching

DEST_DIR="/Users/jimmyhmiller/Documents/Code/readings"
SEARCH_DIRS=(
    "$HOME/Documents"
    "$HOME/Downloads"
    "$HOME/Desktop"
)

LLM_URL="http://localhost:8080/completion"

echo "Using local LLM server at $LLM_URL"
echo "Searching for programming-related PDFs using LLM classification..."
echo ""

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

    # Extract first 2 pages of text (enough for classification)
    text=$(pdftotext -l 2 -q "$pdf" - 2>/dev/null | head -c 1500)

    # Skip if extraction failed or text is too short
    if [ -z "$text" ] || [ ${#text} -lt 100 ]; then
        ((skipped++))
        if [ $((count % 50)) -eq 0 ]; then
            echo "... processed $count PDFs (found: $found new, skipped: $skipped, exists: $already_exists)"
        fi
        continue
    fi

    # Build prompt with few-shot examples
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

    # Call local LLM API
    response=$(curl -s "$LLM_URL" \
        -H "Content-Type: application/json" \
        -d "$(jq -n --arg prompt "$prompt" '{
            prompt: $prompt,
            temperature: 0.1,
            max_tokens: 5
        }')" 2>/dev/null | python3 -c "import sys, json; print(json.load(sys.stdin).get('content', ''))" 2>/dev/null)

    # Extract yes/no from response
    answer=$(echo "$response" | tr '[:upper:]' '[:lower:]' | grep -oE '(yes|no)' | head -1)

    if [ "$answer" = "yes" ]; then
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
echo "Total in directory now: $(ls -1 "$DEST_DIR"/*.pdf 2>/dev/null | wc -l | tr -d ' ')"
echo "Copied to: $DEST_DIR"
