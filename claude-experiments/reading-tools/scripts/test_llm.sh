#!/bin/bash

# Test LLM classification on a few PDFs

MODEL="qwen2.5:0.5b"

echo "Testing LLM classification with $MODEL"
echo ""

# Test on a programming PDF
PDF="CH41_Copeland_Shagrir_Sprevak_by_MarkSprevak.pdf"
if [ -f "$PDF" ]; then
    echo "Testing: $PDF (should be YES)"
    text=$(pdftotext -l 2 -q "$PDF" - 2>/dev/null | head -c 2000)

    prompt="You are a PDF classifier. Analyze this text excerpt and determine if it's a RESEARCH PAPER, TECHNICAL ARTICLE, or BOOK about computer science, programming, software engineering, algorithms, or computation theory.

EXCLUDE: resumes, CVs, invoices, tax forms, flight manuals, aviation documents, personal documents, legal documents, contracts, receipts.

Text excerpt:
\"\"\"
$text
\"\"\"

Is this a programming/CS research paper, article, or book? Answer ONLY with 'yes' or 'no'."

    response=$(echo "$prompt" | ollama run "$MODEL" 2>/dev/null | tr '[:upper:]' '[:lower:]' | grep -oE '^(yes|no)' | head -1)
    echo "Response: $response"
    echo ""
fi

# Test on a personal document (if we can find one)
PDF="JimmyMillerResume.pdf"
if [ -f "$PDF" ]; then
    echo "Testing: $PDF (should be NO - it's a resume)"
    text=$(pdftotext -l 1 -q "$PDF" - 2>/dev/null | head -c 2000)

    prompt="You are a PDF classifier. Analyze this text excerpt and determine if it's a RESEARCH PAPER, TECHNICAL ARTICLE, or BOOK about computer science, programming, software engineering, algorithms, or computation theory.

EXCLUDE: resumes, CVs, invoices, tax forms, flight manuals, aviation documents, personal documents, legal documents, contracts, receipts.

Text excerpt:
\"\"\"
$text
\"\"\"

Is this a programming/CS research paper, article, or book? Answer ONLY with 'yes' or 'no'."

    response=$(echo "$prompt" | ollama run "$MODEL" 2>/dev/null | tr '[:upper:]' '[:lower:]' | grep -oE '^(yes|no)' | head -1)
    echo "Response: $response"
    echo ""
fi

echo "Test complete!"
