#!/bin/bash
# Quick demo script to show GPT-2 text generation

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         GPT-2 Interactive Generation Demo             ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Virtual environment not found. Creating...${NC}"
    python3 -m venv venv
    source venv/bin/activate
    echo -e "${YELLOW}Installing tiktoken...${NC}"
    pip install -q tiktoken
    echo -e "${GREEN}✓ Setup complete!${NC}"
    echo ""
else
    source venv/bin/activate
fi

# Check if model file exists
if [ ! -f "gpt2_124M.bin" ]; then
    echo -e "${RED}Error: gpt2_124M.bin not found!${NC}"
    echo "Please place the GPT-2 model file in this directory."
    exit 1
fi

# Get prompt from argument or use default
PROMPT="${1:-Hello world}"

echo -e "${GREEN}Prompt:${NC} \"$PROMPT\""
echo ""

# Encode the prompt to show tokens
echo -e "${BLUE}Encoding prompt...${NC}"
python tokenize.py encode "$PROMPT"
echo ""

# Run the interactive demo
echo -e "${BLUE}Running interactive generation...${NC}"
echo ""
python interactive.py "$PROMPT" lisp

echo ""
echo -e "${GREEN}Demo complete!${NC}"
echo ""
echo -e "Try different prompts:"
echo -e "  ${YELLOW}./demo.sh 'The quick brown fox'${NC}"
echo -e "  ${YELLOW}./demo.sh 'Once upon a time'${NC}"
echo -e "  ${YELLOW}./demo.sh 'In the beginning'${NC}"
echo ""
