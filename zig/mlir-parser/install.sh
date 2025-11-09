#!/bin/bash

# Build and Install Script for mlir-to-lisp
# This script builds the project and installs binaries to ~/.local/bin

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}Building mlir-parser project (release mode)...${NC}"
zig build -Doptimize=ReleaseFast

echo -e "${GREEN}✓ Build successful${NC}"

# Create ~/.local/bin if it doesn't exist
mkdir -p ~/.local/bin

# Install mlir-to-lisp
echo -e "${BLUE}Installing mlir-to-lisp...${NC}"
ln -sf "$(pwd)/zig-out/bin/mlir-to-lisp" ~/.local/bin/mlir-to-lisp

# Install mlir_parser (the main parser binary)
echo -e "${BLUE}Installing mlir_parser...${NC}"
ln -sf "$(pwd)/zig-out/bin/mlir_parser" ~/.local/bin/mlir_parser

# Install debug_printer
echo -e "${BLUE}Installing debug_printer...${NC}"
ln -sf "$(pwd)/zig-out/bin/debug_printer" ~/.local/bin/debug_printer

echo -e "${GREEN}✓ Installation complete${NC}"
echo ""
echo "Installed binaries:"
echo "  - mlir-to-lisp   -> ~/.local/bin/mlir-to-lisp"
echo "  - mlir_parser    -> ~/.local/bin/mlir_parser"
echo "  - debug_printer  -> ~/.local/bin/debug_printer"
echo ""

# Check if ~/.local/bin is in PATH
if echo "$PATH" | grep -q "$HOME/.local/bin"; then
    echo -e "${GREEN}✓ ~/.local/bin is in your PATH${NC}"
    echo "You can now use the commands from anywhere:"
    echo "  mlir-to-lisp --help"
else
    echo -e "${RED}⚠ ~/.local/bin is NOT in your PATH${NC}"
    echo ""
    echo "Add this line to your ~/.zshrc (or ~/.bashrc):"
    echo '  export PATH="$HOME/.local/bin:$PATH"'
    echo ""
    echo "Then run: source ~/.zshrc"
fi
