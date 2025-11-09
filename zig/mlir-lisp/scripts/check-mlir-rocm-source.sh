#!/bin/bash
# Check if we can find MLIR ROCm runtime wrapper source

echo "Checking for MLIR ROCm runtime wrapper source..."

# Try to find in LLVM installation
MLIR_SRC="/opt/homebrew/Cellar/llvm/20.1.7/include/mlir"
if [ -d "$MLIR_SRC" ]; then
    echo "✓ Found MLIR headers at: $MLIR_SRC"
    find "$MLIR_SRC" -name "*rocm*" -o -name "*gpu*runtime*" 2>/dev/null | head -10
else
    echo "❌ MLIR headers not found"
fi

# Check for ExecutionEngine libs
echo ""
echo "Checking for MLIR ExecutionEngine libraries..."
find /opt/homebrew/Cellar/llvm/20.1.7/lib -name "*ExecutionEngine*" -o -name "*mlir_runner*" 2>/dev/null

# Check for any GPU-related MLIR libraries
echo ""
echo "Checking for GPU-related MLIR libraries..."
ls -la /opt/homebrew/Cellar/llvm/20.1.7/lib/ | grep -i "gpu\|rocm\|cuda" || echo "None found"
