#!/bin/sh
# Compile a Zig file to LLVM IR
# Usage: ./zig-to-llvm-ir.sh <input.zig> [output.ll]

set -e

if [ $# -lt 1 ]; then
  echo "Usage: $0 <input.zig> [output.ll]"
  echo "Example: $0 src/c_api_transform.zig c_api_transform.ll"
  exit 1
fi

INPUT_FILE="$1"
OUTPUT_FILE="${2:-${INPUT_FILE%.zig}.ll}"

if [ ! -f "$INPUT_FILE" ]; then
  echo "Error: Input file '$INPUT_FILE' not found"
  exit 1
fi

echo "Compiling $INPUT_FILE to LLVM IR..."

# Compile Zig to LLVM IR
# -femit-llvm-ir outputs LLVM IR
# -fno-emit-bin prevents binary generation
zig build-obj "$INPUT_FILE" -femit-llvm-ir="$OUTPUT_FILE" -fno-emit-bin

echo "Generated LLVM IR: $OUTPUT_FILE"
echo ""
echo "To convert to MLIR, run:"
echo "  mlir-translate --import-llvm $OUTPUT_FILE -o ${OUTPUT_FILE%.ll}.mlir"
