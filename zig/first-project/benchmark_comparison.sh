#!/bin/bash

echo "======================================"
echo "GPT-2 Performance Comparison"
echo "======================================"
echo ""

# Benchmark 1: Original C implementation
echo "1. Benchmarking ORIGINAL C implementation..."
echo "   (100 forward passes, T=8, -O3 optimization)"
cd /Users/jimmyhmiller/Documents/Code/PlayGround/zig/first-project/original_llm_c
time ./simple_bench 2>&1 | tail -5
echo ""

# Benchmark 2: Your Lisp implementation  
echo "2. Benchmarking YOUR LISP implementation..."
echo "   (100 tokens with fixed context window=8, -O3 optimization)"
cd /Users/jimmyhmiller/Documents/Code/PlayGround/zig/first-project/examples/llm
time ./llm 2>&1 | grep -A 5 "Generating 100"
echo ""

echo "======================================"
echo "Summary:"
echo "  - Both use -O3 optimization"
echo "  - Both process context window of 8 tokens"
echo "  - Original C: 100 forward passes"
echo "  - Your Lisp: 100 token generation (100 forward passes)"
echo "======================================"
