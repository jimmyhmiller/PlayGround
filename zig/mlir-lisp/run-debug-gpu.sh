#!/bin/bash
set -e

# Rebuild
zig build

# Run with debug
DEBUG_GPU_PASSES=1 ./zig-out/bin/mlir_lisp examples/gpu_hello_simple.lisp 2>&1 | grep -A 50 "After Stage"
