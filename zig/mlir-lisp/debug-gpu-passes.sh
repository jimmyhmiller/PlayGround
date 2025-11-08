#!/bin/bash
# Run on remote with debug output
DEBUG_GPU_PASSES=1 ./zig-out/bin/mlir_lisp "$@"
