#!/usr/bin/env bash
# GPT-2 forward pass benchmark: all backends
#
# Usage: ./bench/run_bench.sh [seq_len] [warmup] [iters]
#   defaults: seq_len=8, warmup=2, iters=5

set -euo pipefail
cd "$(dirname "$0")/.."

SEQ=${1:-8}
WARMUP=${2:-2}
ITERS=${3:-5}

echo "=== GPT-2 Forward Pass Benchmark (T=$SEQ, warmup=$WARMUP, iters=$ITERS) ==="
echo ""

# --- Build ---
echo "Building..." >&2
cc -O3 -march=native -o bench/gpt2_cpu bench/gpt2_cpu.c -lm
cc -O3 -march=native -DUSE_ACCELERATE -DACCELERATE_NEW_LAPACK -o bench/gpt2_cpu_blas bench/gpt2_cpu.c -lm -framework Accelerate 2>/dev/null
cargo build --release -p tensor-lang-gpu --features native --bin bench-gpt2 2>/dev/null
echo "" >&2

# --- Run all ---
./bench/gpt2_cpu "$SEQ" "$WARMUP" "$ITERS" 2>/dev/null
./bench/gpt2_cpu_blas "$SEQ" "$WARMUP" "$ITERS" 2>/dev/null

# PyTorch (if available)
TORCH_PY="${TORCH_PY:-/tmp/torch_bench/bin/python3}"
if [ -x "$TORCH_PY" ]; then
    "$TORCH_PY" bench/gpt2_pytorch.py "$SEQ" "$WARMUP" "$ITERS" 2>/dev/null
fi

# WASM + GPU
./target/release/bench-gpt2 "$SEQ" "$WARMUP" "$ITERS" 2>/dev/null | grep -E "^(WASM|GPU)"

echo ""
