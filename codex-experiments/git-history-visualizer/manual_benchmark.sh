#!/bin/bash
# Manual benchmark since Python git-of-theseus has dependency issues

REPO=$1
if [ -z "$REPO" ]; then
    echo "Usage: $0 <repo_path>"
    exit 1
fi

RUST_BIN="target/release/git-history-visualizer"
OUT_RUST="/tmp/bench-rust"
OUT_PYTHON="/tmp/bench-python"

rm -rf "$OUT_RUST" "$OUT_PYTHON"
mkdir -p "$OUT_RUST" "$OUT_PYTHON"

echo "Benchmarking Rust implementation..."
echo "======================================"
time "$RUST_BIN" analyze "$REPO" --outdir "$OUT_RUST" --quiet --jobs 2

echo ""
echo "Results:"
echo "--------"
ls -lh "$OUT_RUST"/*.json | awk '{print $9, $5}'
