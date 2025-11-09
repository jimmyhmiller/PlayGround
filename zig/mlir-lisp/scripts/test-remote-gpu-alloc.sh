#!/bin/bash
# Sync and test GPU allocation on remote machine

set -e

HOST="jimmyhmiller@192.168.68.147"
REMOTE_DIR="~/mlir-lisp-remote"

echo "==> Syncing files to remote..."
rsync -avz --exclude 'zig-cache' --exclude 'zig-out' --exclude '.git' \
  test_gpu_memcpy.mlir test-gpu-alloc-pipeline.sh \
  "$HOST:$REMOTE_DIR/"

echo ""
echo "==> Running pipeline test on remote AMD machine..."
ssh "$HOST" "cd $REMOTE_DIR && bash test-gpu-alloc-pipeline.sh test_gpu_memcpy.mlir"
