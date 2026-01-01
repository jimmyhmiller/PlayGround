#!/bin/bash
# Remote test script for lispier GPU code
# Syncs files, compiles, and runs on remote machine

set -e

LOCAL_DIR="/Users/jimmyhmiller/Documents/Code/PlayGround/claude-experiments/lispier"
REMOTE_DIR="~/Code/lispier"
PRIMARY_HOST="jimmyhmiller@192.168.0.55"
BACKUP_HOST="jimmyhmiller@computer.jimmyhmiller.com"

# Determine which host to use
echo "Testing SSH connection..."
if ssh -o ConnectTimeout=5 -o BatchMode=yes "$PRIMARY_HOST" "true" 2>/dev/null; then
    HOST="$PRIMARY_HOST"
    echo "Using primary host: $HOST"
else
    HOST="$BACKUP_HOST"
    echo "Primary host unavailable, using backup: $HOST"
fi

# Create remote directory if needed
echo ""
echo "=== Creating remote directory ==="
ssh "$HOST" "mkdir -p $REMOTE_DIR"

# Sync files using rsync
echo ""
echo "=== Syncing files to remote ==="
rsync -avz --exclude 'target' --exclude '.git' --exclude '*.o' --exclude '*.so' \
    "$LOCAL_DIR/" "$HOST:$REMOTE_DIR/"

# Build on remote
echo ""
echo "=== Building on remote ==="
ssh "$HOST" "cd $REMOTE_DIR && \
    export MLIR_SYS_210_PREFIX=/usr/lib/llvm-21 && \
    export TABLEGEN_210_PREFIX=/usr/lib/llvm-21 && \
    ~/.cargo/bin/cargo build --release 2>&1 | tail -30"

# Run the GPT-2 GPU test
echo ""
echo "=== Running GPT-2 GPU inference ==="
ssh "$HOST" "cd $REMOTE_DIR && \
    export MLIR_SYS_210_PREFIX=/usr/lib/llvm-21 && \
    export TABLEGEN_210_PREFIX=/usr/lib/llvm-21 && \
    ~/.cargo/bin/cargo run --release -- run examples/gpt2/gpt2_generate_gpu.lisp 2>&1"
