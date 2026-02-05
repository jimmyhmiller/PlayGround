#!/bin/bash
# Run integration tests on remote server

set -e

REMOTE_HOST="jimmyhmiller@computer.jimmyhmiller.com"
REMOTE_DIR="~/keep-running-test"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== Syncing code to $REMOTE_HOST ==="
rsync -avz --delete \
    --exclude 'target/' \
    --exclude '.git/' \
    --exclude '*.log' \
    "$LOCAL_DIR/" "$REMOTE_HOST:$REMOTE_DIR/"

echo ""
echo "=== Running tests on $REMOTE_HOST ==="
ssh "$REMOTE_HOST" "source ~/.cargo/env 2>/dev/null || true; cd $REMOTE_DIR && cargo test --test integration -- --nocapture"
