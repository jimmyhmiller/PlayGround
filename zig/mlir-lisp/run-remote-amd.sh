#!/bin/bash
set -e

# Configuration
REMOTE_HOST="192.168.0.55"
REMOTE_USER="${REMOTE_USER:-jimmyhmiller}"
REMOTE_DIR="~/mlir-lisp-remote"
LISP_FILE="${1}"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}==> Syncing project to remote host...${NC}"
rsync -avz --delete \
  --exclude '.zig-cache' \
  --exclude 'zig-out' \
  --exclude '.git' \
  --exclude '*.o' \
  --exclude 'lib' \
  ./ "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"

echo -e "${BLUE}==> Building on remote AMD machine...${NC}"
ssh "${REMOTE_USER}@${REMOTE_HOST}" "cd ${REMOTE_DIR} && zig build -Doptimize=ReleaseFast"

if [ $? -ne 0 ]; then
    echo -e "${RED}==> Build failed!${NC}"
    exit 1
fi

echo -e "${BLUE}==> Running on remote AMD machine...${NC}"
echo -e "${GREEN}========================================${NC}"

if [ -n "$LISP_FILE" ]; then
    echo -e "${YELLOW}Running: ${LISP_FILE}${NC}"
    ssh "${REMOTE_USER}@${REMOTE_HOST}" "cd ${REMOTE_DIR} && ./zig-out/bin/mlir_lisp $LISP_FILE"
else
    echo -e "${YELLOW}Starting REPL${NC}"
    ssh -t "${REMOTE_USER}@${REMOTE_HOST}" "cd ${REMOTE_DIR} && ./zig-out/bin/mlir_lisp"
fi

EXIT_CODE=$?

echo -e "${GREEN}========================================${NC}"

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}==> Success!${NC}"
else
    echo -e "${RED}==> Failed with exit code: ${EXIT_CODE}${NC}"
fi

exit $EXIT_CODE
