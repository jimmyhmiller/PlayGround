#!/bin/bash
# Remove the host-switch daemon, binary, and any managed /etc/hosts block.
# Run with: sudo ./uninstall.sh
set -euo pipefail

LABEL="com.jimmyhmiller.host-switch"
BIN_DST="/usr/local/bin/host-switch"
PLIST_DST="/Library/LaunchDaemons/${LABEL}.plist"

if [[ "$(id -u)" -ne 0 ]]; then
    echo "Must run as root: sudo ./uninstall.sh" >&2
    exit 1
fi

echo "==> Unloading daemon"
launchctl bootout system "$PLIST_DST" 2>/dev/null || true

echo "==> Removing managed /etc/hosts block"
# Strip the block before deleting the binary.
"$BIN_DST" away 2>/dev/null || \
    /usr/bin/sed -i '' '/# >>> host-switch (managed) >>>/,/# <<< host-switch (managed) <<</d' /etc/hosts || true

echo "==> Removing files"
rm -f "$PLIST_DST" "$BIN_DST"

echo "==> Done."
