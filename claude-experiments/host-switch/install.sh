#!/bin/bash
# Install host-switch as a root LaunchDaemon (runs at boot + every 30s + on network change).
# Run with: sudo ./install.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
LABEL="com.jimmyhmiller.host-switch"
BIN_DST="/usr/local/bin/host-switch"
PLIST_DST="/Library/LaunchDaemons/${LABEL}.plist"

if [[ "$(id -u)" -ne 0 ]]; then
    echo "Must run as root: sudo ./install.sh" >&2
    exit 1
fi

echo "==> Building release binary"
# Build as the invoking (non-root) user so cargo uses their toolchain/cache.
if [[ -n "${SUDO_USER:-}" ]]; then
    sudo -u "$SUDO_USER" bash -lc "cd '$HERE' && cargo build --release"
else
    (cd "$HERE" && cargo build --release)
fi

echo "==> Installing binary -> $BIN_DST"
install -d /usr/local/bin
install -m 0755 -o root -g wheel "$HERE/target/release/host-switch" "$BIN_DST"

echo "==> Installing LaunchDaemon -> $PLIST_DST"
install -m 0644 -o root -g wheel "$HERE/${LABEL}.plist" "$PLIST_DST"

echo "==> (Re)loading daemon"
launchctl bootout system "$PLIST_DST" 2>/dev/null || true
launchctl bootstrap system "$PLIST_DST"
launchctl enable "system/${LABEL}"
launchctl kickstart -k "system/${LABEL}"

echo "==> Done. Current decision:"
"$BIN_DST" status
echo
echo "Logs: /var/log/host-switch.log   (tail -f to watch)"
