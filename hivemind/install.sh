#!/usr/bin/env bash
# Symlink the hivemind CLI into ~/.local/bin (which is on PATH).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BIN="$HERE/bin/hivemind"
DEST="${1:-$HOME/.local/bin}"

chmod +x "$BIN"
mkdir -p "$DEST"
ln -sf "$BIN" "$DEST/hivemind"
echo "linked $DEST/hivemind -> $BIN"

if ! command -v hivemind >/dev/null 2>&1; then
  echo "warning: 'hivemind' not found on PATH. Add this to your shell rc:"
  echo "  export PATH=\"$DEST:\$PATH\""
fi
