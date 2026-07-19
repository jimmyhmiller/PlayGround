#!/usr/bin/env bash
# Symlink bin/verify -> ~/.local/bin/verify  (which must be on your PATH).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
dest="${HOME}/.local/bin"
mkdir -p "$dest"
chmod +x "$here/bin/verify"
ln -sf "$here/bin/verify" "$dest/verify"
echo "linked $dest/verify -> $here/bin/verify"
echo "ensure $dest is on your PATH, then: verify --help"
