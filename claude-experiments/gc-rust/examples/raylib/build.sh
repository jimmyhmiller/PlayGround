#!/usr/bin/env bash
# Build the gc-rust + raylib game. Requires: raylib (brew install raylib) and a
# built `gcr` (cargo build --bin gcr). Run ./dodge to play (mouse moves the
# player, R restarts).
set -euo pipefail
cd "$(dirname "$0")"

RAYLIB_PREFIX="${RAYLIB_PREFIX:-/opt/homebrew}"
GCR="${GCR:-../../target/debug/gcr}"

# 1. Compile the thin C shim (scalar wrappers around raylib's by-value-Color /
#    bool-returning calls).
cc -c rayshim.c -I"$RAYLIB_PREFIX/include" -o rayshim.o

# 2. Build the gc-rust game, linking the shim + raylib + the macOS frameworks
#    raylib needs (the dylib pulls most, but static/explicit is safe).
"$GCR" build dodge.gcr -o dodge \
  --link-arg rayshim.o \
  --link-arg -L"$RAYLIB_PREFIX/lib" --link-arg -lraylib \
  --link-arg -framework --link-arg CoreVideo \
  --link-arg -framework --link-arg IOKit \
  --link-arg -framework --link-arg Cocoa \
  --link-arg -framework --link-arg GLUT \
  --link-arg -framework --link-arg OpenGL

echo "built: $(pwd)/dodge   (run it: ./dodge)"
