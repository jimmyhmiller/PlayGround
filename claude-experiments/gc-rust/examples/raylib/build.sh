#!/usr/bin/env bash
# Build the gc-rust + raylib game. Requires: raylib (brew install raylib) and a
# built `gcr` (cargo build --bin gcr). Run ./dodge to play (mouse moves the
# player, R restarts).
#
# NO C shim — gc-rust calls raylib's C API directly, including `Color`-by-value
# functions, via its AAPCS64 FFI. We only link raylib + the macOS frameworks.
set -euo pipefail
cd "$(dirname "$0")"

RAYLIB_PREFIX="${RAYLIB_PREFIX:-/opt/homebrew}"
GCR="${GCR:-../../target/debug/gcr}"

build_one() {
  "$GCR" build "$1.gcr" -o "$1" \
    --link-arg -L"$RAYLIB_PREFIX/lib" --link-arg -lraylib \
    --link-arg -framework --link-arg CoreVideo \
    --link-arg -framework --link-arg IOKit \
    --link-arg -framework --link-arg Cocoa \
    --link-arg -framework --link-arg GLUT \
    --link-arg -framework --link-arg OpenGL
}

build_one dodge
echo "built: $(pwd)/dodge   (run it: ./dodge)"
