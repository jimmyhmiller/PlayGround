#!/usr/bin/env bash
# Builds libraygui.dylib from raygui.h against the system raylib.
# Run from the project root or from vendor/.
set -euo pipefail

cd "$(dirname "$0")"

RAYLIB_INC="/opt/homebrew/include"
RAYLIB_LIB="/opt/homebrew/lib"

clang -O2 -fPIC -dynamiclib \
    -I"$RAYLIB_INC" \
    -L"$RAYLIB_LIB" \
    -lraylib \
    -install_name "@rpath/libraygui.dylib" \
    -DRAYGUI_IMPLEMENTATION \
    -x c raygui.h \
    -o libraygui.dylib

echo "built: $(pwd)/libraygui.dylib"
