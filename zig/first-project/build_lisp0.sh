#!/bin/bash
set -e

echo "Building lisp0..."
zig build

# Create build0/bin directories if they don't exist
mkdir -p build0/bin

# Copy to current project
echo "Copying lisp0 to build0/bin/lisp0..."
cp zig-out/bin/lisp0 build0/bin/lisp0

# Copy to lisp-project
LISP_PROJECT_PATH="/Users/jimmyhmiller/Documents/Code/PlayGround/lisp-project"
if [ -d "$LISP_PROJECT_PATH" ]; then
    mkdir -p "$LISP_PROJECT_PATH/build0/bin"
    echo "Copying lisp0 to $LISP_PROJECT_PATH/build0/bin/lisp0..."
    cp zig-out/bin/lisp0 "$LISP_PROJECT_PATH/build0/bin/lisp0"
else
    echo "Warning: lisp-project directory not found at $LISP_PROJECT_PATH"
fi

echo "Build complete! lisp0 binary is ready."
