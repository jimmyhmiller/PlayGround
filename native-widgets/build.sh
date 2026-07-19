#!/bin/sh
#
# Builds the library, its tests, and the raylib demo.

set -e
cd "$(dirname "$0")"

RAYLIB_CFLAGS="$(pkg-config --cflags raylib 2>/dev/null || echo -I/opt/homebrew/include)"
RAYLIB_LIBS="$(pkg-config --libs raylib 2>/dev/null || echo '-L/opt/homebrew/lib -lraylib')"

echo "library"
coil build src/widgets.coil --lib -o libnativewidgets.a

echo "tests"
cc -Wall -Wextra -Iinclude tests/abi_test.c libnativewidgets.a -o abi_test
./abi_test

echo "demo"
cc -Wall -Wextra -Iinclude -Ibackends/raylib $RAYLIB_CFLAGS \
  backends/raylib/nw_raylib.c examples/catalog.c libnativewidgets.a \
  -o catalog $RAYLIB_LIBS -lm
./catalog --check

echo
echo "built libnativewidgets.a and ./catalog"
