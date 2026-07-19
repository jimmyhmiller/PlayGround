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

echo "raylib driver"
cc -Wall -Wextra -Iinclude -Ibackends/raylib -Iexamples $RAYLIB_CFLAGS \
  backends/raylib/nw_raylib.c examples/catalog_pages.c examples/catalog_raylib.c \
  libnativewidgets.a -o catalog $RAYLIB_LIBS -lm
./catalog --check

# The same catalog_pages.c, no graphics library in sight. If a renderer
# assumption had leaked into the widgets this would not link, let alone draw.
echo "svg driver"
cc -Wall -Wextra -Iinclude -Ibackends/svg -Iexamples \
  backends/svg/nw_svg.c examples/catalog_pages.c examples/catalog_svg.c \
  libnativewidgets.a -o catalog_svg -lm

echo
echo "built libnativewidgets.a, ./catalog (raylib) and ./catalog_svg (svg)"
