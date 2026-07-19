#!/bin/sh
#
# Pixel-parity gate: renders the catalog twice — once from flowline's original
# raylib-coupled widgets, once through the extracted library and its raylib
# backend — and compares the two.
#
# Exits 0 only when every page is pixel-identical.
#
# Both renders need flowline's fonts, so both run with that repo as the working
# directory.

set -e
cd "$(dirname "$0")"

PARITY_DIR="$(pwd)"
ROOT="$(cd ../.. && pwd)"
FLOWLINE="$(cd ../../../claude-experiments/agent && pwd)"
OUT="${PARITY_DIR}/out"

mkdir -p "$OUT"
rm -f "$OUT"/*.png

echo "building reference (original widgets)"
coil build reference.coil -o "$OUT/reference" -lraylib -lm >/dev/null

echo "building catalog (extracted library)"
cd "$ROOT"
coil build src/widgets.coil --lib -o libnativewidgets.a >/dev/null
cc -Wall -Wextra -Iinclude -Ibackends/raylib \
  $(pkg-config --cflags raylib 2>/dev/null || echo -I/opt/homebrew/include) \
  backends/raylib/nw_raylib.c examples/catalog.c libnativewidgets.a \
  -o "$OUT/catalog" \
  $(pkg-config --libs raylib 2>/dev/null || echo "-L/opt/homebrew/lib -lraylib") -lm

# raylib's TakeScreenshot prefixes the working directory, so both renderers
# must write relative names and have them moved afterwards.
echo "rendering"
cd "$FLOWLINE"
"$OUT/reference" >/dev/null 2>&1
mv ref-*.png "$OUT"/
"$OUT/catalog" --snapshot "new-" >/dev/null 2>&1
mv new-*.png "$OUT"/

echo "comparing"
python3 "$PARITY_DIR/compare.py" "$OUT"
