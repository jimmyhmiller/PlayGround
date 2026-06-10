#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLUGIN_DIR="$ROOT/target/graphviz"
GRAPHVIZ_PLUGIN_DIR="$(pkg-config --variable=libdir libgvc)/graphviz"
CONFIG="$PLUGIN_DIR/config8"

"$ROOT/scripts/build.sh"

find "$GRAPHVIZ_PLUGIN_DIR" -maxdepth 1 -type f -name 'libgvplugin_*.dylib' | while read -r plugin; do
  ln -sf "$plugin" "$PLUGIN_DIR/$(basename "$plugin")"
done

rm -f "$CONFIG"
cp "$GRAPHVIZ_PLUGIN_DIR"/config8 "$CONFIG"
chmod u+w "$CONFIG"

if ! grep -q "libgvplugin_ion.8.dylib" "$CONFIG"; then
  cat >> "$CONFIG" <<'CFG'
libgvplugin_ion.8.dylib ion {
	layout {
		ion 0
	}
}
CFG
fi

echo "Local plugin dir: $PLUGIN_DIR"
echo "Run with:"
echo "  GVBINDIR=$PLUGIN_DIR dot -Kion -Tsvg examples/diamond.dot > /tmp/ion-diamond.svg"
