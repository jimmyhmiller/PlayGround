#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

cargo build --release

mkdir -p target/graphviz

cc -dynamiclib \
  -g \
  -o target/graphviz/libgvplugin_ion.8.dylib \
  plugin/gvplugin_ion.c \
  target/release/libion_layout.a \
  -Iinclude \
  $(pkg-config --cflags --libs libgvc libcgraph)

ln -sf libgvplugin_ion.8.dylib target/graphviz/libgvplugin_ion.dylib

echo "Built $ROOT/target/graphviz/libgvplugin_ion.8.dylib"
