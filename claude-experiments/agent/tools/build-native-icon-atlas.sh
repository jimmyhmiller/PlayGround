#!/bin/sh
set -eu

native_checkout=${1:?pass the vercel-labs/native checkout path}
workspace=$(mktemp -d)
trap 'rm -rf "$workspace"' EXIT

for source in "$native_checkout"/src/primitives/canvas/icons/*.svg; do
  name=$(basename "$source" .svg)
  sed 's/currentColor/white/g' "$source" > "$workspace/$name.svg"
  rsvg-convert -w 48 -h 48 "$workspace/$name.svg" -o "$workspace/$name.png"
done

magick montage "$workspace"/*.png \
  -background none \
  -geometry 48x48+8+8 \
  -tile 8x7 \
  assets/native-icons.png
