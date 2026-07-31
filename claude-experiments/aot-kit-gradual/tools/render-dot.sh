#!/usr/bin/env bash
# Split tools/dot-dump.coil's marker-separated output into one .dot per fixture and render
# each to SVG. Reads the dump on stdin, or generates it if given no input.
#
#   tools/render-dot.sh                 # build, dump, render into out/dot/
#   coil run tools/dot-dump.coil | tools/render-dot.sh
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=out/dot
mkdir -p "$OUT"
rm -f "$OUT"/*.dot "$OUT"/*.svg

# Read a dump from stdin only when explicitly asked. Testing `[ -t 0 ]` instead means any
# non-tty caller (a script, a tool, CI) silently blocks on a `cat` that never gets EOF.
if [ "${1:-}" = "--stdin" ]; then
  cat > "$OUT/all.txt"
else
  coil run tools/dot-dump.coil > "$OUT/all.txt"
fi

awk -v out="$OUT" '
  /^=== FIXTURE /  { name=$3; file=out "/" name ".dot"; next }
  /^=== END$/      { close(file); name=""; next }
  name != ""       { print > file }
' "$OUT/all.txt"

for f in "$OUT"/*.dot; do
  dot -Tsvg "$f" -o "${f%.dot}.svg"
  printf '%-22s %5s bytes of dot -> svg\n' "$(basename "${f%.dot}")" "$(wc -c < "$f" | tr -d ' ')"
done
