#!/usr/bin/env bash
# Render every corpus graph with both -Kion and -Kdot and build an HTML
# gallery (target/gallery/index.html) for side-by-side human review.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT="$ROOT/target/gallery"
mkdir -p "$OUT"
export GVBINDIR="$ROOT/target/graphviz"

HTML="$OUT/index.html"
cat > "$HTML" <<'HEAD'
<!doctype html>
<meta charset="utf-8">
<title>ion-layout gallery: -Kion vs -Kdot</title>
<style>
  body { font: 14px/1.4 -apple-system, sans-serif; margin: 24px; background: #fafafa; }
  h2 { margin: 36px 0 8px; font-size: 16px; }
  .pair { display: flex; gap: 16px; align-items: flex-start; }
  .half { flex: 1; min-width: 0; background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 8px; }
  .half h3 { margin: 0 0 6px; font-size: 12px; color: #666; font-weight: 600; }
  .half img { max-width: 100%; height: auto; max-height: 720px; object-fit: contain; }
</style>
<h1>ion-layout gallery</h1>
<p>Left: <code>dot -Kion</code> (this project). Right: <code>dot -Kdot</code> (stock Graphviz).</p>
HEAD

for f in corpus/*.dot; do
  name="$(basename "$f" .dot)"
  ion_svg="$OUT/$name.ion.svg"
  dot_svg="$OUT/$name.dot.svg"
  if ! dot -Kion -Tsvg "$f" > "$ion_svg" 2> "$OUT/$name.ion.err"; then
    echo "ION FAILED: $name (see $OUT/$name.ion.err)"
    ion_svg=""
  fi
  if ! dot -Kdot -Tsvg "$f" > "$dot_svg" 2> "$OUT/$name.dot.err"; then
    dot_svg=""
  fi
  {
    echo "<h2>$name</h2><div class=\"pair\">"
    if [ -n "$ion_svg" ]; then
      echo "<div class=\"half\"><h3>-Kion</h3><img src=\"$name.ion.svg\"></div>"
    else
      echo "<div class=\"half\"><h3>-Kion</h3><p>FAILED</p></div>"
    fi
    if [ -n "$dot_svg" ]; then
      echo "<div class=\"half\"><h3>-Kdot</h3><img src=\"$name.dot.svg\"></div>"
    else
      echo "<div class=\"half\"><h3>-Kdot</h3><p>FAILED</p></div>"
    fi
    echo "</div>"
  } >> "$HTML"
done

echo "Gallery: $HTML"
