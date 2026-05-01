#!/usr/bin/env bash
# Renders Showcase.html → showcase.pdf via headless Chrome.
# Requires the static server to be running (python3 -m http.server 8765 in this dir).
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
PORT="${PORT:-8765}"
URL="http://localhost:${PORT}/Showcase.html?print=1"
OUT="${1:-${DIR}/showcase.pdf}"

CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
[[ -x "$CHROME" ]] || CHROME="$(command -v google-chrome || command -v chromium || true)"
[[ -x "$CHROME" ]] || { echo "Chrome not found"; exit 1; }

if ! curl -sf -o /dev/null "$URL"; then
  echo "Server not reachable at $URL — start it with: python3 -m http.server $PORT"
  exit 1
fi

RAW="${OUT%.pdf}.raw.pdf"
QDF="${OUT%.pdf}.qdf.pdf"

"$CHROME" \
  --headless=new \
  --disable-gpu \
  --no-pdf-header-footer \
  --hide-scrollbars \
  --window-size=1280,800 \
  --force-device-scale-factor=1 \
  --virtual-time-budget=20000 \
  --run-all-compositor-stages-before-draw \
  --print-to-pdf="$RAW" \
  "$URL"

# Trim the trailing whitespace while preserving link annotations.
# 1. Detect the content bounding box via ghostscript.
# 2. Convert raw PDF to QDF (text-editable form).
# 3. Edit the MediaBox in place to clip the page to that bbox.
# 4. qpdf re-saves with proper xref offsets and keeps all annotations.
if command -v gs >/dev/null 2>&1 && command -v qpdf >/dev/null 2>&1; then
  BBOX=$(gs -dQUIET -dNOPAUSE -dBATCH -sDEVICE=bbox "$RAW" 2>&1 \
    | awk '/%%BoundingBox:/ { print $2,$3,$4,$5; exit }')
  read BL BB BR BT <<<"$BBOX"
  if [[ -n "$BT" ]]; then
    qpdf --qdf --object-streams=disable "$RAW" "$QDF"
    # Replace the MediaBox y-bottom (was 0) with the content's bottom (BB).
    # The MediaBox in QDF mode is one number per line. Match the exact pattern
    # we know Chrome emits: [ 0 0 960 14400 ].
    python3 - "$QDF" "$BB" <<'PY'
import re, sys
path, bb = sys.argv[1], sys.argv[2]
with open(path, "rb") as f:
    data = f.read()
# Match the MediaBox dictionary entry and replace the second '0' (y-bottom).
pattern = re.compile(
    rb"(/MediaBox\s*\[\s*\n\s*0\s*\n\s*)0(\s*\n\s*960\s*\n\s*14400\s*\n\s*\])",
    re.MULTILINE,
)
new = pattern.sub(rb"\g<1>" + bb.encode() + rb"\g<2>", data)
with open(path, "wb") as f:
    f.write(new)
PY
    qpdf "$QDF" "$OUT"
    rm -f "$RAW" "$QDF"
  else
    mv "$RAW" "$OUT"
    echo "(could not detect bbox — output not trimmed)"
  fi
else
  mv "$RAW" "$OUT"
  echo "(gs or qpdf missing — output not trimmed)"
fi

echo "Wrote $OUT"
