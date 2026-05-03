#!/usr/bin/env bash
# Builds a self-contained static index.html from the React-driven Showcase.html.
# Renders the page with headless Chrome, captures the post-mount DOM, strips
# the runtime scripts, and writes a single static file that only needs
# styles.css and the assets/ directory.
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
PORT="${PORT:-8765}"
URL="http://localhost:${PORT}/Showcase.html"
OUT="${1:-${DIR}/index.html}"

CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
[[ -x "$CHROME" ]] || CHROME="$(command -v google-chrome || command -v chromium || true)"
[[ -x "$CHROME" ]] || { echo "Chrome not found"; exit 1; }

if ! curl -sf -o /dev/null "$URL"; then
  echo "Server not reachable at $URL — start it with: python3 -m http.server $PORT"
  exit 1
fi

DUMP="$(mktemp -t showcase-dom.XXXXXX.html)"
trap 'rm -f "$DUMP"' EXIT

"$CHROME" \
  --headless=new \
  --disable-gpu \
  --hide-scrollbars \
  --window-size=1280,800 \
  --force-device-scale-factor=1 \
  --virtual-time-budget=15000 \
  --run-all-compositor-stages-before-draw \
  --dump-dom \
  "$URL" > "$DUMP"

python3 - "$DUMP" "$OUT" <<'PY'
import sys, re

dump_path, out_path = sys.argv[1], sys.argv[2]
with open(dump_path, 'r', encoding='utf-8') as f:
    html = f.read()

# 1) Drop every <script> tag (React, Babel, our inline app, edit-mode code).
html = re.sub(r'<script\b[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)

# 2) Drop preconnect / preload tags that pointed at unpkg etc — keep the
#    Google Fonts ones since styles.css references them.
html = re.sub(
    r'<link\s+rel="preconnect"\s+href="https://unpkg[^"]*"[^/]*/?>',
    '',
    html,
)

# 3) Drop the runtime-injected <style id="__page_size"> if any leaked through.
html = re.sub(
    r'<style\s+id="__page_size">.*?</style>',
    '',
    html,
    flags=re.DOTALL,
)

# 4) Reset the document title that the React app may have rewritten.
html = re.sub(
    r'<title>[^<]*</title>',
    '<title>jimmyhmiller — selected work</title>',
    html,
    count=1,
)

# 5) Inject a tiny inline script that re-creates the print-mode width pinning
#    so print.sh keeps working against this static file.
print_script = """
<script>
(function () {
  if (!/[?&]print=1/.test(location.search)) return;
  var d = document;
  d.documentElement.style.width = "1280px";
  d.body.style.width = "1280px";
  d.body.style.margin = "0 auto";
  d.documentElement.style.setProperty("--section-pad-y", "90px");
  d.documentElement.style.setProperty("--gutter", "60px");
  var s = d.createElement("style");
  s.textContent =
    ".hero { padding: 110px 60px 80px !important; } " +
    ".project { padding: 90px 60px !important; } " +
    ".imgblock video { display: none !important; } " +
    ".imgblock .video-poster { display: block !important; width: 100%; height: auto; border-radius: 6px; }";
  d.head.appendChild(s);
})();
</script>
""".strip()

html = html.replace('</body>', print_script + '\n</body>', 1)

with open(out_path, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"Wrote {out_path}")
PY
