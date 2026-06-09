#!/usr/bin/env python3
"""Build a native-style macOS app icon from a full-bleed square source.

Lays the artwork on Apple's 1024 grid: an 824x824 rounded (continuous-corner
squircle) body centered with 100px transparent margin. No drop shadow.

Usage: make-icon.py SOURCE.png OUT-1024.png
"""
import sys
from PIL import Image, ImageDraw

SRC, OUT = sys.argv[1], sys.argv[2]

CANVAS = 1024
BODY = 824                    # icon content square (Apple grid)
MARGIN = (CANVAS - BODY) // 2 # 100
RADIUS_RATIO = 0.2237         # continuous-corner squircle ratio
SS = 4                        # supersample for crisp mask edges

def squircle_mask(size, ratio):
    big = size * SS
    m = Image.new("L", (big, big), 0)
    ImageDraw.Draw(m).rounded_rectangle(
        [0, 0, big - 1, big - 1], radius=ratio * size * SS, fill=255)
    return m.resize((size, size), Image.LANCZOS)

im = Image.open(SRC).convert("RGBA")
body = im.resize((BODY, BODY), Image.LANCZOS)
body.putalpha(squircle_mask(BODY, RADIUS_RATIO))

canvas = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
canvas.paste(body, (MARGIN, MARGIN), body)
canvas.save(OUT)
print(f"wrote {OUT}  ({BODY}px squircle body, {MARGIN}px margin, no shadow)")
