"""Compares the reference renders against the extracted library's renders.

One region is excluded: the spinner animates off the wall clock, so the two
renderers cannot agree on its arc unless they run in the same instant. The
original differs from its own second run there by a comparable amount. The
exclusion is reported on every run, never applied silently, and the pixel count
inside it is printed so a real regression hiding behind it would still show.
"""

import sys

import numpy as np
from PIL import Image, ImageChops

PAGES = ["controls", "surfaces", "navigation", "overlays"]

LOGICAL_WIDTH = 960.0

# page -> [(label, left, top, width, height)] in logical coordinates
ANIMATED = {
    "surfaces": [("spinner", 824.0, 458.0, 28.0, 28.0)],
}


def compare(out_dir):
    failures = 0

    for page in PAGES:
        reference = Image.open(f"{out_dir}/ref-{page}.png").convert("RGB")
        rendered = Image.open(f"{out_dir}/new-{page}.png").convert("RGB")

        if reference.size != rendered.size:
            print(f"{page:<11} size mismatch {reference.size} vs {rendered.size}")
            failures += 1
            continue

        diff = np.asarray(ImageChops.difference(reference, rendered))
        differing = diff.any(axis=2)
        total = reference.size[0] * reference.size[1]
        scale = reference.size[0] / LOGICAL_WIDTH

        excluded_notes = []
        for label, left, top, width, height in ANIMATED.get(page, []):
            x0, y0 = int(left * scale), int(top * scale)
            x1, y1 = int((left + width) * scale), int((top + height) * scale)
            inside = int(np.count_nonzero(differing[y0:y1, x0:x1]))
            differing[y0:y1, x0:x1] = False
            excluded_notes.append(f"{label} excluded (animated), {inside} px differ inside it")

        count = int(np.count_nonzero(differing))
        note = "; ".join(excluded_notes)
        suffix = f"  [{note}]" if note else ""

        if count == 0:
            print(f"{page:<11} identical{suffix}")
            continue

        rows, cols = np.nonzero(differing)
        bbox = (int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1)
        worst = int(diff.max())
        print(
            f"{page:<11} {count} / {total} px differ ({100.0 * count / total:.4f}%), "
            f"max channel delta {worst}, bbox {bbox}{suffix}"
        )
        Image.fromarray((differing * 255).astype(np.uint8)).save(f"{out_dir}/diff-{page}.png")
        failures += 1

    print()
    if failures:
        print(f"{failures} of {len(PAGES)} pages differ")
        return 1
    print(f"all {len(PAGES)} pages pixel-identical")
    return 0


if __name__ == "__main__":
    sys.exit(compare(sys.argv[1] if len(sys.argv) > 1 else "out"))
