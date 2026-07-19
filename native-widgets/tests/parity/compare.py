"""Compares the reference renders against the extracted library's renders."""

import sys

import numpy as np
from PIL import Image, ImageChops

PAGES = ["controls", "surfaces", "navigation", "overlays"]


def compare(out_dir):
    failures = 0
    for page in PAGES:
        reference = Image.open(f"{out_dir}/ref-{page}.png").convert("RGB")
        rendered = Image.open(f"{out_dir}/new-{page}.png").convert("RGB")

        if reference.size != rendered.size:
            print(f"{page:<11} size mismatch {reference.size} vs {rendered.size}")
            failures += 1
            continue

        diff = ImageChops.difference(reference, rendered)
        bbox = diff.getbbox()
        if bbox is None:
            print(f"{page:<11} identical")
            continue

        pixels = np.asarray(diff)
        differing = int(np.count_nonzero(pixels.any(axis=2)))
        total = reference.size[0] * reference.size[1]
        worst = int(pixels.max())
        print(
            f"{page:<11} {differing} / {total} px differ "
            f"({100.0 * differing / total:.4f}%), max channel delta {worst}, "
            f"bbox {bbox}"
        )
        diff.save(f"{out_dir}/diff-{page}.png")
        failures += 1

    print()
    if failures:
        print(f"{failures} of {len(PAGES)} pages differ")
        return 1
    print("all pages pixel-identical")
    return 0


if __name__ == "__main__":
    sys.exit(compare(sys.argv[1] if len(sys.argv) > 1 else "out"))
