#!/usr/bin/env python3
"""Better alignment using ECC and manual tuning."""

import cv2
import numpy as np

def load_normalized(path):
    """Load and normalize to dark text on light background."""
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if np.mean(gray) < 128:
        gray = 255 - gray
    return gray

def match_histograms(src, ref):
    """Match histogram of src to ref."""
    src_hist, _ = np.histogram(src.flatten(), 256, [0, 256])
    ref_hist, _ = np.histogram(ref.flatten(), 256, [0, 256])
    src_cdf = src_hist.cumsum() / src_hist.sum()
    ref_cdf = ref_hist.cumsum() / ref_hist.sum()
    lookup = np.searchsorted(ref_cdf, src_cdf).astype(np.uint8)
    return lookup[src]

def align_ecc(img1, img2):
    """Align using Enhanced Correlation Coefficient."""
    # Resize img2 to match img1 dimensions first
    img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Define motion model (Euclidean: rotation + translation)
    warp_mode = cv2.MOTION_AFFINE
    warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Criteria for termination
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-8)

    try:
        cc, warp_matrix = cv2.findTransformECC(
            img1.astype(np.float32),
            img2_resized.astype(np.float32),
            warp_matrix,
            warp_mode,
            criteria,
            inputMask=None,
            gaussFiltSize=5
        )
        print(f"ECC correlation: {cc:.4f}")
        print(f"Warp matrix:\n{warp_matrix}")

        aligned = cv2.warpAffine(
            img2_resized, warp_matrix,
            (img1.shape[1], img1.shape[0]),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderValue=255
        )
        return aligned, warp_matrix
    except cv2.error as e:
        print(f"ECC failed: {e}")
        return img2_resized, warp_matrix

def main():
    print("Loading...")
    img1 = load_normalized("first.png")
    img2 = load_normalized("second.png")

    print(f"Sizes: {img1.shape} vs {img2.shape}")

    # Match histograms
    img2_matched = match_histograms(img2, img1)

    # Align using ECC
    print("\nAligning with ECC...")
    aligned, warp = align_ecc(img1, img2_matched)

    # Create overlay
    overlay = np.zeros((*img1.shape, 3), dtype=np.uint8)
    overlay[:, :, 2] = img1  # Red
    overlay[:, :, 1] = aligned  # Green
    overlay[:, :, 0] = aligned  # Blue (cyan)

    cv2.imwrite("ecc_overlay.png", overlay)
    cv2.imwrite("ecc_img1.png", img1)
    cv2.imwrite("ecc_img2_aligned.png", aligned)

    # Difference
    diff = cv2.absdiff(img1, aligned)
    print(f"\nDiff stats: mean={np.mean(diff):.2f}, max={np.max(diff)}")

    cv2.imwrite("ecc_diff.png", np.clip(diff * 10, 0, 255).astype(np.uint8))

    # Signed diff
    signed = img1.astype(np.int16) - aligned.astype(np.int16)
    signed_viz = np.zeros((*img1.shape, 3), dtype=np.uint8)
    signed_viz[:, :, 2] = np.clip(signed * 5, 0, 255).astype(np.uint8)
    signed_viz[:, :, 0] = np.clip(-signed * 5, 0, 255).astype(np.uint8)
    cv2.imwrite("ecc_signed.png", signed_viz)

    print("\nSaved: ecc_overlay.png, ecc_diff.png, ecc_signed.png")

if __name__ == "__main__":
    main()
