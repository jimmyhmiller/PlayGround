#!/usr/bin/env python3
"""Final comparison with histogram matching and row alignment."""

import cv2
import numpy as np

def load_and_normalize(path):
    """Load image, convert to grayscale, normalize to text=dark, bg=light."""
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Make text dark on light background
    if np.mean(gray) < 128:
        gray = 255 - gray

    return img, gray

def histogram_match(source, reference):
    """Match histogram of source to reference."""
    # Get histograms
    src_hist, bins = np.histogram(source.flatten(), 256, [0, 256])
    ref_hist, _ = np.histogram(reference.flatten(), 256, [0, 256])

    # Cumulative distribution functions
    src_cdf = src_hist.cumsum()
    ref_cdf = ref_hist.cumsum()

    # Normalize
    src_cdf = src_cdf / src_cdf[-1]
    ref_cdf = ref_cdf / ref_cdf[-1]

    # Create lookup table
    lookup = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        j = np.searchsorted(ref_cdf, src_cdf[i])
        lookup[i] = min(j, 255)

    # Apply transformation
    return lookup[source]

def align_row(row1, row2):
    """Align row2 to row1, return aligned row2."""
    # Resize if needed
    if row1.shape[0] != row2.shape[0]:
        row2 = cv2.resize(row2, (row2.shape[1], row1.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Find shift using phase correlation
    shift, _ = cv2.phaseCorrelate(row1.astype(np.float32), row2.astype(np.float32))

    # Apply shift
    M = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
    aligned = cv2.warpAffine(row2, M, (row1.shape[1], row1.shape[0]),
                              flags=cv2.INTER_LINEAR, borderValue=255)

    return aligned, shift

def extract_text_mask(gray):
    """Create a mask of text pixels (dark pixels after normalization)."""
    # Text is dark, background is light
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    return mask

def main():
    print("Loading images...")
    img1_color, gray1 = load_and_normalize("first.png")
    img2_color, gray2 = load_and_normalize("second.png")

    print(f"Image 1: {gray1.shape}")
    print(f"Image 2: {gray2.shape}")

    # Match histogram of image 2 to image 1
    print("Matching histograms...")
    gray2_matched = histogram_match(gray2, gray1)

    # Row-by-row alignment
    num_rows = 4
    h1, w1 = gray1.shape
    h2, w2 = gray2_matched.shape
    row_h1 = h1 // num_rows
    row_h2 = h2 // num_rows

    print(f"Row heights: {row_h1} vs {row_h2}")

    aligned_rows = []
    ref_rows = []
    shifts = []

    for i in range(num_rows):
        r1 = gray1[i*row_h1:(i+1)*row_h1, :]
        r2 = gray2_matched[i*row_h2:(i+1)*row_h2, :]

        aligned, shift = align_row(r1, r2)
        aligned_rows.append(aligned)
        ref_rows.append(r1)
        shifts.append(shift)
        print(f"  Row {i}: shift=({shift[0]:.2f}, {shift[1]:.2f})")

    # Combine aligned rows
    aligned_full = np.vstack(aligned_rows)
    ref_full = np.vstack(ref_rows)

    # Compute difference
    diff = cv2.absdiff(ref_full, aligned_full)

    print(f"\nAfter histogram matching and row alignment:")
    print(f"  Mean difference: {np.mean(diff):.2f}")
    print(f"  Max difference: {np.max(diff)}")
    print(f"  Pixels diff > 5: {np.sum(diff > 5)} ({100*np.sum(diff > 5)/diff.size:.2f}%)")
    print(f"  Pixels diff > 10: {np.sum(diff > 10)} ({100*np.sum(diff > 10)/diff.size:.2f}%)")
    print(f"  Pixels diff > 20: {np.sum(diff > 20)} ({100*np.sum(diff > 20)/diff.size:.2f}%)")

    # Create text mask to analyze only text areas
    text_mask = extract_text_mask(ref_full)
    text_pixels = diff[text_mask > 0]

    if len(text_pixels) > 0:
        print(f"\nText pixels only:")
        print(f"  Count: {len(text_pixels)}")
        print(f"  Mean diff: {np.mean(text_pixels):.2f}")
        print(f"  Max diff: {np.max(text_pixels)}")
        print(f"  Std diff: {np.std(text_pixels):.2f}")

    # --- Visualizations ---

    # 1. Side by side (normalized)
    side_by_side = np.hstack([ref_full, np.ones((ref_full.shape[0], 10), dtype=np.uint8)*128, aligned_full])
    cv2.imwrite("final_side_by_side.png", side_by_side)

    # 2. Difference (raw and amplified)
    cv2.imwrite("final_diff_raw.png", diff)
    diff_amp = np.clip(diff * 10, 0, 255).astype(np.uint8)
    cv2.imwrite("final_diff_10x.png", diff_amp)

    # 3. Red-cyan overlay (perfect alignment = gray, misalignment = color fringe)
    overlay = np.zeros((*ref_full.shape, 3), dtype=np.uint8)
    overlay[:, :, 2] = ref_full  # Red = first
    overlay[:, :, 1] = aligned_full  # Green = second
    overlay[:, :, 0] = aligned_full  # Blue = second (makes cyan)
    cv2.imwrite("final_overlay.png", overlay)

    # 4. Signed difference visualization
    signed = ref_full.astype(np.int16) - aligned_full.astype(np.int16)
    signed_viz = np.zeros((*ref_full.shape, 3), dtype=np.uint8)
    # Red where first is lighter (weaker text edge in first)
    signed_viz[:, :, 2] = np.clip(signed * 5, 0, 255).astype(np.uint8)
    # Blue where second is lighter (weaker text edge in second)
    signed_viz[:, :, 0] = np.clip(-signed * 5, 0, 255).astype(np.uint8)
    cv2.imwrite("final_signed.png", signed_viz)

    # 5. Difference only in text regions
    diff_text_only = np.zeros_like(diff)
    diff_text_only[text_mask > 0] = diff[text_mask > 0]
    diff_text_amp = np.clip(diff_text_only * 10, 0, 255).astype(np.uint8)
    cv2.imwrite("final_diff_text_only.png", diff_text_amp)

    # 6. Difference heatmap on text
    heatmap = cv2.applyColorMap(diff_amp, cv2.COLORMAP_JET)
    cv2.imwrite("final_heatmap.png", heatmap)

    # 7. Flicker comparison (animated GIF would be ideal, but save frames)
    cv2.imwrite("final_frame1.png", ref_full)
    cv2.imwrite("final_frame2.png", aligned_full)

    # 8. Blend at 50%
    blend = cv2.addWeighted(ref_full, 0.5, aligned_full, 0.5, 0)
    cv2.imwrite("final_blend.png", blend)

    print("\nOutput files:")
    print("  final_side_by_side.png - First | Second (aligned)")
    print("  final_overlay.png - Red-cyan overlay (gray=match, color=diff)")
    print("  final_diff_10x.png - Difference amplified 10x")
    print("  final_signed.png - Red=first lighter, Blue=second lighter")
    print("  final_diff_text_only.png - Differences in text areas only")
    print("  final_heatmap.png - Heatmap of differences")
    print("  final_blend.png - 50% blend")
    print("  final_frame1.png, final_frame2.png - For flicker comparison")

if __name__ == "__main__":
    main()
