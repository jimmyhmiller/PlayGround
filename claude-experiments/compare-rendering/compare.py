#!/usr/bin/env python3
"""Compare font rendering between two images, ignoring color differences."""

import cv2
import numpy as np
from PIL import Image

def load_and_grayscale(path):
    """Load image and convert to grayscale."""
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray

def normalize_contrast(gray):
    """Normalize grayscale image to full contrast range."""
    # Invert if the text is light on dark (so text becomes dark on light)
    # This helps with comparison when colors are inverted
    mean_val = np.mean(gray)
    if mean_val < 128:
        # Dark background, light text - invert
        gray = 255 - gray

    # Normalize to full range
    min_val, max_val = gray.min(), gray.max()
    if max_val > min_val:
        normalized = ((gray - min_val) * 255 / (max_val - min_val)).astype(np.uint8)
    else:
        normalized = gray
    return normalized

def align_images(img1, img2):
    """Align img2 to img1 using phase correlation."""
    # Ensure same size by padding
    h1, w1 = img1.shape
    h2, w2 = img2.shape

    # Use the larger dimensions
    h = max(h1, h2)
    w = max(w1, w2)

    # Pad images to same size
    padded1 = np.zeros((h, w), dtype=np.float32)
    padded2 = np.zeros((h, w), dtype=np.float32)

    padded1[:h1, :w1] = img1.astype(np.float32)
    padded2[:h2, :w2] = img2.astype(np.float32)

    # Phase correlation to find shift
    shift, response = cv2.phaseCorrelate(padded1, padded2)

    dx, dy = shift
    print(f"Detected shift: dx={dx:.2f}, dy={dy:.2f}")

    # Create translation matrix
    M = np.float32([[1, 0, dx], [0, 1, dy]])

    # Apply translation to img2
    aligned = cv2.warpAffine(padded2.astype(np.uint8), M, (w, h))

    return padded1[:h1, :w1].astype(np.uint8), aligned[:h1, :w1]

def create_difference_visualization(img1, img2, aligned1, aligned2):
    """Create visualizations of the differences."""
    # Compute absolute difference
    diff = cv2.absdiff(aligned1, aligned2)

    # Amplify differences for visibility
    diff_amplified = np.clip(diff * 5, 0, 255).astype(np.uint8)

    # Threshold to show only significant differences
    _, diff_thresh = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)

    # Create a color overlay showing differences
    # Red = first image stronger, Blue = second image stronger
    h, w = aligned1.shape
    diff_color = np.zeros((h, w, 3), dtype=np.uint8)

    # Where first image is darker (stronger rendering)
    first_stronger = aligned1 < aligned2
    # Where second image is darker (stronger rendering)
    second_stronger = aligned2 < aligned1

    diff_magnitude = np.abs(aligned1.astype(np.int16) - aligned2.astype(np.int16))

    # Red channel for first image stronger
    diff_color[:, :, 2] = np.where(first_stronger, np.clip(diff_magnitude * 3, 0, 255), 0).astype(np.uint8)
    # Blue channel for second image stronger
    diff_color[:, :, 0] = np.where(second_stronger, np.clip(diff_magnitude * 3, 0, 255), 0).astype(np.uint8)

    return diff, diff_amplified, diff_thresh, diff_color

def main():
    # Load images
    print("Loading images...")
    img1_color, gray1 = load_and_grayscale("first.png")
    img2_color, gray2 = load_and_grayscale("second.png")

    print(f"Image 1 size: {gray1.shape}")
    print(f"Image 2 size: {gray2.shape}")

    # Normalize contrast
    print("Normalizing contrast...")
    norm1 = normalize_contrast(gray1)
    norm2 = normalize_contrast(gray2)

    # Align images
    print("Aligning images...")
    aligned1, aligned2 = align_images(norm1, norm2)

    # Create difference visualizations
    print("Computing differences...")
    diff, diff_amp, diff_thresh, diff_color = create_difference_visualization(
        img1_color, img2_color, aligned1, aligned2
    )

    # Save results
    print("Saving results...")
    cv2.imwrite("aligned_first.png", aligned1)
    cv2.imwrite("aligned_second.png", aligned2)
    cv2.imwrite("difference_raw.png", diff)
    cv2.imwrite("difference_amplified.png", diff_amp)
    cv2.imwrite("difference_threshold.png", diff_thresh)
    cv2.imwrite("difference_color.png", diff_color)

    # Create side-by-side comparison
    h = max(aligned1.shape[0], aligned2.shape[0])
    w1, w2 = aligned1.shape[1], aligned2.shape[1]

    comparison = np.zeros((h, w1 + w2 + 10), dtype=np.uint8)
    comparison[:aligned1.shape[0], :w1] = aligned1
    comparison[:aligned2.shape[0], w1+10:] = aligned2
    cv2.imwrite("side_by_side.png", comparison)

    # Stats
    total_pixels = aligned1.shape[0] * aligned1.shape[1]
    different_pixels = np.sum(diff > 10)
    print(f"\nResults:")
    print(f"  Total pixels compared: {total_pixels}")
    print(f"  Pixels with difference > 10: {different_pixels} ({100*different_pixels/total_pixels:.2f}%)")
    print(f"  Mean difference: {np.mean(diff):.2f}")
    print(f"  Max difference: {np.max(diff)}")

    print("\nOutput files:")
    print("  - aligned_first.png: First image normalized")
    print("  - aligned_second.png: Second image aligned & normalized")
    print("  - difference_raw.png: Raw pixel differences")
    print("  - difference_amplified.png: Differences amplified 5x")
    print("  - difference_threshold.png: Binary diff (threshold=10)")
    print("  - difference_color.png: Color-coded (red=first stronger, blue=second)")
    print("  - side_by_side.png: Side-by-side comparison")

if __name__ == "__main__":
    main()
