#!/usr/bin/env python3
"""Compare font rendering between two images with better visualization."""

import cv2
import numpy as np

def load_and_grayscale(path):
    """Load image and convert to grayscale."""
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray

def normalize_to_binary_like(gray):
    """Normalize grayscale focusing on text edges."""
    # Invert if dark background
    mean_val = np.mean(gray)
    if mean_val < 128:
        gray = 255 - gray

    # Apply adaptive thresholding to extract text shape
    return gray

def align_images_subpixel(img1, img2):
    """Align img2 to img1 using phase correlation with subpixel accuracy."""
    h1, w1 = img1.shape
    h2, w2 = img2.shape

    h = max(h1, h2)
    w = max(w1, w2)

    padded1 = np.zeros((h, w), dtype=np.float32)
    padded2 = np.zeros((h, w), dtype=np.float32)

    padded1[:h1, :w1] = img1.astype(np.float32)
    padded2[:h2, :w2] = img2.astype(np.float32)

    shift, response = cv2.phaseCorrelate(padded1, padded2)
    dx, dy = shift
    print(f"Detected shift: dx={dx:.3f}, dy={dy:.3f}")

    # Use higher precision transformation
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    aligned = cv2.warpAffine(img2.astype(np.float32), M, (w1, h1),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    return img1, aligned.astype(np.uint8)

def main():
    print("Loading images...")
    img1_color, gray1 = load_and_grayscale("first.png")
    img2_color, gray2 = load_and_grayscale("second.png")

    print(f"Image 1: {gray1.shape}, Image 2: {gray2.shape}")

    # Normalize
    norm1 = normalize_to_binary_like(gray1)
    norm2 = normalize_to_binary_like(gray2)

    # Align
    print("Aligning...")
    aligned1, aligned2 = align_images_subpixel(norm1, norm2)

    # Compute difference
    diff = cv2.absdiff(aligned1, aligned2)

    # Create overlay visualization
    h, w = aligned1.shape

    # Method 1: Red-Cyan overlay (view with red-cyan glasses would show 3D)
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    overlay[:, :, 2] = aligned1  # Red channel = first image
    overlay[:, :, 1] = aligned2  # Green channel = second image
    overlay[:, :, 0] = aligned2  # Blue channel = second image (cyan)

    # Method 2: Difference heatmap
    diff_amplified = np.clip(diff * 50, 0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(diff_amplified, cv2.COLORMAP_JET)

    # Method 3: XOR-like visualization (shows any difference as white)
    xor_viz = np.clip(diff * 127, 0, 255).astype(np.uint8)

    # Method 4: Signed difference (red = first stronger, blue = second)
    signed_diff = aligned1.astype(np.int16) - aligned2.astype(np.int16)
    diff_signed_color = np.zeros((h, w, 3), dtype=np.uint8)
    # Red where first image is lighter (text lighter in first)
    diff_signed_color[:, :, 2] = np.clip(signed_diff * 10, 0, 255).astype(np.uint8)
    # Blue where second image is lighter
    diff_signed_color[:, :, 0] = np.clip(-signed_diff * 10, 0, 255).astype(np.uint8)

    # Method 5: Blend with 50% opacity
    blend = cv2.addWeighted(
        cv2.cvtColor(aligned1, cv2.COLOR_GRAY2BGR), 0.5,
        cv2.cvtColor(aligned2, cv2.COLOR_GRAY2BGR), 0.5, 0
    )

    # Method 6: Checkerboard composite
    checker = np.zeros((h, w), dtype=np.uint8)
    checker_size = 20
    for i in range(0, h, checker_size):
        for j in range(0, w, checker_size):
            if ((i // checker_size) + (j // checker_size)) % 2 == 0:
                checker[i:i+checker_size, j:j+checker_size] = aligned1[i:i+checker_size, j:j+checker_size]
            else:
                checker[i:i+checker_size, j:j+checker_size] = aligned2[i:i+checker_size, j:j+checker_size]

    # Save all visualizations
    print("Saving visualizations...")
    cv2.imwrite("viz_aligned1.png", aligned1)
    cv2.imwrite("viz_aligned2.png", aligned2)
    cv2.imwrite("viz_red_cyan_overlay.png", overlay)
    cv2.imwrite("viz_diff_heatmap.png", heatmap)
    cv2.imwrite("viz_diff_xor.png", xor_viz)
    cv2.imwrite("viz_diff_signed.png", diff_signed_color)
    cv2.imwrite("viz_blend_50.png", blend)
    cv2.imwrite("viz_checkerboard.png", checker)

    # Statistics
    print(f"\nDifference statistics:")
    print(f"  Mean: {np.mean(diff):.3f}")
    print(f"  Std:  {np.std(diff):.3f}")
    print(f"  Max:  {np.max(diff)}")
    print(f"  Pixels > 1: {np.sum(diff > 1)} ({100*np.sum(diff > 1)/diff.size:.2f}%)")
    print(f"  Pixels > 5: {np.sum(diff > 5)} ({100*np.sum(diff > 5)/diff.size:.2f}%)")

    # Show where differences occur
    if np.max(diff) > 0:
        diff_locations = np.where(diff > 1)
        if len(diff_locations[0]) > 0:
            print(f"\nDifference regions (y, x):")
            # Group by approximate location
            y_coords = diff_locations[0]
            x_coords = diff_locations[1]
            print(f"  Y range: {y_coords.min()} - {y_coords.max()}")
            print(f"  X range: {x_coords.min()} - {x_coords.max()}")

    print("\nOutput files:")
    print("  viz_red_cyan_overlay.png - Red=first, Cyan=second (differences show color fringing)")
    print("  viz_diff_heatmap.png - Heat map of differences (blue=none, red=large)")
    print("  viz_diff_signed.png - Red=first lighter, Blue=second lighter")
    print("  viz_blend_50.png - 50% blend (ghosting shows misalignment)")
    print("  viz_checkerboard.png - Checkerboard pattern for direct comparison")

if __name__ == "__main__":
    main()
