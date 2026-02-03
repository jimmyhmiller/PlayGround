#!/usr/bin/env python3
"""Compare font rendering using feature-based alignment."""

import cv2
import numpy as np

def load_and_process(path):
    """Load image and convert to normalized grayscale."""
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Normalize: make text dark on light background
    if np.mean(gray) < 128:
        gray = 255 - gray

    return img, gray

def align_using_features(img1, img2):
    """Use ORB feature matching for alignment."""
    # Create ORB detector
    orb = cv2.ORB_create(nfeatures=5000)

    # Find keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    print(f"Found {len(kp1)} and {len(kp2)} keypoints")

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        print("Not enough features, using simple alignment")
        return None, None

    # Match features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Use top matches
    good_matches = matches[:min(50, len(matches))]
    print(f"Using {len(good_matches)} matches")

    if len(good_matches) < 4:
        return None, None

    # Extract matched points
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Find homography
    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)

    if H is None:
        return None, None

    # Warp img2 to align with img1
    h, w = img1.shape
    aligned = cv2.warpPerspective(img2, H, (w, h), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=255)

    return aligned, H

def analyze_row_by_row(img1, img2, num_rows=4):
    """Analyze differences row by row."""
    h1, w1 = img1.shape
    h2, w2 = img2.shape

    row_height1 = h1 // num_rows
    row_height2 = h2 // num_rows

    print(f"\nRow-by-row analysis (row height: {row_height1} vs {row_height2}):")

    results = []
    for i in range(num_rows):
        row1 = img1[i*row_height1:(i+1)*row_height1, :]
        row2 = img2[i*row_height2:(i+1)*row_height2, :]

        # Resize row2 to match row1 height
        if row1.shape[0] != row2.shape[0]:
            row2 = cv2.resize(row2, (row2.shape[1], row1.shape[0]))

        # Align this row
        shift, _ = cv2.phaseCorrelate(row1.astype(np.float32), row2.astype(np.float32))

        # Apply shift
        M = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
        row2_aligned = cv2.warpAffine(row2, M, (row1.shape[1], row1.shape[0]))

        diff = cv2.absdiff(row1, row2_aligned)

        results.append({
            'row': i,
            'shift_x': shift[0],
            'shift_y': shift[1],
            'mean_diff': np.mean(diff),
            'max_diff': np.max(diff),
            'row1': row1,
            'row2_aligned': row2_aligned,
            'diff': diff
        })

        print(f"  Row {i}: shift=({shift[0]:.2f}, {shift[1]:.2f}), "
              f"mean_diff={np.mean(diff):.2f}, max_diff={np.max(diff)}")

    return results

def main():
    print("Loading images...")
    img1_color, gray1 = load_and_process("first.png")
    img2_color, gray2 = load_and_process("second.png")

    print(f"Image 1: {gray1.shape}")
    print(f"Image 2: {gray2.shape}")

    # Try feature-based alignment
    print("\nAttempting feature-based alignment...")
    aligned2, H = align_using_features(gray1, gray2)

    if aligned2 is not None:
        diff = cv2.absdiff(gray1, aligned2)
        print(f"Feature alignment: mean_diff={np.mean(diff):.2f}, max_diff={np.max(diff)}")

        # Create visualizations
        # Red-cyan overlay
        overlay = np.zeros((*gray1.shape, 3), dtype=np.uint8)
        overlay[:, :, 2] = gray1
        overlay[:, :, 1] = aligned2
        overlay[:, :, 0] = aligned2
        cv2.imwrite("feature_overlay.png", overlay)

        # Difference amplified
        diff_amp = np.clip(diff * 10, 0, 255).astype(np.uint8)
        cv2.imwrite("feature_diff_amp.png", diff_amp)

        # Signed difference
        signed = gray1.astype(np.int16) - aligned2.astype(np.int16)
        signed_viz = np.zeros((*gray1.shape, 3), dtype=np.uint8)
        signed_viz[:, :, 2] = np.clip(signed * 5, 0, 255).astype(np.uint8)  # Red = first lighter
        signed_viz[:, :, 0] = np.clip(-signed * 5, 0, 255).astype(np.uint8)  # Blue = second lighter
        cv2.imwrite("feature_signed.png", signed_viz)

        cv2.imwrite("feature_aligned1.png", gray1)
        cv2.imwrite("feature_aligned2.png", aligned2)

    # Analyze row by row
    rows = analyze_row_by_row(gray1, gray2)

    # Create row-by-row visualization
    all_diffs = np.vstack([r['diff'] for r in rows])
    all_diffs_amp = np.clip(all_diffs * 10, 0, 255).astype(np.uint8)
    cv2.imwrite("row_diff_amplified.png", all_diffs_amp)

    # Row overlay
    row_overlay = []
    for r in rows:
        overlay = np.zeros((*r['row1'].shape, 3), dtype=np.uint8)
        overlay[:, :, 2] = r['row1']
        overlay[:, :, 1] = r['row2_aligned']
        overlay[:, :, 0] = r['row2_aligned']
        row_overlay.append(overlay)

    combined_overlay = np.vstack(row_overlay)
    cv2.imwrite("row_overlay.png", combined_overlay)

    print("\nOutput files:")
    print("  feature_overlay.png - Feature-aligned red-cyan overlay")
    print("  feature_diff_amp.png - Feature-aligned difference (10x)")
    print("  feature_signed.png - Red=first lighter, Blue=second lighter")
    print("  row_overlay.png - Row-by-row aligned overlay")
    print("  row_diff_amplified.png - Row-by-row differences (10x)")

if __name__ == "__main__":
    main()
