"""Compare Mojo and upstream layer-0 intermediates."""
import numpy as np

# All ours are cond batch only (b=2 but we wrote b=2 contiguous → first half is cond).
# Need to slice to cond.
SHAPES = {
    "xnorm":     (2, 46, 1024),
    "qlin":      (2, 46, 1024),
    "klin":      (2, 46, 1024),
    "vlin":      (2, 46, 1024),
    "qrope":     (2, 16, 46, 64),
    "krope":     (2, 16, 46, 64),
    "vperm":     (2, 16, 46, 64),
    "qklogits":  (2, 16, 46, 46),
    "attnprobs": (2, 16, 46, 46),
    "av":        (2, 16, 46, 64),
    "attnout":   (2, 46, 1024),
    "postattn":  (2, 46, 1024),
}

for tag, mj_shape in SHAPES.items():
    mj = np.fromfile(f"/tmp/t3_dump/mojo_l0_{tag}.bin", dtype=np.float32).reshape(mj_shape)
    # Take cond batch (index 0).
    mj_c = mj[0]
    up = np.load(f"/tmp/t3_dump/upstream_l0_{tag}.npy")
    # Upstream is shape (1, ...); squeeze leading batch.
    up_c = up[0]
    if mj_c.shape != up_c.shape:
        print(f"{tag}: SHAPE MISMATCH mj={mj_c.shape} up={up_c.shape}")
        continue
    diff = mj_c.flatten() - up_c.flatten()
    rel_l2 = np.linalg.norm(diff) / max(np.linalg.norm(up_c.flatten()), 1e-8)
    max_abs = np.abs(diff).max()
    cos = np.dot(mj_c.flatten(), up_c.flatten()) / (np.linalg.norm(mj_c) * np.linalg.norm(up_c) + 1e-8)
    print(f"{tag:>12s}: rel_l2={rel_l2:.4e}  max_abs={max_abs:.4e}  cos={cos:.6f}")
