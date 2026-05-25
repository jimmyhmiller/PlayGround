"""Compare Mojo and upstream per-layer hidden states to localize divergence."""
import numpy as np

# Layer -1 is input embedding; layers 0..29 are post-block outputs.
# Upstream returns 31 entries (input + 30 blocks); we save them with labels -1..29.
S = 46          # prefix len (T_COND + T_TEXT + T_BOS = 34+10+2)
D = 1024

for L in range(-1, 30):
    up_path = f"/tmp/t3_dump/upstream_layer_{L}.npy"
    mj_path = f"/tmp/t3_dump/mojo_layer_{L}.bin"
    try:
        up = np.load(up_path).reshape(-1)
        mj = np.fromfile(mj_path, dtype=np.float32)
    except FileNotFoundError as e:
        print(f"L={L}: MISSING ({e})")
        continue
    if up.size != mj.size:
        print(f"L={L}: size mismatch up={up.size} mj={mj.size}")
        continue
    diff = up - mj
    rel_l2 = np.linalg.norm(diff) / max(np.linalg.norm(up), 1e-8)
    max_abs = np.abs(diff).max()
    cos = np.dot(up, mj) / (np.linalg.norm(up) * np.linalg.norm(mj) + 1e-8)
    print(f"L={L:>2}: rel_l2={rel_l2:.4e}  max_abs={max_abs:.4e}  cos_sim={cos:.6f}")
