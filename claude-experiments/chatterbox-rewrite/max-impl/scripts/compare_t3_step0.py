"""Compare Mojo and upstream step-0 T3 logits."""
import numpy as np

upstream = np.load("/tmp/t3_dump/upstream_logits.npz")
V = upstream["logits_cond"].shape[-1]

mojo_cond = np.fromfile("/tmp/t3_dump/mojo_step0_cond.bin", dtype=np.float32)
mojo_unc  = np.fromfile("/tmp/t3_dump/mojo_step0_uncond.bin", dtype=np.float32)
mojo_cfg  = np.fromfile("/tmp/t3_dump/mojo_step0_cfg.bin", dtype=np.float32)

print(f"V = {V}, Mojo cond len = {mojo_cond.size}")
assert mojo_cond.size == V

def stats(name, mojo, up):
    diff = mojo - up
    rel_l2 = np.linalg.norm(diff) / max(np.linalg.norm(up), 1e-8)
    max_abs = np.abs(diff).max()
    cos = np.dot(mojo, up) / (np.linalg.norm(mojo) * np.linalg.norm(up) + 1e-8)
    print(f"  {name:8s}: rel_l2={rel_l2:.6e}  max_abs={max_abs:.4f}  cos_sim={cos:.6f}")
    print(f"           upstream top-5: {np.argsort(up)[-5:][::-1].tolist()}")
    print(f"           mojo     top-5: {np.argsort(mojo)[-5:][::-1].tolist()}")
    print(f"           upstream max={up.max():.3f} min={up.min():.3f}")
    print(f"           mojo     max={mojo.max():.3f} min={mojo.min():.3f}")

print("=== STEP 0 ===")
stats("cond", mojo_cond, upstream["logits_cond"][0])
stats("uncond", mojo_unc, upstream["logits_uncond"][0])
stats("cfg", mojo_cfg, upstream["logits_cfg"][0])
