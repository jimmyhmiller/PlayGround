"""
Decode-step SDPA parity case.

Models one autoregressive decode step on top of a partial KV cache:
  - History K/V of length T_HIST (the "cache" we've accumulated so far).
  - One new (q_new, k_new, v_new) — the token currently being processed.

The full K/V seen by attention is the cat of (history, new) along the seq dim.
Causal masking is trivial: the single new query may attend to every position
including itself, so no mask is needed.

We dump:
  q_new        (B, H, 1, D)
  k_hist       (B, H, T_HIST, D)
  v_hist       (B, H, T_HIST, D)
  k_new        (B, H, 1, D)
  v_new        (B, H, 1, D)
  logits       (B, H, 1, T_HIST + 1)
  probs        (B, H, 1, T_HIST + 1)
  expected     (B, H, 1, D)      eager-attention output for the new token

Binary format identical to dump_sdpa_case.py.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures"
OUT = FIX / "decode_sdpa"
OUT.mkdir(parents=True, exist_ok=True)

BATCH = 1
N_HEADS = 4
T_HIST = 15
HEAD_DIM = 64
SCALE = HEAD_DIM ** -0.5


def write_tensor(path: Path, arr: np.ndarray) -> None:
    if arr.dtype == np.float32:
        tag = 0
        raw = arr.astype(np.float32, copy=False).tobytes()
    elif arr.dtype == np.uint16:
        tag = 1
        raw = arr.astype(np.uint16, copy=False).tobytes()
    else:
        raise TypeError(f"unsupported dtype {arr.dtype}")
    with path.open("wb") as f:
        f.write(struct.pack("<q", len(arr.shape)))
        f.write(struct.pack(f"<{len(arr.shape)}q", *arr.shape))
        f.write(struct.pack("<i", tag))
        f.write(raw)


def fp32_to_bf16_bits(arr_fp32: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(np.ascontiguousarray(arr_fp32)).to(torch.bfloat16)
    return t.view(torch.int16).cpu().numpy().astype(np.uint16, copy=False)


def eager_decode(q_new, k_hist, v_hist, k_new, v_new, scale: float):
    """Reference one-step decode SDPA. Trivial mask (full attend)."""
    # K/V seen by the new query = cat of [hist, new] along seq dim.
    k_all = torch.cat([k_hist, k_new], dim=2)  # (B, H, T_HIST+1, D)
    v_all = torch.cat([v_hist, v_new], dim=2)

    logits = torch.matmul(q_new, k_all.transpose(-1, -2)) * scale  # (B, H, 1, T_HIST+1)
    probs = torch.softmax(logits.float(), dim=-1).to(q_new.dtype)
    out = torch.matmul(probs, v_all)  # (B, H, 1, D)
    return out, logits, probs


def main() -> None:
    g = torch.Generator(device="cpu").manual_seed(0xC0FFEE)
    q_new = torch.randn(BATCH, N_HEADS, 1, HEAD_DIM, generator=g, dtype=torch.float32)
    k_hist = torch.randn(BATCH, N_HEADS, T_HIST, HEAD_DIM, generator=g, dtype=torch.float32)
    v_hist = torch.randn(BATCH, N_HEADS, T_HIST, HEAD_DIM, generator=g, dtype=torch.float32)
    k_new = torch.randn(BATCH, N_HEADS, 1, HEAD_DIM, generator=g, dtype=torch.float32)
    v_new = torch.randn(BATCH, N_HEADS, 1, HEAD_DIM, generator=g, dtype=torch.float32)

    # ---- fp32 ----
    out_fp32, logits_fp32, probs_fp32 = eager_decode(
        q_new, k_hist, v_hist, k_new, v_new, SCALE
    )

    write_tensor(OUT / "q_new_fp32.bin", q_new.numpy().astype(np.float32))
    write_tensor(OUT / "k_hist_fp32.bin", k_hist.numpy().astype(np.float32))
    write_tensor(OUT / "v_hist_fp32.bin", v_hist.numpy().astype(np.float32))
    write_tensor(OUT / "k_new_fp32.bin", k_new.numpy().astype(np.float32))
    write_tensor(OUT / "v_new_fp32.bin", v_new.numpy().astype(np.float32))
    write_tensor(OUT / "logits_fp32.bin", logits_fp32.numpy().astype(np.float32))
    write_tensor(OUT / "probs_fp32.bin", probs_fp32.numpy().astype(np.float32))
    write_tensor(OUT / "expected_fp32.bin", out_fp32.numpy().astype(np.float32))

    print(f"[fp32] shapes — q_new:{tuple(q_new.shape)} k_hist:{tuple(k_hist.shape)} out:{tuple(out_fp32.shape)}")
    print(f"  q_new[0,0,0,:4]   = {q_new[0,0,0,:4].numpy()}")
    print(f"  out_fp32[0,0,0,:4] = {out_fp32[0,0,0,:4].numpy()}")
    print(f"  probs row 0 sum   = {probs_fp32[0,0,0].sum().item():.6f}")
    print(f"  probs row 0 first/last = {probs_fp32[0,0,0,0].item():.4e} / {probs_fp32[0,0,0,-1].item():.4e}")

    # ---- bf16 ----
    q_new_bf = q_new.to(torch.bfloat16)
    k_hist_bf = k_hist.to(torch.bfloat16)
    v_hist_bf = v_hist.to(torch.bfloat16)
    k_new_bf = k_new.to(torch.bfloat16)
    v_new_bf = v_new.to(torch.bfloat16)
    out_bf16, _, _ = eager_decode(
        q_new_bf, k_hist_bf, v_hist_bf, k_new_bf, v_new_bf, SCALE
    )

    write_tensor(OUT / "q_new_bf16.bin",
                 fp32_to_bf16_bits(q_new_bf.float().numpy()).reshape(q_new.shape))
    write_tensor(OUT / "k_hist_bf16.bin",
                 fp32_to_bf16_bits(k_hist_bf.float().numpy()).reshape(k_hist.shape))
    write_tensor(OUT / "v_hist_bf16.bin",
                 fp32_to_bf16_bits(v_hist_bf.float().numpy()).reshape(v_hist.shape))
    write_tensor(OUT / "k_new_bf16.bin",
                 fp32_to_bf16_bits(k_new_bf.float().numpy()).reshape(k_new.shape))
    write_tensor(OUT / "v_new_bf16.bin",
                 fp32_to_bf16_bits(v_new_bf.float().numpy()).reshape(v_new.shape))
    write_tensor(OUT / "expected_bf16.bin",
                 fp32_to_bf16_bits(out_bf16.float().numpy()).reshape(out_fp32.shape))
    print(f"[bf16] out[0,0,0,:4] (decoded) = {out_bf16.float()[0,0,0,:4].numpy()}")


if __name__ == "__main__":
    main()
