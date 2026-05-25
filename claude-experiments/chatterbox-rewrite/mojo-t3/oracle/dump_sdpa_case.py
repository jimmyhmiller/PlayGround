"""
Produce self-contained SDPA parity case for the Mojo test.

Uses eager_attention_forward (the reference algorithm) as oracle, NOT
PyTorch's fused SDPA. Eager spec:
  logits = Q @ K^T * scale + causal_mask
  probs  = softmax(logits, dim=-1, dtype=fp32).to(input_dtype)
  out    = probs @ V

Shapes (T3 Llama: head_dim=64, num_heads=16, num_kv_heads=16, num_kv_groups=1):
  Q, K, V         (B, H, S, D)
  logits, probs   (B, H, S, S)
  out             (B, H, S, D)   (we keep this layout — HF transposes back to
                                   (B, S, H, D) but we test the pre-transpose
                                   value to keep the test focused on SDPA itself)

We dump fixtures for both fp32 and bf16 from the same eager pipeline.

Binary format (same as RMSNorm/RoPE):
  i64        rank
  i64[rank]  shape
  i32        dtype_tag    (0=fp32, 1=bf16-as-uint16)
  payload    raw bytes
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures"
OUT = FIX / "sdpa"
OUT.mkdir(parents=True, exist_ok=True)

BATCH = 1
N_HEADS = 4         # subset of 16; per-head independence makes this OK
SEQ = 16
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


def make_causal_mask(seq: int, dtype: torch.dtype) -> torch.Tensor:
    """Additive causal mask: 0 for allowed, -inf for masked. Shape (1,1,S,S)."""
    # HF style: float-min for masked positions (clamped to be safe in fp16/bf16).
    finfo = torch.finfo(dtype)
    mask = torch.zeros(seq, seq, dtype=dtype)
    mask.masked_fill_(torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1), finfo.min)
    return mask.unsqueeze(0).unsqueeze(0)  # (1,1,S,S)


def eager_sdpa(q, k, v, mask, scale: float):
    """Reference SDPA (matches HF eager_attention_forward)."""
    logits = torch.matmul(q, k.transpose(-1, -2)) * scale
    logits = logits + mask
    # Softmax in fp32, cast back.
    probs = torch.softmax(logits.float(), dim=-1).to(q.dtype)
    out = torch.matmul(probs, v)
    return out, logits, probs


def main() -> None:
    g = torch.Generator(device="cpu").manual_seed(0xC0FFEE)
    q_fp32 = torch.randn(BATCH, N_HEADS, SEQ, HEAD_DIM, generator=g, dtype=torch.float32)
    k_fp32 = torch.randn(BATCH, N_HEADS, SEQ, HEAD_DIM, generator=g, dtype=torch.float32)
    v_fp32 = torch.randn(BATCH, N_HEADS, SEQ, HEAD_DIM, generator=g, dtype=torch.float32)

    # ---- fp32 ----
    mask_fp32 = make_causal_mask(SEQ, torch.float32)
    out_fp32, logits_fp32, probs_fp32 = eager_sdpa(q_fp32, k_fp32, v_fp32, mask_fp32, SCALE)

    write_tensor(OUT / "q_fp32.bin", q_fp32.numpy().astype(np.float32))
    write_tensor(OUT / "k_fp32.bin", k_fp32.numpy().astype(np.float32))
    write_tensor(OUT / "v_fp32.bin", v_fp32.numpy().astype(np.float32))
    write_tensor(OUT / "mask_fp32.bin", mask_fp32.numpy().astype(np.float32).reshape(SEQ, SEQ))
    write_tensor(OUT / "expected_fp32.bin", out_fp32.numpy().astype(np.float32))
    # Intermediates for granular debugging.
    write_tensor(OUT / "logits_fp32.bin", logits_fp32.numpy().astype(np.float32))
    write_tensor(OUT / "probs_fp32.bin", probs_fp32.numpy().astype(np.float32))

    print(f"[fp32] q/k/v/out shapes : {tuple(q_fp32.shape)} {tuple(out_fp32.shape)}")
    print(f"  q[0,0,0,:4]   = {q_fp32[0,0,0,:4].numpy()}")
    print(f"  out[0,0,0,:4] = {out_fp32[0,0,0,:4].numpy()}")
    print(f"  probs[0,0,5]  (row 5, sums to 1) = {probs_fp32[0,0,5].numpy()}")
    print(f"  probs[0,0,5].sum() = {probs_fp32[0,0,5].sum().item():.6f}")

    # ---- bf16 ----
    q_bf16 = q_fp32.to(torch.bfloat16)
    k_bf16 = k_fp32.to(torch.bfloat16)
    v_bf16 = v_fp32.to(torch.bfloat16)
    mask_bf16 = make_causal_mask(SEQ, torch.bfloat16)
    out_bf16, logits_bf16, probs_bf16 = eager_sdpa(q_bf16, k_bf16, v_bf16, mask_bf16, SCALE)

    write_tensor(OUT / "q_bf16.bin", fp32_to_bf16_bits(q_bf16.float().numpy()).reshape(q_fp32.shape))
    write_tensor(OUT / "k_bf16.bin", fp32_to_bf16_bits(k_bf16.float().numpy()).reshape(k_fp32.shape))
    write_tensor(OUT / "v_bf16.bin", fp32_to_bf16_bits(v_bf16.float().numpy()).reshape(v_fp32.shape))
    # Mask in bf16: store -inf positions as the bf16 finfo.min bit pattern.
    write_tensor(OUT / "mask_bf16.bin",
                 fp32_to_bf16_bits(mask_bf16.float().numpy().reshape(SEQ, SEQ)).reshape(SEQ, SEQ))
    write_tensor(OUT / "expected_bf16.bin",
                 fp32_to_bf16_bits(out_bf16.float().numpy()).reshape(out_fp32.shape))

    print(f"[bf16] out[0,0,0,:4] (decoded) = {out_bf16.float()[0,0,0,:4].numpy()}")


if __name__ == "__main__":
    main()
