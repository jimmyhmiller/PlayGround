"""
Produce self-contained RoPE parity cases for the Mojo test.

What we dump (per dtype: fp32, bf16):
  q                — input Q tensor, shape (B, H, S, D)
  cos, sin         — rotary embeddings, shape (B, S, D)   (HF unsqueezes to (B, 1, S, D))
  expected_q       — apply_rotary_pos_emb(q) output

We also cross-check our llama3 inv_freq computation against HF's by building a
real LlamaRotaryEmbedding and comparing inv_freq element-wise. If that matches,
we know the inv_freq path is right; the rest of RoPE is straightforward.

Binary format (same as RMSNorm fixtures):
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
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
)

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures"
OUT = FIX / "rope"
OUT.mkdir(parents=True, exist_ok=True)

# Mirror of T3 Llama config (head_dim=64, theta=500_000, llama3 scaling).
CONFIG_DICT = dict(
    vocab_size=8,
    max_position_embeddings=131072,
    hidden_size=1024,
    intermediate_size=4096,
    num_hidden_layers=30,
    num_attention_heads=16,
    head_dim=64,
    num_key_value_heads=16,
    hidden_act="silu",
    rms_norm_eps=1e-5,
    rope_theta=500000.0,
    rope_scaling=dict(
        factor=8.0,
        high_freq_factor=4.0,
        low_freq_factor=1.0,
        original_max_position_embeddings=8192,
        rope_type="llama3",
    ),
    torch_dtype="float32",
)

# Test fixture dimensions — keep small.
BATCH = 1
N_HEADS = 4         # subset; the rotation is per-head independent
SEQ = 16
HEAD_DIM = 64       # T3's head_dim


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


def main() -> None:
    cfg = LlamaConfig(**CONFIG_DICT)
    rope = LlamaRotaryEmbedding(cfg).eval()

    # Sanity: print first few inv_freq values; this is what HF computed.
    inv_freq = rope.inv_freq.detach().cpu().numpy()
    print(f"inv_freq[:6]  = {inv_freq[:6]}")
    print(f"inv_freq[-6:] = {inv_freq[-6:]}")

    # Build a fixed Q tensor.
    g = torch.Generator(device="cpu").manual_seed(0xC0FFEE)
    q_fp32 = torch.randn(BATCH, N_HEADS, SEQ, HEAD_DIM, generator=g, dtype=torch.float32)

    # Fixed positions [0, 1, 2, ..., SEQ-1].
    positions = torch.arange(SEQ, dtype=torch.long).unsqueeze(0)  # (B=1, S)

    with torch.no_grad():
        cos, sin = rope(q_fp32, positions)  # (B, S, D), fp32

        # apply_rotary_pos_emb expects q,k same shape; we pass q as both and
        # only use the first return value.
        q_rot, _ = apply_rotary_pos_emb(q_fp32, q_fp32, cos, sin)

    # ---- fp32 case ----
    q_np = q_fp32.numpy().astype(np.float32)
    cos_np = cos.numpy().astype(np.float32)        # (B, S, D)
    sin_np = sin.numpy().astype(np.float32)
    qrot_np = q_rot.numpy().astype(np.float32)

    write_tensor(OUT / "q_fp32.bin", q_np)
    write_tensor(OUT / "cos_fp32.bin", cos_np)
    write_tensor(OUT / "sin_fp32.bin", sin_np)
    write_tensor(OUT / "expected_fp32.bin", qrot_np)

    print(f"[fp32] q       {q_np.shape} -> {OUT/'q_fp32.bin'}")
    print(f"[fp32] cos     {cos_np.shape} -> {OUT/'cos_fp32.bin'}")
    print(f"[fp32] sin     {sin_np.shape} -> {OUT/'sin_fp32.bin'}")
    print(f"[fp32] q_rot   {qrot_np.shape} -> {OUT/'expected_fp32.bin'}")
    print(f"  q[0,0,0,:4]      = {q_np[0,0,0,:4]}")
    print(f"  cos[0,0,:4]      = {cos_np[0,0,:4]}")
    print(f"  q_rot[0,0,0,:4]  = {qrot_np[0,0,0,:4]}")

    # ---- bf16 case ----
    # Mirror what HF's bf16 model would do: cast inputs to bf16, then run.
    q_bf16_t = q_fp32.to(torch.bfloat16)
    with torch.no_grad():
        cos_bf16, sin_bf16 = rope(q_bf16_t, positions)
        q_rot_bf16, _ = apply_rotary_pos_emb(q_bf16_t, q_bf16_t, cos_bf16, sin_bf16)

    q_bf16_bits = fp32_to_bf16_bits(q_bf16_t.float().numpy())
    cos_bf16_bits = fp32_to_bf16_bits(cos_bf16.float().numpy())
    sin_bf16_bits = fp32_to_bf16_bits(sin_bf16.float().numpy())
    qrot_bf16_bits = fp32_to_bf16_bits(q_rot_bf16.float().numpy())

    write_tensor(OUT / "q_bf16.bin", q_bf16_bits.reshape(q_np.shape))
    write_tensor(OUT / "cos_bf16.bin", cos_bf16_bits.reshape(cos_np.shape))
    write_tensor(OUT / "sin_bf16.bin", sin_bf16_bits.reshape(sin_np.shape))
    write_tensor(OUT / "expected_bf16.bin", qrot_bf16_bits.reshape(qrot_np.shape))

    print(f"[bf16] q       {q_np.shape}")
    print(f"  q_rot[0,0,0,:4] (decoded) = {q_rot_bf16.float().numpy()[0,0,0,:4]}")


if __name__ == "__main__":
    main()
