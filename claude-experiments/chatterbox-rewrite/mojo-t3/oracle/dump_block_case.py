"""
Single transformer block parity case using real T3 layer-0 weights.

LlamaDecoderLayer.forward:
  residual = x
  x = input_layernorm(x)
  x_attn = self_attn(x, cos, sin, mask)
  x = residual + x_attn
  residual = x
  x = post_attention_layernorm(x)
  x_mlp = mlp(x)
  x = residual + x_mlp

Inside self_attn:
  q = q_proj(x).view(B,S,H,D).transpose(1,2)  → (B,H,S,D)
  k, v same
  q, k = apply_rotary_pos_emb(q, k, cos, sin)
  attn_out = eager_sdpa(q, k, v, mask) → (B,H,S,D)
  attn_out = attn_out.transpose(1,2).contiguous().view(B,S,H*D)
  attn_out = o_proj(attn_out)

We dump all weights pre-transposed to (in, out) shape so the Mojo side does
straight matmul. Cos/sin and causal mask come from the same setup as the
RoPE/SDPA fixtures.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaRotaryEmbedding,
)

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures"
WEIGHTS = FIX / "llama_t3_weights_fp32.safetensors"
ACTS = FIX / "activations_fp32.npz"
OUT = FIX / "block"
OUT.mkdir(parents=True, exist_ok=True)

# T3 Llama config (subset).
HIDDEN = 1024
N_HEADS = 16
HEAD_DIM = 64
INTERMEDIATE = 4096
SEQ = 16
BATCH = 1


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
    finfo = torch.finfo(dtype)
    mask = torch.zeros(seq, seq, dtype=dtype)
    mask.masked_fill_(torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1), finfo.min)
    return mask  # (S, S); HF will broadcast / unsqueeze inside


def build_block(state: dict, dtype: torch.dtype):
    cfg = LlamaConfig(
        vocab_size=8, hidden_size=HIDDEN, intermediate_size=INTERMEDIATE,
        num_attention_heads=N_HEADS, num_key_value_heads=N_HEADS, head_dim=HEAD_DIM,
        num_hidden_layers=30, rms_norm_eps=1e-5,
        rope_theta=500_000.0,
        rope_scaling=dict(
            factor=8.0, high_freq_factor=4.0, low_freq_factor=1.0,
            original_max_position_embeddings=8192, rope_type="llama3",
        ),
        max_position_embeddings=131072,
        mlp_bias=False, attention_bias=False, hidden_act="silu",
        attn_implementation="eager",  # match our oracle SDPA path
    )
    block = LlamaDecoderLayer(cfg, layer_idx=0)
    pfx = "layers.0."
    block.input_layernorm.weight.data = state[pfx + "input_layernorm.weight"].to(dtype)
    block.post_attention_layernorm.weight.data = state[pfx + "post_attention_layernorm.weight"].to(dtype)
    block.self_attn.q_proj.weight.data = state[pfx + "self_attn.q_proj.weight"].to(dtype)
    block.self_attn.k_proj.weight.data = state[pfx + "self_attn.k_proj.weight"].to(dtype)
    block.self_attn.v_proj.weight.data = state[pfx + "self_attn.v_proj.weight"].to(dtype)
    block.self_attn.o_proj.weight.data = state[pfx + "self_attn.o_proj.weight"].to(dtype)
    block.mlp.gate_proj.weight.data = state[pfx + "mlp.gate_proj.weight"].to(dtype)
    block.mlp.up_proj.weight.data = state[pfx + "mlp.up_proj.weight"].to(dtype)
    block.mlp.down_proj.weight.data = state[pfx + "mlp.down_proj.weight"].to(dtype)
    return block.to(dtype=dtype, device="cpu").eval(), cfg


def main() -> None:
    state = load_file(str(WEIGHTS))
    acts = np.load(ACTS)

    # Use input_embeds (B=1, S=16, H=1024) as the block input.
    x_fp32 = torch.from_numpy(acts["input_embeds"].astype(np.float32))  # (1, 16, 1024)

    positions = torch.arange(SEQ, dtype=torch.long).unsqueeze(0)

    # ---- fp32 ----
    block_fp32, cfg = build_block(state, torch.float32)
    rope_fp32 = LlamaRotaryEmbedding(cfg).eval()
    with torch.no_grad():
        cos_fp32, sin_fp32 = rope_fp32(x_fp32, positions)
        mask_fp32 = make_causal_mask(SEQ, torch.float32).reshape(1, 1, SEQ, SEQ)
        out_fp32 = block_fp32(
            x_fp32,
            attention_mask=mask_fp32,
            position_ids=positions,
            position_embeddings=(cos_fp32, sin_fp32),
        )

    # Pre-transpose all projection weights so Mojo can do straight A @ B.
    qw = state["layers.0.self_attn.q_proj.weight"].t().contiguous().float()  # (1024, 1024)
    kw = state["layers.0.self_attn.k_proj.weight"].t().contiguous().float()
    vw = state["layers.0.self_attn.v_proj.weight"].t().contiguous().float()
    ow = state["layers.0.self_attn.o_proj.weight"].t().contiguous().float()
    in_norm = state["layers.0.input_layernorm.weight"].float()
    post_norm = state["layers.0.post_attention_layernorm.weight"].float()
    gate_w = state["layers.0.mlp.gate_proj.weight"].t().contiguous().float()
    up_w = state["layers.0.mlp.up_proj.weight"].t().contiguous().float()
    down_w = state["layers.0.mlp.down_proj.weight"].t().contiguous().float()

    # Inputs.
    write_tensor(OUT / "x_fp32.bin", x_fp32.numpy().astype(np.float32))
    write_tensor(OUT / "cos_fp32.bin", cos_fp32.numpy().astype(np.float32))   # (1, 16, 64)
    write_tensor(OUT / "sin_fp32.bin", sin_fp32.numpy().astype(np.float32))
    write_tensor(OUT / "mask_fp32.bin",
                 mask_fp32.reshape(SEQ, SEQ).numpy().astype(np.float32))

    # Weights.
    write_tensor(OUT / "in_norm_fp32.bin", in_norm.numpy().astype(np.float32))
    write_tensor(OUT / "post_norm_fp32.bin", post_norm.numpy().astype(np.float32))
    write_tensor(OUT / "qw_fp32.bin", qw.numpy().astype(np.float32))
    write_tensor(OUT / "kw_fp32.bin", kw.numpy().astype(np.float32))
    write_tensor(OUT / "vw_fp32.bin", vw.numpy().astype(np.float32))
    write_tensor(OUT / "ow_fp32.bin", ow.numpy().astype(np.float32))
    write_tensor(OUT / "gate_w_fp32.bin", gate_w.numpy().astype(np.float32))
    write_tensor(OUT / "up_w_fp32.bin", up_w.numpy().astype(np.float32))
    write_tensor(OUT / "down_w_fp32.bin", down_w.numpy().astype(np.float32))

    # Output.
    write_tensor(OUT / "expected_fp32.bin", out_fp32.numpy().astype(np.float32))

    print(f"[fp32] x         {tuple(x_fp32.shape)}")
    print(f"[fp32] expected  {tuple(out_fp32.shape)}")
    print(f"  x[0,0,:4]    = {x_fp32[0,0,:4].numpy()}")
    print(f"  out[0,0,:4]  = {out_fp32[0,0,:4].numpy()}")

    # ---- bf16 ----
    block_bf16, _ = build_block(state, torch.bfloat16)
    rope_bf16 = LlamaRotaryEmbedding(cfg).eval()  # bf16 inputs cast cos/sin internally
    x_bf16 = x_fp32.to(torch.bfloat16)
    with torch.no_grad():
        cos_bf16, sin_bf16 = rope_bf16(x_bf16, positions)
        mask_bf16 = make_causal_mask(SEQ, torch.bfloat16).reshape(1, 1, SEQ, SEQ)
        out_bf16 = block_bf16(
            x_bf16,
            attention_mask=mask_bf16,
            position_ids=positions,
            position_embeddings=(cos_bf16, sin_bf16),
        )

    write_tensor(OUT / "x_bf16.bin",
                 fp32_to_bf16_bits(x_bf16.float().numpy()).reshape(x_fp32.shape))
    write_tensor(OUT / "cos_bf16.bin",
                 fp32_to_bf16_bits(cos_bf16.float().numpy()).reshape(cos_fp32.shape))
    write_tensor(OUT / "sin_bf16.bin",
                 fp32_to_bf16_bits(sin_bf16.float().numpy()).reshape(sin_fp32.shape))
    write_tensor(OUT / "mask_bf16.bin",
                 fp32_to_bf16_bits(mask_bf16.float().reshape(SEQ, SEQ).numpy()).reshape(SEQ, SEQ))

    write_tensor(OUT / "in_norm_bf16.bin",
                 fp32_to_bf16_bits(in_norm.numpy()).reshape(in_norm.shape))
    write_tensor(OUT / "post_norm_bf16.bin",
                 fp32_to_bf16_bits(post_norm.numpy()).reshape(post_norm.shape))
    write_tensor(OUT / "qw_bf16.bin",
                 fp32_to_bf16_bits(qw.numpy()).reshape(qw.shape))
    write_tensor(OUT / "kw_bf16.bin",
                 fp32_to_bf16_bits(kw.numpy()).reshape(kw.shape))
    write_tensor(OUT / "vw_bf16.bin",
                 fp32_to_bf16_bits(vw.numpy()).reshape(vw.shape))
    write_tensor(OUT / "ow_bf16.bin",
                 fp32_to_bf16_bits(ow.numpy()).reshape(ow.shape))
    write_tensor(OUT / "gate_w_bf16.bin",
                 fp32_to_bf16_bits(gate_w.numpy()).reshape(gate_w.shape))
    write_tensor(OUT / "up_w_bf16.bin",
                 fp32_to_bf16_bits(up_w.numpy()).reshape(up_w.shape))
    write_tensor(OUT / "down_w_bf16.bin",
                 fp32_to_bf16_bits(down_w.numpy()).reshape(down_w.shape))
    write_tensor(OUT / "expected_bf16.bin",
                 fp32_to_bf16_bits(out_bf16.float().numpy()).reshape(out_fp32.shape))

    print(f"[bf16] out[0,0,:4] (decoded) = {out_bf16.float()[0,0,:4].numpy()}")


if __name__ == "__main__":
    main()
