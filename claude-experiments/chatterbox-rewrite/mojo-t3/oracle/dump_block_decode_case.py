"""
Single-block decode-step parity case using real T3 layer-0 weights.

Models one autoregressive decode step on top of a 15-token prefill:
  1. Prefill: run the block on x_prefill (1, 15, 1024). Collect K/V from this
     pass — these are the "history" entries the decode step will attend to.
  2. Decode: run the same block on x_decode (1, 1, 1024) but during attention,
     concatenate the new K/V with the prefill K/V along the seq dim. The
     decode query attends to all 16 K/V positions; no causal mask is needed
     (the single query may attend to everything before it including itself).

We use a pure-tensor eager implementation here (no HF Cache class) so the
oracle is small, explicit, and exactly matches what our Mojo decode path
does step-for-step.

Outputs (fp32 + bf16):
  x_prefill        (1, 15, 1024)   — block input for the prefill pass
  x_decode         (1, 1, 1024)    — block input for the decode step
  in_norm, post_norm, qw, kw, vw, ow, gate_w, up_w, down_w   (weights)
  cos_prefill      (1, 15, 64)     — RoPE cos/sin at positions [0..14]
  sin_prefill      (1, 15, 64)
  cos_decode       (1, 1, 64)      — RoPE cos/sin at position 15
  sin_decode       (1, 1, 64)
  mask_prefill     (15, 15)        — additive causal mask for prefill
  k_hist           (1, 16, 15, 64) — K cache after prefill (BHSD)
  v_hist           (1, 16, 15, 64) — V cache after prefill (BHSD)
  expected         (1, 1, 1024)    — block output for the decode step
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures"
WEIGHTS = FIX / "llama_t3_weights_fp32.safetensors"
ACTS = FIX / "activations_fp32.npz"
OUT = FIX / "block_decode"
OUT.mkdir(parents=True, exist_ok=True)

HIDDEN = 1024
N_HEADS = 16
HEAD_DIM = 64
INTERMEDIATE = 4096
T_HIST = 15
T_TOTAL = T_HIST + 1   # 16
BATCH = 1
EPS = 1e-5
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


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Llama RMSNorm. Internal computation in fp32 to match HF exactly."""
    orig_dtype = x.dtype
    xf = x.to(torch.float32)
    var = xf.pow(2).mean(-1, keepdim=True)
    xn = xf * torch.rsqrt(var + eps)
    return (xn * weight.to(torch.float32).to(orig_dtype).to(torch.float32)).to(orig_dtype)


def rms_norm_hf_match(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Bit-exact HF LlamaRMSNorm:
      var = x.pow(2).mean(-1, keepdim=True)
      x = x * rsqrt(var + eps)
      return weight * x.to(input_dtype)
    Internal compute in fp32 with weight in fp32; result cast to input dtype.
    """
    in_dtype = x.dtype
    x = x.to(torch.float32)
    var = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(var + eps)
    out = weight.to(torch.float32) * x
    return out.to(in_dtype)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """HF apply_rotary_pos_emb (rotate_half style). q/k: (B,H,S,D); cos/sin: (B,S,D)."""
    # Expand cos/sin for the head dim broadcasting.
    cos_e = cos.unsqueeze(1)  # (B,1,S,D)
    sin_e = sin.unsqueeze(1)
    def rotate_half(t):
        x1 = t[..., : t.shape[-1] // 2]
        x2 = t[..., t.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    q_emb = (q * cos_e) + (rotate_half(q) * sin_e)
    k_emb = (k * cos_e) + (rotate_half(k) * sin_e)
    return q_emb, k_emb


def silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return F.silu(gate) * up


def block_prefill(x, cos, sin, mask, w):
    """Run the full block on a prefill input. Returns (out, k_after, v_after).

    `mask` is additive (S, S); we'll broadcast to (1,1,S,S) inside.
    `w` is a dict of weights matching the keys in dump_block_case.py (already
    transposed for A @ B).
    """
    B, S, _ = x.shape
    # Residual 1.
    r1 = x
    x_n = rms_norm_hf_match(x, w["in_norm"], EPS)
    q = (x_n @ w["qw"]).view(B, S, N_HEADS, HEAD_DIM).transpose(1, 2)  # (B,H,S,D)
    k = (x_n @ w["kw"]).view(B, S, N_HEADS, HEAD_DIM).transpose(1, 2)
    v = (x_n @ w["vw"]).view(B, S, N_HEADS, HEAD_DIM).transpose(1, 2)
    q, k = apply_rope(q, k, cos, sin)
    # Eager SDPA.
    logits = (q @ k.transpose(-1, -2)) * SCALE
    logits = logits + mask.view(1, 1, S, S)
    probs = torch.softmax(logits.float(), dim=-1).to(q.dtype)
    attn = probs @ v  # (B,H,S,D)
    attn = attn.transpose(1, 2).contiguous().view(B, S, HIDDEN)
    attn = attn @ w["ow"]
    x = r1 + attn

    # Residual 2.
    r2 = x
    x_n = rms_norm_hf_match(x, w["post_norm"], EPS)
    gate = x_n @ w["gate_w"]
    up = x_n @ w["up_w"]
    hidden = silu_mul(gate, up)
    mlp_out = hidden @ w["down_w"]
    x = r2 + mlp_out

    return x, k, v   # k, v are post-RoPE, BHSD layout


def block_decode(x, cos, sin, k_hist, v_hist, w):
    """Run the block on a single new token, attending to (k_hist, v_hist) plus
    the new K/V. Returns out (B,1,H).
    """
    B = x.shape[0]
    r1 = x
    x_n = rms_norm_hf_match(x, w["in_norm"], EPS)
    q = (x_n @ w["qw"]).view(B, 1, N_HEADS, HEAD_DIM).transpose(1, 2)
    k_new = (x_n @ w["kw"]).view(B, 1, N_HEADS, HEAD_DIM).transpose(1, 2)
    v_new = (x_n @ w["vw"]).view(B, 1, N_HEADS, HEAD_DIM).transpose(1, 2)
    q, k_new = apply_rope(q, k_new, cos, sin)
    k_all = torch.cat([k_hist, k_new], dim=2)  # (B,H,T+1,D)
    v_all = torch.cat([v_hist, v_new], dim=2)
    logits = (q @ k_all.transpose(-1, -2)) * SCALE  # (B,H,1,T+1)
    probs = torch.softmax(logits.float(), dim=-1).to(q.dtype)
    attn = probs @ v_all  # (B,H,1,D)
    attn = attn.transpose(1, 2).contiguous().view(B, 1, HIDDEN)
    attn = attn @ w["ow"]
    x = r1 + attn

    r2 = x
    x_n = rms_norm_hf_match(x, w["post_norm"], EPS)
    gate = x_n @ w["gate_w"]
    up = x_n @ w["up_w"]
    hidden = silu_mul(gate, up)
    mlp_out = hidden @ w["down_w"]
    x = r2 + mlp_out
    return x


def load_weights(state: dict, dtype: torch.dtype) -> dict:
    pfx = "layers.0."
    return {
        "in_norm": state[pfx + "input_layernorm.weight"].to(dtype),
        "post_norm": state[pfx + "post_attention_layernorm.weight"].to(dtype),
        "qw": state[pfx + "self_attn.q_proj.weight"].t().contiguous().to(dtype),
        "kw": state[pfx + "self_attn.k_proj.weight"].t().contiguous().to(dtype),
        "vw": state[pfx + "self_attn.v_proj.weight"].t().contiguous().to(dtype),
        "ow": state[pfx + "self_attn.o_proj.weight"].t().contiguous().to(dtype),
        "gate_w": state[pfx + "mlp.gate_proj.weight"].t().contiguous().to(dtype),
        "up_w": state[pfx + "mlp.up_proj.weight"].t().contiguous().to(dtype),
        "down_w": state[pfx + "mlp.down_proj.weight"].t().contiguous().to(dtype),
    }


def make_causal_mask(seq: int, dtype: torch.dtype) -> torch.Tensor:
    finfo = torch.finfo(dtype)
    m = torch.zeros(seq, seq, dtype=dtype)
    m.masked_fill_(torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1), finfo.min)
    return m


def dump_for_dtype(state: dict, acts_np: np.ndarray, dtype: torch.dtype, tag: str) -> None:
    """tag is 'fp32' or 'bf16'."""
    # Pick first 16 tokens of input_embeds — same source as test_forward.
    x_all = torch.from_numpy(acts_np.astype(np.float32))[:, :T_TOTAL, :].to(dtype)
    assert x_all.shape == (BATCH, T_TOTAL, HIDDEN)
    x_prefill = x_all[:, :T_HIST, :]
    x_decode = x_all[:, T_HIST:T_TOTAL, :]  # (1, 1, 1024)

    cfg = LlamaConfig(
        vocab_size=8, hidden_size=HIDDEN, intermediate_size=INTERMEDIATE,
        num_attention_heads=N_HEADS, num_key_value_heads=N_HEADS, head_dim=HEAD_DIM,
        num_hidden_layers=30, rms_norm_eps=EPS,
        rope_theta=500_000.0,
        rope_scaling=dict(
            factor=8.0, high_freq_factor=4.0, low_freq_factor=1.0,
            original_max_position_embeddings=8192, rope_type="llama3",
        ),
        max_position_embeddings=131072,
        mlp_bias=False, attention_bias=False, hidden_act="silu",
        attn_implementation="eager",
    )
    rope = LlamaRotaryEmbedding(cfg).eval()
    dummy = torch.zeros(BATCH, T_TOTAL, HEAD_DIM, dtype=dtype)
    positions_prefill = torch.arange(T_HIST, dtype=torch.long).unsqueeze(0)
    positions_decode = torch.tensor([[T_HIST]], dtype=torch.long)  # position 15
    with torch.no_grad():
        cos_prefill, sin_prefill = rope(dummy[:, :T_HIST, :], positions_prefill)
        cos_decode, sin_decode = rope(dummy[:, :1, :], positions_decode)

    w = load_weights(state, dtype)
    mask_prefill = make_causal_mask(T_HIST, dtype)

    with torch.no_grad():
        _, k_hist, v_hist = block_prefill(x_prefill, cos_prefill, sin_prefill, mask_prefill, w)
        out_decode = block_decode(x_decode, cos_decode, sin_decode, k_hist, v_hist, w)

    def transform(arr_fp32: np.ndarray) -> np.ndarray:
        if tag == "fp32":
            return arr_fp32.astype(np.float32)
        return fp32_to_bf16_bits(arr_fp32).reshape(arr_fp32.shape)

    write_tensor(OUT / f"x_prefill_{tag}.bin", transform(x_prefill.float().numpy()))
    write_tensor(OUT / f"x_decode_{tag}.bin", transform(x_decode.float().numpy()))
    write_tensor(OUT / f"cos_prefill_{tag}.bin", transform(cos_prefill.float().numpy()))
    write_tensor(OUT / f"sin_prefill_{tag}.bin", transform(sin_prefill.float().numpy()))
    write_tensor(OUT / f"cos_decode_{tag}.bin", transform(cos_decode.float().numpy()))
    write_tensor(OUT / f"sin_decode_{tag}.bin", transform(sin_decode.float().numpy()))
    if tag == "fp32":
        write_tensor(OUT / f"mask_prefill_{tag}.bin", mask_prefill.float().numpy().astype(np.float32))
    else:
        write_tensor(OUT / f"mask_prefill_{tag}.bin",
                     fp32_to_bf16_bits(mask_prefill.float().numpy()).reshape(T_HIST, T_HIST))
    write_tensor(OUT / f"k_hist_{tag}.bin", transform(k_hist.float().numpy()))
    write_tensor(OUT / f"v_hist_{tag}.bin", transform(v_hist.float().numpy()))

    for name in ("in_norm", "post_norm", "qw", "kw", "vw", "ow", "gate_w", "up_w", "down_w"):
        wt = w[name].float().numpy()
        write_tensor(OUT / f"{name}_{tag}.bin", transform(wt))

    write_tensor(OUT / f"expected_{tag}.bin", transform(out_decode.float().numpy()))

    print(f"[{tag}] x_prefill {tuple(x_prefill.shape)} k_hist {tuple(k_hist.shape)} expected {tuple(out_decode.shape)}")
    print(f"  expected[0,0,:4] = {out_decode.float()[0,0,:4].numpy()}")


def main() -> None:
    state = load_file(str(WEIGHTS))
    acts = np.load(ACTS)
    # input_embeds is (1, 16, 1024) — exactly enough for prefill(15) + decode(1).
    dump_for_dtype(state, acts["input_embeds"], torch.float32, "fp32")
    dump_for_dtype(state, acts["input_embeds"], torch.bfloat16, "bf16")


if __name__ == "__main__":
    main()
